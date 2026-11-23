"""Subtitle tidy-up: every video's English subtitles as plain SRT files beside it.

Films and non-anime TV get one file, ``<video>.en.srt``. Anime gets at most two:
``<video>.en.srt`` matches the original (Japanese) audio, and ``<video>.en.cc.srt`` matches the
English dub (only when the file has English audio). Each anime file is complete: the
signs-and-songs lines are merged in at the top of the screen (``{\\an8}``), collapsed to one cue
per on-screen text instead of one per animation frame. Plain SRT plays in every app, so Plex
never has to burn subtitles in.

Sources, best first: an existing sidecar, an embedded English text track (extracted; video files
are never changed), then Bazarr (which ends with SpeechDock's Whisper provider). Picture (PGS) signs
are read by OCR. Which embedded track matches the dub comes from the dub caption ledger (measured
agreement with the English audio, 0.80 or more), never from track titles.

Nothing is deleted. A replaced or unwanted file (other languages, duplicates, numbered copies,
forced-only, an anime track that matches neither audio) moves to the quarantine folder, and every
change is journalled in the job's report so it can be undone.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

LIBRARIES: tuple[str, ...] = ("/plex-media/tv", "/plex-media/movies")  # replaced by the media mounts at start-up
OCR_DIR = Path(os.environ.get("SPDOCK_DATA_DIR", "/data")) / "ocr-cache"
VIDEO_EXT = {".mkv", ".mp4", ".m4v", ".avi", ".mov", ".ts", ".m2ts", ".wmv", ".mpg", ".mpeg", ".webm"}
SUB_EXT = {".srt", ".ass", ".ssa", ".vtt"}  # picture sidecars (.sub/.idx/.sup) are left alone; there are none today
TEXT_CODECS = {"subrip": "srt", "srt": "srt", "ass": "ass", "ssa": "ass", "mov_text": "srt", "webvtt": "srt", "text": "srt"}
PICTURE_CODECS = {"hdmv_pgs_subtitle", "pgssub", "dvd_subtitle", "dvb_subtitle", "vobsub"}
ENGLISH = {"en", "eng", "english"}
NO_LANGUAGE = {"", "und", "zxx", "mis", "mul"}
DUB_AGREEMENT = 0.80  # the dub caption job's threshold (captions/pipeline.py), measured on this library
DUB_MISMATCH = "dub caption that does not match the dub audio"
MIN_CUES = 30  # an extracted dialogue track with fewer cues is a signs or forced track
MIN_COVERAGE = 0.5  # an extracted dialogue track must run past half the video
ENGLISH_RATIO = 0.2  # share of common English words that marks untagged text as English

SIGNS_TITLE = re.compile(r"sign|song|forced|s&s|lyric|karaoke", re.I)
FULL_TITLE = re.compile(r"full|dialog|main|complete", re.I)
HINT_TITLE = re.compile(r"sdh|\bcc\b|hearing|closed|dub", re.I)  # only orders plain tracks first, never decides
COMMENTARY = re.compile(r"commentar|director", re.I)
# Style and effect rules proven on this library's ASS releases (SpeechDock's parser, 2026-11-03 sign-flood fix).
SIGN_STYLE = re.compile(r"sign|song|kara|^op|^ed\b|lyric|title|note|^ts|typeset|insert|screen|text", re.I)
DIALOGUE_STYLE = re.compile(r"default|main|dialog|italic|flashback|top|alt|overlap|internal|narrat|thought", re.I)
POSITIONED = re.compile(r"\\(?:pos|move|org|i?clip)\(|\\[kK][fo]?\d")
DRAWING = re.compile(r"\\p[1-9]")
TOP = re.compile(r"\\an[789]|\\a(?:5|6|7|9|10|11)\b")
STAMP = re.compile(r"^\s*((?:\d+:)?\d{1,2}:\d{2}[,.]\d{1,3})\s*-->\s*((?:\d+:)?\d{1,2}:\d{2}[,.]\d{1,3})")
COMMON = set("the you i to a and it is of that in what we me this he for my your on have be was not with do are "
              "just know no but all so can get like they she her him there here go up out about don't i'm it's "
              "if right yeah oh come one did will at want now why how who okay well".split())


# ---------------------------------------------------------------------------------------------- cues


@dataclass(frozen=True)
class Cue:
    start: int  # ms
    end: int
    text: str  # as shown; a sign starts with {\an8}
    sign: bool = False


def norm(text: str) -> str:
    return " ".join(re.sub(r"\{[^}]*\}|<[^>]*>", "", text).lower().split())


def decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _ms(stamp: str) -> int:
    parts = stamp.replace(",", ".").split(":")
    seconds, _, fraction = parts[-1].partition(".")
    hours = int(parts[-3]) if len(parts) > 2 else 0
    return ((hours * 60 + int(parts[-2])) * 60 + int(seconds)) * 1000 + int(fraction.ljust(3, "0")[:3])


def parse_srt(text: str, all_signs: bool = False) -> list[Cue]:
    """SRT/WebVTT blocks, text kept as written (tags included) so a rewrite reproduces it."""
    cues, lines, i = [], text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), 0
    while i < len(lines):
        match = STAMP.match(lines[i])
        i += 1
        if not match:
            continue
        body = []
        while i < len(lines) and lines[i].strip():
            body.append(lines[i].rstrip())
            i += 1
        start, end = _ms(match.group(1)), _ms(match.group(2))
        content = "\n".join(body).strip()
        if not norm(content) or end <= start:
            continue
        if all_signs and not content.startswith("{\\an"):
            content = "{\\an8}" + content
        cues.append(Cue(start, end, content, all_signs))
    return cues


def _ass_text(body: str) -> str:
    text = re.sub(r"\{[^}]*\}", "", body).replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
    return "\n".join(" ".join(line.split()) for line in html.unescape(text).split("\n") if line.strip())


def parse_ass(text: str, all_signs: bool = False) -> list[Cue]:
    """Events of an ASS/SSA script; signs (by style, position or karaoke) are marked, drawings dropped."""
    styles: dict[str, int] = {}
    style_fields = event_fields = None
    cues = []
    for line in text.splitlines():
        if line.startswith("Format:"):
            names = [name.strip() for name in line[7:].split(",")]
            if "Start" in names and "Text" in names:
                event_fields = names
            elif "Name" in names and "Alignment" in names:
                style_fields = names
            continue
        if line.startswith("Style:") and style_fields:
            values = dict(zip(style_fields, line[6:].split(",")))
            try:
                styles[values["Name"].strip()] = int(values.get("Alignment", "2"))
            except ValueError:
                pass
            continue
        if not line.startswith("Dialogue:") or not event_fields:
            continue
        values = dict(zip(event_fields, line[9:].split(",", len(event_fields) - 1)))
        style, body = values.get("Style", "").strip(), values.get("Text", "")
        if DRAWING.search(body):
            continue
        try:
            start, end = _ms(values["Start"].strip()), _ms(values["End"].strip())
        except (KeyError, ValueError, IndexError):
            continue
        content = _ass_text(body)
        if not content or end <= start:
            continue
        dialogue_style = bool(DIALOGUE_STYLE.search(style))
        # Positioned or karaoke text is on-screen text whatever its style (JJK's title cards use a "Main" style).
        sign = all_signs or bool(POSITIONED.search(body)) or (bool(SIGN_STYLE.search(style)) and not dialogue_style)
        top = sign or bool(TOP.search(body)) or styles.get(style, 2) in (7, 8, 9)
        cues.append(Cue(start, end, ("{\\an8}" if top else "") + content, sign))
    return cues


def parse_file(data: bytes, kind: str, all_signs: bool = False) -> list[Cue]:
    text = decode(data)
    if kind in ("ass", "ssa") or text.lstrip().startswith("[Script Info]"):
        return parse_ass(text, all_signs)
    return parse_srt(text, all_signs)


def collapse(cues: list[Cue]) -> list[Cue]:
    """One cue per run of the same text: ASS layers and per-frame sign animation become one cue."""
    groups: dict[tuple[bool, str], list[Cue]] = {}
    for cue in cues:
        groups.setdefault((cue.sign, norm(cue.text)), []).append(cue)
    out = []
    for (sign, key), items in groups.items():
        items.sort(key=lambda c: (c.start, c.end))
        current = items[0]
        for cue in items[1:]:
            if cue.start <= current.end + 150:
                current = Cue(current.start, max(current.end, cue.end), current.text, sign)
            else:
                out.append(current)
                current = cue
        out.append(current)
    if any(c.sign for c in out):
        out = [c for c in out if not c.sign or (len(norm(c.text)) >= 3 and c.end - c.start >= 250)]
    return sorted(out, key=lambda c: (c.start, c.end, c.text))


def merge_signs(base: list[Cue], signs: list[Cue]) -> list[Cue]:
    """``base`` plus every sign it does not already show at about the same time."""
    seen: dict[str, list[tuple[int, int]]] = {}
    for cue in base:
        seen.setdefault(norm(cue.text), []).append((cue.start, cue.end))
    out = list(base)
    for sign in signs:
        spans = seen.get(norm(sign.text), [])
        if any(start < sign.end + 1000 and sign.start < end + 1000 for start, end in spans):
            continue
        out.append(sign)
        seen.setdefault(norm(sign.text), []).append((sign.start, sign.end))
    return collapse(out)


def render(cues: list[Cue]) -> str:
    def stamp(ms: int) -> str:
        return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"
    return "".join(f"{n}\n{stamp(c.start)} --> {stamp(c.end)}\n{c.text}\n\n" for n, c in enumerate(cues, 1))


def english_ratio(cues: list[Cue]) -> float:
    words = [w for c in cues for w in re.findall(r"[a-z']+", norm(c.text))]
    return sum(w in COMMON for w in words) / len(words) if len(words) >= 100 else 0.0


# ---------------------------------------------------------------------------------------------- media


@dataclass
class Track:
    index: int
    codec: str
    lang: str
    title: str
    forced: bool
    picture: bool
    signs: bool
    agreement: float | None = None

    @property
    def text(self) -> bool:
        return self.codec in TEXT_CODECS


@dataclass
class Side:
    path: Path
    tokens: list[str]
    lang: str | None  # 'en', another code, or None when the name has none
    forced: bool
    cc: bool
    numbered: bool
    agreement: float | None = None
    cues: list[Cue] | None = None

    @property
    def kind(self) -> str:
        return "ass" if self.path.suffix.lower() in (".ass", ".ssa") else "srt"



def probe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=index,codec_type,codec_name:stream_tags=language,title:stream_disposition=forced,comment,visual_impaired"
         ":format=duration,start_time", "-of", "json", str(video)], capture_output=True, text=True, timeout=300)
    if out.returncode:
        raise RuntimeError("ffprobe failed: " + out.stderr.strip()[-300:])
    return json.loads(out.stdout)


def tag(stream: dict, name: str) -> str:
    return next((str(v) for k, v in (stream.get("tags") or {}).items() if k.lower() == name), "")


def lang_of(stream: dict) -> str:
    return tag(stream, "language").strip().lower().split("-")[0]


def english_audio(streams: list[dict]) -> bool:
    return any(s.get("codec_type") == "audio" and lang_of(s) in ENGLISH and not (s.get("disposition") or {}).get("comment")
               and not (s.get("disposition") or {}).get("visual_impaired") and not COMMENTARY.search(tag(s, "title"))
               for s in streams)


def tracks_of(streams: list[dict]) -> list[Track]:
    result = []
    for s in streams:
        if s.get("codec_type") != "subtitle":
            continue
        lang = lang_of(s)
        if lang not in ENGLISH and lang not in NO_LANGUAGE:
            continue
        title = tag(s, "title")
        forced = bool((s.get("disposition") or {}).get("forced"))
        signs = forced or (bool(SIGNS_TITLE.search(title)) and not FULL_TITLE.search(title))
        codec = (s.get("codec_name") or "").lower()
        if codec not in TEXT_CODECS and codec not in PICTURE_CODECS:
            continue
        result.append(Track(s["index"], codec, "en" if lang in ENGLISH else "", title, forced, codec in PICTURE_CODECS, signs))
    return result


def sidecars(video: Path, stems: list[str]) -> list[Side]:
    """Subtitle files that belong to ``video`` (the longest video name in the folder that prefixes them)."""
    found = []
    try:
        entries = list(os.scandir(video.parent))
    except OSError:
        return found
    for entry in entries:
        name = entry.name
        suffix = Path(name).suffix.lower()
        if suffix not in SUB_EXT or not entry.is_file() or name.startswith("."):
            continue
        owner = max((s for s in stems if name.startswith(s + ".")), key=len, default=None)
        if owner != video.stem:
            continue
        tokens = [t.lower() for t in name[len(video.stem) + 1:-len(suffix)].split(".") if t]
        lang = None
        for t in tokens:
            if t in ("cc", "sdh", "hi", "forced", "default", "full", "sub", "subs") or t.isdigit():
                continue
            if re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,4})?", t) or t in ("english",):
                lang = "en" if t.split("-")[0] in ENGLISH else t
                break
        found.append(Side(Path(entry.path), tokens, lang, "forced" in tokens, bool({"cc", "sdh", "hi"} & set(tokens)),
                          any(t.isdigit() for t in tokens)))
    return sorted(found, key=lambda s: s.path.name)


def record_valid(video: Path, record: dict | None) -> dict | None:
    """The dub caption ledger's record, only while it describes this exact file."""
    if not record:
        return None
    stat = video.stat()
    if record.get("video_size") != stat.st_size or record.get("video_mtime_ns") != stat.st_mtime_ns:
        return None
    return record


def measured_as_is(path: Path, cand: dict, record: dict) -> bool:
    """Whether a sidecar is still the file the ledger scored. Bazarr replaces subtitles under the same name, and a
    score taken from the earlier file then describes different text (fingerprint; older records: not modified since)."""
    if cand.get("sha256"):
        return hashlib.sha256(path.read_bytes()).hexdigest() == cand["sha256"]
    measured = record.get("updated_at")
    return not measured or path.stat().st_mtime <= datetime.fromisoformat(measured).timestamp()


# ---------------------------------------------------------------------------------------------- decide


@dataclass
class Plan:
    video: str
    anime: bool
    english_audio: bool
    duration: float
    orig: dict | None = None  # {"sidecar": path} | {"stream": index, "codec": c}
    dub: dict | None = None
    signs: list[dict] = field(default_factory=list)
    rename: list[tuple[str, str]] = field(default_factory=list)
    quarantine: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)
    pending: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    ocr: list[int] = field(default_factory=list)  # picture tracks to OCR (signs)


def ocr_cache(video: Path, index: int) -> Path:
    """Where the OCR'd text of one picture track of this exact file is kept."""
    stat = video.stat()
    key = hashlib.sha1(f"{video}|{stat.st_size}|{stat.st_mtime_ns}|{index}".encode()).hexdigest()[:20]
    return OCR_DIR / f"{key}.srt"


def ocr_clean(cues: list[Cue]) -> list[Cue]:
    """OCR'd signs: drop lines that came out as symbols rather than words."""
    kept = []
    for cue in cues:
        plain = norm(cue.text).replace(" ", "")
        if len(plain) >= 3 and sum(ch.isalpha() for ch in plain) / len(plain) >= 0.6:
            kept.append(cue)
    return kept


def ocr_dialogue(path: Path) -> list[Cue]:
    """An OCR'd picture dialogue track from the cache, symbol-only lines dropped."""
    return ocr_clean(parse_srt(path.read_text(errors="replace")))


def _rank(track: Track) -> tuple:
    return (bool(HINT_TITLE.search(track.title)), not FULL_TITLE.search(track.title), track.index)


def decide(video: Path, info: dict, sides: list[Side], *, anime: bool, record: dict | None) -> Plan:
    streams = info.get("streams", [])
    duration = float((info.get("format") or {}).get("duration") or 0)
    plan = Plan(str(video), anime, english_audio(streams), duration)
    tracks = tracks_of(streams)
    rec = record_valid(video, record)
    if rec:
        for cand in (rec.get("details") or {}).get("candidates") or []:
            if cand.get("source") in ("embedded", "ocr"):
                for t in tracks:
                    if t.index == cand.get("stream"):
                        t.agreement = cand.get("agreement")
            elif cand.get("source") == "sidecar":
                for s in sides:
                    if s.path.name == cand.get("file") and measured_as_is(s.path, cand, rec):
                        s.agreement = cand.get("agreement")

    target_orig = video.with_name(video.stem + ".en.srt")
    target_dub = video.with_name(video.stem + ".en.cc.srt")
    for s in sides:
        if s.lang is None:
            s.cues = parse_file(s.path.read_bytes(), s.kind)
            s.lang = "en" if english_ratio(s.cues) >= ENGLISH_RATIO else "?"
    english = [s for s in sides if s.lang == "en" and not s.forced]
    for s in sides:
        if s.lang != "en":
            plan.quarantine.append((str(s.path), "not English" if s.lang != "?" else "language unknown, not English text"))
        elif s.forced:
            plan.quarantine.append((str(s.path), "forced-only"))
    taken: set[Path] = set()

    def is_dub(item) -> bool | None:
        return None if item.agreement is None else item.agreement >= DUB_AGREEMENT

    dialogue_text = [t for t in tracks if not t.signs and t.text]
    dialogue_picture = [t for t in tracks if not t.signs and t.picture]
    signs_text = [t for t in tracks if t.signs and t.text]
    signs_picture = [t for t in tracks if t.signs and t.picture]

    # ---- the dub-matched file (anime with English audio only)
    if anime and plan.english_audio:
        cc = next((s for s in english if s.path == target_dub), None)
        dub_side = next((s for s in english if s.path != target_dub and is_dub(s)), None)
        dub_track = next((t for t in sorted(dialogue_text, key=lambda t: -(t.agreement or 0)) if is_dub(t)), None)
        dub_picture = next((t for t in dialogue_picture if is_dub(t) and ocr_cache(video, t.index).exists()), None)
        if cc and is_dub(cc) is not False:
            plan.dub = {"sidecar": str(cc.path)}
            taken.add(cc.path)
        elif dub_side:
            plan.dub = {"sidecar": str(dub_side.path)}
            taken.add(dub_side.path)
        elif dub_track:
            plan.dub = {"stream": dub_track.index, "codec": dub_track.codec}
        elif dub_picture:
            plan.dub = {"ocr": str(ocr_cache(video, dub_picture.index)), "ocr_stream": dub_picture.index}
        elif rec is None:
            plan.pending.append("speechdock")  # no measurement yet: SpeechDock writes or finds the dub caption
        else:
            # Measured, but nothing matches the dub (now, or under an older threshold): measure again, so SpeechDock
            # writes its own dub caption once the non-matching file has gone to the quarantine.
            plan.pending.append("speechdock-dub")

    # ---- the original-matched file (every video)
    def orig_ok(item) -> bool:
        return not (anime and plan.english_audio and is_dub(item))

    side_orig = next((s for s in english if s.path == target_orig and s.path not in taken and orig_ok(s)), None)
    if side_orig is None:
        others = [s for s in english if s.path not in taken and s.path != target_dub and orig_ok(s)]
        others.sort(key=lambda s: (s.cc, s.numbered, len(s.tokens), s.path.name))
        side_orig = others[0] if others else None
    if not anime and side_orig is None:
        # A film's or show's own dub caption follows the dub, not the original: it stays only until Bazarr
        # brings a real translation, and is never renamed to .en.srt (Bazarr would then stop looking).
        cc = next((s for s in english if s.path == target_dub and s.path not in taken), None)
        if cc and not [t for t in dialogue_text if is_dub(t) is not True]:
            taken.add(cc.path)
            plan.notes.append("only a dub caption exists; kept until Bazarr finds a translation")
    track_orig = next((t for t in sorted(dialogue_text, key=_rank) if t.lang == "en" and is_dub(t) is not True), None)
    if side_orig:
        plan.orig = {"sidecar": str(side_orig.path)}
        taken.add(side_orig.path)
    elif track_orig:
        plan.orig = {"stream": track_orig.index, "codec": track_orig.codec}
    else:
        und = [t for t in dialogue_text if t.lang == ""]
        if und:
            plan.orig = {"stream": und[0].index, "codec": und[0].codec, "check_english": True}
        else:
            # Picture-only English dialogue: OCR it; for an anime with an English dub, SpeechDock must first
            # measure the OCR'd text against the dub audio, so a dub script never becomes the translation.
            for t in sorted((t for t in dialogue_picture if t.lang == "en"), key=_rank):
                cache = ocr_cache(video, t.index)
                if not cache.exists():
                    plan.ocr.append(t.index)  # all of them now: tidy() decides again once they are read
                    plan.pending.append("ocr-dialogue")
                    continue
                if is_dub(t) or not coverage_ok(ocr_dialogue(cache), duration):
                    continue
                if plan.ocr:
                    break  # an earlier-ranked track is still being read
                if anime and plan.english_audio and t.agreement is None:
                    plan.pending.append("speechdock-ocr")
                else:
                    plan.orig = {"ocr": str(cache), "ocr_stream": t.index}
                break
            if plan.orig is None:
                plan.pending.append("bazarr")

    # ---- signs (anime only): a signs track, else the sign events inside a styled dialogue track
    if anime:
        if signs_text:
            best = sorted(signs_text, key=lambda t: (t.codec not in ("ass", "ssa"), t.index))[0]
            plan.signs.append({"stream": best.index, "codec": best.codec, "all": True})
        else:
            styled = sorted((t for t in dialogue_text if t.codec in ("ass", "ssa")),
                            key=lambda t: (is_dub(t) is True, not FULL_TITLE.search(t.title), t.index))
            for t in styled:
                plan.signs.append({"stream": t.index, "codec": t.codec, "all": False})
                break
            else:
                english_pictures = [t for t in signs_picture if t.lang == "en"] or signs_picture
                if english_pictures:
                    cache = ocr_cache(video, english_pictures[0].index)
                    if cache.exists():
                        plan.signs.append({"ocr": str(cache), "all": True})
                    else:
                        plan.ocr.append(english_pictures[0].index)
                        plan.pending.append("ocr-signs")

    # ---- everything else English beside the video
    for s in english:
        if s.path in taken:
            continue
        reason = "duplicate English subtitle"
        if anime and plan.english_audio and s.path == target_dub and plan.dub is None:
            reason = DUB_MISMATCH
        plan.quarantine.append((str(s.path), reason))
    for item, target in ((plan.orig, target_orig), (plan.dub, target_dub)):
        if not item or not item.get("sidecar") or item["sidecar"] == str(target):
            continue
        if Path(item["sidecar"]).suffix.lower() == ".srt":
            plan.rename.append((item["sidecar"], str(target)))
        else:
            item["convert"] = True  # an ASS/WebVTT file is rewritten as SRT, the original kept in quarantine
            plan.quarantine.append((item["sidecar"], "converted to SRT"))
    return plan


# ---------------------------------------------------------------------------------------------- apply


class Journal:
    """Every change of one job: files moved to the quarantine and files written, in order."""

    def __init__(self, quarantine: Path):
        self.quarantine = quarantine
        self.entries: list[dict] = []

    def log(self, **entry):
        self.entries.append({**entry, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})

    def stash(self, path: Path, reason: str) -> None:
        root = next(lib for lib in LIBRARIES if str(path).startswith(lib + "/"))
        dest = self.quarantine / root.strip("/") / str(path)[len(root) + 1:]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest = dest.with_name(f"{dest.stem}.{int(time.time() * 1000)}{dest.suffix}")
        shutil.move(str(path), str(dest))
        self.log(op="move", src=str(path), dst=str(dest), reason=reason)

    def write(self, path: Path, data: bytes, reason: str) -> None:
        tmp = path.with_name(f".{path.name}.speechdock.tmp")
        tmp.write_bytes(data)
        os.chmod(tmp, 0o666)
        if path.exists():
            self.stash(path, "replaced: " + reason)
        os.replace(tmp, path)
        self.log(op="create", path=str(path), sha256=hashlib.sha256(data).hexdigest(), reason=reason)

    def rename(self, src: Path, dst: Path) -> None:
        if dst.exists():
            self.stash(dst, "replaced by " + src.name)
        os.replace(src, dst)
        self.log(op="move", src=str(src), dst=str(dst), reason="renamed to the standard name")


def extract(video: Path, wanted: list[dict], work: Path) -> dict[int, Path]:
    """One read of the file for every embedded text track needed."""
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video)]
    outputs = {}
    for item in wanted:
        kind = TEXT_CODECS[item["codec"]]
        out = work / f"s{item['stream']}.{kind}"
        command += ["-map", f"0:{item['stream']}", "-c:s", "ass" if kind == "ass" else "srt", str(out)]
        outputs[item["stream"]] = out
    result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    if result.returncode:
        raise RuntimeError("ffmpeg failed: " + result.stderr.strip()[-300:])
    return {k: v for k, v in outputs.items() if v.exists() and v.stat().st_size}


def coverage_ok(cues: list[Cue], duration: float) -> bool:
    dialogue = [c for c in cues if not c.sign]
    return len(dialogue) >= MIN_CUES and (not duration or max(c.end for c in dialogue) >= duration * 1000 * MIN_COVERAGE)


def apply_plan(plan: Plan, journal: Journal) -> dict:
    video = Path(plan.video)
    outcome = {"written": [], "renamed": [], "quarantined": [], "pending": list(plan.pending), "errors": []}
    target_orig = video.with_name(video.stem + ".en.srt")
    target_dub = video.with_name(video.stem + ".en.cc.srt")
    wanted = [x for x in (plan.orig, plan.dub, *plan.signs) if x and "stream" in x]
    streams: dict[int, list[Cue]] = {}
    if wanted:
        with tempfile.TemporaryDirectory(prefix="speechdock-tidy-") as tmp:
            try:
                files = extract(video, list({w["stream"]: w for w in wanted}.values()), Path(tmp))
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                outcome["errors"].append(str(exc))
                return outcome
            for item in wanted:
                path = files.get(item["stream"])
                streams[item["stream"]] = parse_file(path.read_bytes(), TEXT_CODECS[item["codec"]], item.get("all", False)) if path else []

    def base_cues(item: dict | None) -> list[Cue] | None:
        if not item:
            return None
        if "sidecar" in item:
            path = Path(item["sidecar"])
            return parse_file(path.read_bytes(), "ass" if path.suffix.lower() in (".ass", ".ssa") else "srt")
        cues = ocr_dialogue(Path(item["ocr"])) if "ocr" in item else streams.get(item["stream"], [])
        if item.get("check_english") and english_ratio(cues) < ENGLISH_RATIO:
            return None
        return cues if coverage_ok(cues, plan.duration) else None

    signs: list[Cue] = []
    if plan.anime:
        for item in plan.signs:
            if "ocr" in item:
                signs += ocr_clean(parse_srt(Path(item["ocr"]).read_text(errors="replace"), all_signs=True))
            else:
                signs += [c for c in streams.get(item["stream"], []) if c.sign]
        signs = collapse(signs)

    # Renames first, so a stash of the target never takes the source.
    for src, dst in plan.rename:
        if Path(src).exists():
            journal.rename(Path(src), Path(dst))
            outcome["renamed"].append([src, dst])
    for item, target in ((plan.orig, target_orig), (plan.dub, target_dub)):
        if not item:
            continue
        converting = bool(item.get("convert"))
        if "sidecar" in item and not converting:
            if not plan.anime:
                continue  # a film's or show's existing subtitle is left exactly as it is
            item = {"sidecar": str(target)}
        cues = base_cues(item)
        if cues is None:
            outcome["pending"].append("bazarr" if target == target_orig else "speechdock")
            continue
        result = merge_signs(collapse(cues), signs) if plan.anime else collapse(cues)
        data = render(result).encode("utf-8")
        if target.exists() and target.read_bytes() == data:
            continue
        reason = (f"converted from {Path(item['sidecar']).name}" if converting else "signs merged" if "sidecar" in item
                  else f"OCR of picture stream {item['ocr_stream']}" if "ocr" in item else f"extracted from stream {item['stream']}")
        journal.write(target, data, reason)
        outcome["written"].append(str(target))
    for path, reason in plan.quarantine:
        # The standard names stay, except an anime's .en.cc.srt measured as not matching the dub: it must go, or
        # SpeechDock keeps the foreign file and never writes the dub caption.
        standard = Path(path) in (target_orig, target_dub) and reason != DUB_MISMATCH
        if Path(path).exists() and not standard:
            journal.stash(Path(path), reason)
            outcome["quarantined"].append(path)
    outcome["pending"] = sorted(set(outcome["pending"]))
    return outcome


# ---------------------------------------------------------------------------------------------- one video


def ocr_track(video: Path, index: int) -> Path | None:
    """OCR one picture subtitle track (mkvextract + pgsrip/Tesseract) into the cache; None when it cannot."""
    out = ocr_cache(video, index)
    if out.exists():
        return out
    with tempfile.TemporaryDirectory(prefix="speechdock-ocr-") as tmp:
        sup = Path(tmp) / "track.en.sup"
        # mkvmerge track ids follow the file's track order, as ffprobe's stream indexes do.
        done = subprocess.run(["mkvextract", str(video), "tracks", f"{index}:{sup}"], capture_output=True, text=True, timeout=3600)
        if done.returncode or not sup.exists() or not sup.stat().st_size:
            return None
        subprocess.run(["pgsrip", "rip", "-l", "en", "--force", "-w", "2", "--engine", "tesseract", str(sup)],
                       capture_output=True, text=True, timeout=3600, env={**os.environ, "HOME": tmp})
        srt = next(Path(tmp).glob("*.srt"), None)
        if not srt or not srt.stat().st_size:
            return None
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_suffix(".part")
        shutil.copyfile(srt, part)
        part.replace(out)
        return out


def tidy(video: Path, *, anime: bool, record: dict | None, quarantine: Path) -> dict:
    """Bring one video's subtitles into the standard layout. Blocking (ffprobe, ffmpeg, OCR)."""
    stems = [Path(name).stem for name in os.listdir(video.parent) if Path(name).suffix.lower() in VIDEO_EXT]
    info = probe(video)
    plan = decide(video, info, sidecars(video, stems), anime=anime, record=record)
    ocr_failed = []
    if plan.ocr:
        for index in plan.ocr:
            if ocr_track(video, index) is None:
                ocr_failed.append(index)
        plan = decide(video, info, sidecars(video, stems), anime=anime, record=record)
    journal = Journal(quarantine)
    result = apply_plan(plan, journal)
    if ocr_failed:
        result["errors"].append(f"OCR failed for picture track(s) {ocr_failed}")
    return {"video": plan.video, "anime": plan.anime, "english_audio": plan.english_audio, "orig": plan.orig,
            "dub": plan.dub, "signs": plan.signs, "notes": plan.notes, **result, "journal": journal.entries}


def undo(entries: list[dict]) -> int:
    """Reverse a job's journal: remove the files it wrote (unchanged since) and move every file back."""
    done = 0
    for entry in reversed(entries):
        if entry["op"] == "create":
            path = Path(entry["path"])
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]:
                path.unlink()
                done += 1
        elif entry["op"] == "move":
            src, dst = Path(entry["src"]), Path(entry["dst"])
            if dst.exists() and not src.exists():
                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dst), str(src))
                done += 1
    return done
