"""Quick check: listen to three short clips before reading the whole file.

Reading a video's English audio out of the file is usually most of a caption job (One-Punch Man S02E01:
275 s reading, 13 s transcribing). When the file already carries a real dub caption, three 90-second
clips are enough to see it, and only those parts of the file are read.

The check can only end a job as ``matched_existing``, and only when one subtitle agrees with the audio
at ``PASS`` or better in each of ``NEEDED`` clips. A clip with too little speech (an opening or ending
song, a silent stretch) is not judged; the next position is tried instead, up to ``MAX_CLIPS`` reads.
Anything else - a lower score in any clip, too few clips with speech, a clip that cannot be read - runs
the full job unchanged. The middle clip goes first and a subtitle that fails a clip is not scored
again, so a file with only translations usually costs one clip.
"""
from __future__ import annotations

import asyncio
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ..audio import RATE
from ..engine import Word
from . import agreement, cues as cue_tools, media, windows
from .subtitles import SubCue, parse_subtitle

# Share of the file at each clip's middle, in the order tried. 30-70 % stays clear of anime openings
# (often at 5-15 %) and endings (85-95 %), where a 12 %/85 % layout found too little speech in 13 of 19
# dub-captioned episodes; the outer positions only replace clips without enough speech.
POINTS = (0.50, 0.30, 0.70, 0.40, 0.60, 0.20, 0.80)
NEEDED = 3  # clips that must each agree
MAX_CLIPS = 5  # clips read at most
CLIP_SECONDS = 90.0
EDGE = 2.0  # seconds at each end of a clip in which cues are not scored: the cut may split them
SLACK = 0.5  # heard words this close outside a track's first and last scored cue still count
PASS = 0.85  # per clip; a whole-file match needs 0.75 (pipeline.AGREEMENT_THRESHOLD)
MIN_HEARD = 60  # recognized words a clip, and a track's scored span, must hold to be judged
MIN_DURATION = 900.0  # shorter files are cheap to read whole, and their clips would overlap
MAX_START = 0.5  # seconds; a file whose timeline starts later is read whole (clip cue times would shift)


@dataclass(frozen=True)
class ClipScore:
    agreement: float
    precision: float
    recall: float
    heard: int  # recognized words in the span the track's scored cues cover


def clip_start(duration: float, point: float) -> float:
    return max(0.0, min(duration * point - CLIP_SECONDS / 2, duration - CLIP_SECONDS))


def _tokens(words: list[Word], low: float, high: float) -> list[Word]:
    return [word for word in words if low <= (word.start + word.end) / 2 <= high]


def heard_count(words: list[Word], start: float, end: float) -> int:
    return sum(len(agreement.norm_words(word.text)) for word in _tokens(words, start + EDGE, end - EDGE))


def clip_score(cues: list[SubCue], words: list[Word], start: float, end: float) -> ClipScore:
    """How one subtitle agrees with the words heard in one clip (absolute times).

    Only cues wholly inside the clip, ``EDGE`` seconds clear of its cuts, are scored, against the words
    heard from the first of them to the last: a cue the cut splits counts for neither side, while a gap
    in the track inside that span still costs recall.
    """
    inside = [cue for cue in cues if cue.start_ms >= (start + EDGE) * 1000 and cue.end_ms <= (end - EDGE) * 1000]
    if not inside:
        return ClipScore(0.0, 0.0, 0.0, 0)
    low = min(cue.start_ms for cue in inside) / 1000 - SLACK
    high = max(cue.end_ms for cue in inside) / 1000 + SLACK
    spoken = _tokens(words, low, high)
    result = agreement.score(inside, spoken)
    return ClipScore(result.agreement, result.precision, result.recall, result.spoken_words)


def passes(score: ClipScore) -> bool:
    return score.agreement >= PASS and score.heard >= MIN_HEARD


def applies(duration: float | None, start_time: float | None, tracks: list[dict], sidecars: list[dict]) -> bool:
    """Worth trying: a long enough file with a timeline from zero and at least one subtitle to compare."""
    return bool(duration and duration >= MIN_DURATION and abs(start_time or 0.0) <= MAX_START and (tracks or sidecars))


async def check(*, video: Path, audio: dict, tracks: list[dict], sources: list[dict], duration: float, engine, model: str,
                workdir: Path, progress: Callable[[float], Awaitable[None]]) -> dict:
    """Score every subtitle against the clips in turn; return the match (or why there is none) and a report.

    ``tracks`` are the embedded text tracks (read again for each clip), ``sources`` every subtitle in the
    shape the full job uses: ``{"meta", "cues"}``, where embedded tracks' ``cues`` are not used here.
    """
    by_index = {track["index"]: track for track in tracks}
    live = list(range(len(sources)))
    scores: dict[int, list[ClipScore]] = {i: [] for i in live}
    seen: dict[int, int] = {i: 0 for i in live}
    report = {"matched": False, "reason": "", "clips": []}
    stats, words_total, windows_total, repairs = cue_tools.CleanStats(), 0, 0, 0
    workdir.mkdir(parents=True, exist_ok=True)
    judged = 0
    try:
        for number, point in enumerate(POINTS):
            if number == MAX_CLIPS or judged == NEEDED:
                break
            start = clip_start(duration, point)
            end = start + CLIP_SECONDS
            where = f"{round(point * 100)} %"
            wanted = [by_index[sources[i]["meta"]["stream"]] for i in live if sources[i]["meta"]["source"] == "embedded"]
            try:
                pcm, files = await media.extract_clip(video, audio, wanted, workdir, start, CLIP_SECONDS)
            except media.MediaError as exc:
                report["reason"] = f"The clip at {where} could not be read ({exc}); the whole file was checked."
                break
            bounds = await asyncio.to_thread(windows.window_bounds, pcm)
            heard: list[Word] = []
            for begin, finish in bounds:
                wav = await asyncio.to_thread(windows.read_window, pcm, begin, finish)
                found, fixed = await engine.transcribe_words(wav, model, "en")
                offset = start + begin / RATE
                heard.extend(replace(word, start=word.start + offset, end=word.end + offset) for word in found)
                repairs += fixed
            cleaned, clip_stats = cue_tools.clean_words(heard)
            for field in ("compressed", "unsure", "loops"):
                setattr(stats, field, getattr(stats, field) + getattr(clip_stats, field))
            words_total += len(cleaned)
            windows_total += len(bounds)
            count = heard_count(cleaned, start, end)
            report["clips"].append({"at": where, "start": round(start, 1), "seconds": CLIP_SECONDS, "heard": count})
            await progress((number + 1) / MAX_CLIPS)
            if count < MIN_HEARD:
                continue  # an opening song or a silent stretch: try the next position
            judged += 1
            for i in live:
                meta = sources[i]["meta"]
                if meta["source"] == "embedded":
                    path = files.get(meta["stream"])
                    cues = parse_subtitle(path.read_bytes(), meta["codec"]) if path else []
                else:
                    cues = sources[i]["cues"]
                seen[i] += sum(1 for cue in cues if start * 1000 <= cue.start_ms < end * 1000)
                scores[i].append(clip_score(cues, cleaned, start, end))
            live = [i for i in live if passes(scores[i][-1])]
            if not live:
                report["reason"] = f"No subtitle agrees with the audio in the clip at {where}; the whole file was checked."
                break
            for path in (pcm, *files.values()):
                path.unlink(missing_ok=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    candidates = []
    for i, source in enumerate(sources):
        if not scores[i]:
            continue
        worst = min(scores[i], key=lambda item: item.agreement)
        candidates.append({**source["meta"], "cues": seen[i], "agreement": round(worst.agreement, 3),
                           "precision": round(worst.precision, 3), "recall": round(worst.recall, 3),
                           "clips": [round(item.agreement, 3) for item in scores[i]]})
    matched = None
    if live and judged == NEEDED:
        best = max(live, key=lambda i: min(item.agreement for item in scores[i]))
        matched = next(item for item in candidates if item["label"] == sources[best]["meta"]["label"])
        report.update(matched=True, reason=f"{matched['label']} agrees with the audio in all {NEEDED} clips.")
    elif live and not report["reason"]:
        report["reason"] = (f"Only {judged} of {len(report['clips'])} clips held enough speech to judge; "
                            "the whole file was checked.")
    return {"matched": matched, "candidates": candidates, "report": report,
            "transcript": {"windows": windows_total, "words": words_total, "cues": 0, "dropped": asdict(stats),
                           "timing_repairs": repairs}}
