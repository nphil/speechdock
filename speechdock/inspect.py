"""Audio inspection: what a video's tracks claim to be and what language they really speak.

An import gate (an NZBGet or qBittorrent hook, for example) sends a path from the download folders and
gets every stream back with its tags, plus the language Whisper hears on the audio tracks it asked
about. Nothing here decides whether a file is acceptable; that policy belongs to the caller.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from .audio import SpeechDetector, _pcm_wav, RATE, speech_wav_chunks, tool_failure
from .captions import media
from .captions.media import MediaError
from .engine import Engine, EngineError, EngineUnavailable
from .languages import LANGUAGES, normalize_language

DEFAULT_ROOTS = {"/downloads": "/downloads"}
DETECT_MODES = ("untagged", "all", "none")
REGION_SECONDS = 120  # audio read around each sample point; the best speech window inside it is detected
SAMPLE_POINTS = (0.5, 0.3, 0.7)  # fractions of the duration tried in turn until a sample with speech is found
MAX_DETECTIONS = 8  # audio tracks detected per file at most; the rest keep their tags only
NO_LANGUAGE = frozenset({"", "und", "zxx", "mis", "mul"})

# ISO 639-2 (bibliographic and terminologic) codes as they appear in container tags -> Whisper's codes.
_ISO3 = dict(item.split(":") for item in (
    "eng:en jpn:ja por:pt fre:fr fra:fr ger:de deu:de spa:es ita:it chi:zh zho:zh cmn:zh yue:yue kor:ko rus:ru "
    "dut:nl nld:nl swe:sv nor:no nob:no nno:nn dan:da fin:fi pol:pl tur:tr ara:ar hin:hi tha:th vie:vi ces:cs cze:cs "
    "hun:hu heb:he ell:el gre:el ron:ro rum:ro ukr:uk ind:id msa:ms may:ms tam:ta tel:te mal:ml ben:bn fas:fa per:fa "
    "cat:ca hrv:hr srp:sr slk:sk slo:sk slv:sl bul:bg lit:lt lav:lv est:et isl:is ice:is tgl:tl fil:tl urd:ur bos:bs "
    "mkd:mk mac:mk sqi:sq alb:sq kat:ka geo:ka hye:hy arm:hy aze:az kaz:kk uzb:uz mon:mn nep:ne sin:si khm:km lao:lo "
    "mya:my bur:my swa:sw afr:af glg:gl eus:eu baq:eu cym:cy wel:cy lat:la mar:mr pan:pa guj:gu kan:kn amh:am som:so "
    "hat:ht mlt:mt ltz:lb bel:be tat:tt haw:haw jav:jw sun:su yid:yi pus:ps tuk:tk fao:fo sna:sn yor:yo hau:ha lin:ln "
    "mlg:mg asm:as tgk:tg snd:sd bre:br oci:oc san:sa bod:bo tib:bo bak:ba"
).split())


class InspectError(ValueError):
    """A request that cannot be served, with a message safe to show."""


def inspect_roots() -> dict[str, str]:
    """Path prefix as the caller names it -> where SpeechDock has it mounted. ``SPDOCK_INSPECT_ROOTS``
    (a JSON object like ``SPDOCK_CAPTION_ROOTS``) replaces the default download mount; the caption
    roots are always accepted too, so library files can be inspected as well."""
    raw = os.environ.get("SPDOCK_INSPECT_ROOTS")
    roots = json.loads(raw) if raw else DEFAULT_ROOTS
    if (not isinstance(roots, dict) or not roots
            or any(not isinstance(k, str) or not isinstance(v, str) or not k.startswith("/") or not v.startswith("/") for k, v in roots.items())):
        raise ValueError("SPDOCK_INSPECT_ROOTS must be a JSON object of absolute paths")
    return {**media.media_roots(), **roots}


def resolve_inspect_path(value: object) -> Path:
    try:
        return media.resolve_media_path(value, inspect_roots())
    except MediaError as exc:
        raise InspectError(str(exc).replace("media folders", "inspectable folders")) from None


def language_code(tag: str) -> str:
    """A container language tag as Whisper's code when known (``jpn`` -> ``ja``), else the tag itself."""
    tag = tag.strip().lower().replace("_", "-").split("-")[0]
    if tag in NO_LANGUAGE:
        return ""
    if tag in LANGUAGES:
        return tag
    if tag in _ISO3:
        return _ISO3[tag]
    try:
        return normalize_language(tag)
    except ValueError:
        return tag


def parse_detect(value: object) -> str | list[str]:
    """``untagged`` (default), ``all``, ``none`` or a list of language codes: those tracks, plus untagged ones."""
    if isinstance(value, str):
        if value not in DETECT_MODES:
            raise InspectError("detect must be 'untagged', 'all', 'none' or a list of language codes")
        return value
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        codes = []
        for item in value:
            code = language_code(item)
            if code not in LANGUAGES:
                raise InspectError(f"Unknown language code in detect: {item}")
            codes.append(code)
        return sorted(set(codes))
    raise InspectError("detect must be 'untagged', 'all', 'none' or a list of language codes")


def _stream(stream: dict) -> dict:
    disposition = stream.get("disposition") or {}
    tag = media.tag(stream, "language")
    item = {"index": stream.get("index"), "codec": stream.get("codec_name"), "tag": tag, "language": language_code(tag),
            "title": media.tag(stream, "title") or None, "default": bool(disposition.get("default")),
            "forced": bool(disposition.get("forced"))}
    if stream.get("codec_type") == "audio":
        item["channels"] = stream.get("channels")
        item["commentary"] = media.is_commentary(stream)
    if stream.get("codec_type") == "video":
        item["width"], item["height"] = stream.get("width"), stream.get("height")
    return item


def wants_detection(track: dict, detect: str | list[str]) -> bool:
    if detect == "none" or track["commentary"]:
        return False
    if detect == "all" or not track["language"]:
        return True
    return detect != "untagged" and track["language"] in detect


async def read_region(path: Path, index: int, start: float, seconds: float = REGION_SECONDS) -> bytes:
    """``seconds`` of one audio stream from ``start`` as a mono 16 kHz WAV."""
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostdin", "-v", "warning", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(path),
        "-map", f"0:{index}", "-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "pipe:1",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        pcm, err = await asyncio.wait_for(process.communicate(), timeout=300)
    except BaseException:
        if process.returncode is None:
            process.kill()
        raise
    if process.returncode:
        raise MediaError(tool_failure("Could not decode the audio track", "ffmpeg", process.returncode, err))
    if len(pcm) % 2:
        pcm = pcm[:-1]
    return _pcm_wav(pcm)


async def detect_track(path: Path, track: dict, duration: float | None, engine: Engine, model: str,
                       detector: SpeechDetector) -> dict:
    """The language spoken on one track: Whisper's answer for the longest speech window found around the
    middle of the track (then at 30 % and 70 %), with its probability and where the sample was taken."""
    length = duration or 0.0
    tried = []
    for fraction in SAMPLE_POINTS if length > REGION_SECONDS else (0.0,):
        start = max(0.0, min(length * fraction - REGION_SECONDS / 2, max(0.0, length - REGION_SECONDS)))
        wav = await read_region(path, track["index"], start)
        windows = await asyncio.to_thread(speech_wav_chunks, wav, detector)
        tried.append(f"{round(fraction * 100)} %")
        if not windows:
            continue
        offset, speech = max(windows, key=lambda item: len(item[1]))
        result = await engine.detect(speech, model)
        detected = {"language": result["language_code"], "name": result["detected_language"],
                    "sample_start": round(start + offset, 1), "sample_seconds": round((len(speech) - 44) / (2 * RATE), 1)}
        if "probability" in result:
            detected["probability"] = result["probability"]
        return detected
    raise MediaError("No speech found at " + ", ".join(tried) + " of the track")


async def run_inspection(*, job: dict, progress: Callable[..., Awaitable[None]], engine_for_detection: Callable[[], Awaitable[Engine]],
                         model: str, vad_model: Path) -> dict:
    """The job: probe the file, then detect the language of each audio track the request asks about.

    Detection failures are reported per track (``detection_error``) so the caller still gets the
    tags; only an unreadable file fails the job. Once the engine is found unavailable the remaining
    tracks are not waited for again."""
    request = job["request"]
    path = Path(request["path"])
    detect = parse_detect(request.get("detect", "untagged"))
    info = await media.probe(path)
    groups: dict[str, list[dict]] = {"video": [], "audio": [], "subtitle": []}
    for stream in info["streams"]:
        kind = stream.get("codec_type")
        if kind in groups:
            groups[kind].append(_stream(stream))
    audio = groups["audio"]
    for track in audio:
        track["detected"] = None
    report = {"path": str(path), "duration": info["duration"], "detect": detect,
              "video": groups["video"], "audio": audio, "subtitles": groups["subtitle"]}
    targets = [t for t in audio if wants_detection(t, detect)][:MAX_DETECTIONS]
    await progress(0.0, info["duration"])
    if targets:
        engine = await engine_for_detection()
        detector = await asyncio.to_thread(SpeechDetector, vad_model)
        unavailable = None
        for number, track in enumerate(targets):
            if unavailable:
                track["detection_error"] = unavailable
                continue
            try:
                track["detected"] = await detect_track(path, track, info["duration"], engine, model, detector)
            except EngineUnavailable as exc:
                unavailable = track["detection_error"] = str(exc)
            except (MediaError, EngineError) as exc:
                track["detection_error"] = str(exc)
            await progress((number + 1) / len(targets), info["duration"])
    return {"result_json": report, "duration_seconds": info["duration"]}


_SOURCE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def clean_source(value: object) -> str:
    if not isinstance(value, str) or not _SOURCE.match(value):
        raise InspectError("source must be 1-32 letters, digits, '-' or '_'")
    return value
