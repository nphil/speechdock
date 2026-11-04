"""Media access for dub captions: path rules, ffprobe, stream choice and one-pass extraction."""
from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from ..audio import RATE, tool_failure

DEFAULT_ROOTS = {"/tv": "/plex-media/tv", "/movies": "/plex-media/movies"}
VIDEO_SUFFIXES = frozenset(".mkv .mp4 .m4v .avi .mov .webm .ts .m2ts .wmv .mpg .mpeg".split())
TEXT_CODECS = {"subrip": "srt", "srt": "srt", "ass": "ass", "ssa": "ass", "webvtt": "srt", "mov_text": "srt", "text": "srt"}
SIDECAR_SUFFIXES = (".srt", ".ass", ".ssa", ".vtt")
_ENGLISH = frozenset({"en", "eng", "english"})
_COMMENTARY = re.compile(
    r"commentar|\bcomment\b|director.?s?\s+(?:cut\s+)?(?:track|talk)|audio\s+description|descriptive|described\s+video"
    r"|visually[\s-]impaired|\bisolated\b|\bkaraoke\b|music\s+only|\bm&e\b|\bAD\b", re.I)
MAX_SIDECAR_BYTES = 8 * 1024 * 1024


class MediaError(ValueError):
    """A media file that cannot be used, with a message safe to show to the user."""


def media_roots() -> dict[str, str]:
    """Library prefix as Sonarr/Radarr see it -> where SpeechDock has it mounted.

    ``SPDOCK_CAPTION_ROOTS`` overrides the defaults with a JSON object of the same shape.
    A path already inside one of the mounts needs no mapping.
    """
    raw = os.environ.get("SPDOCK_CAPTION_ROOTS")
    roots = json.loads(raw) if raw else DEFAULT_ROOTS
    if (not isinstance(roots, dict) or not roots
            or any(not isinstance(k, str) or not isinstance(v, str) or not k.startswith("/") or not v.startswith("/") for k, v in roots.items())):
        raise ValueError("SPDOCK_CAPTION_ROOTS must be a JSON object of absolute paths")
    return roots


def resolve_media_path(value: object, roots: dict[str, str] | None = None) -> Path:
    """A video file inside the mounted media folders, mapped from a Sonarr/Radarr path if needed."""
    roots = roots or media_roots()
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value:
        raise MediaError("Media path must be an absolute file path")
    path = os.path.normpath(value)
    for source, local in roots.items():
        source = os.path.normpath(source)
        if path == source or path.startswith(source + "/"):
            path = os.path.normpath(local) + path[len(source):]
            break
    mounts = {os.path.normpath(local) for local in roots.values()}
    if not any(path.startswith(mount + "/") for mount in mounts):
        raise MediaError("Media path must be inside the mounted media folders")
    real = os.path.realpath(path)
    if not any(real.startswith(os.path.realpath(mount) + "/") for mount in mounts):
        raise MediaError("Media path must not leave the mounted media folders")
    if not os.path.isfile(path):
        raise MediaError("Media file not found")
    if Path(path).suffix.lower() not in VIDEO_SUFFIXES:
        raise MediaError("Not a supported video file")
    return Path(path)


async def _run(command: list[str], watch: Callable[[], Awaitable[None]] | None = None, timeout: float = 7200) -> tuple[int, bytes, bytes]:
    """Run a process to completion; ``watch`` is awaited every 2 s and may raise to cancel it."""
    process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    waiter = asyncio.ensure_future(process.communicate())
    try:
        waited = 0.0
        while True:
            done, _ = await asyncio.wait({waiter}, timeout=2.0)
            if done:
                break
            waited += 2.0
            if waited > timeout:
                raise MediaError("Reading the media file timed out")
            if watch:
                await watch()
        stdout, stderr = waiter.result()
        return process.returncode, stdout, stderr
    except BaseException:
        if process.returncode is None:
            process.kill()
        await asyncio.gather(waiter, return_exceptions=True)
        raise


async def probe(path: Path, timeout: float = 180) -> dict:
    """ffprobe summary: ``{"streams": [...], "duration": seconds | None}``."""
    # Warning level, not error: an audio format FFmpeg cannot open (e.g. xHE-AAC) exits 1 and only
    # explains itself with a warning-level line.
    code, out, err = await _run([
        "ffprobe", "-v", "warning", "-show_entries",
        "stream=index,codec_type,codec_name,channels:stream_tags=language,title"
        ":stream_disposition=default,forced,comment,hearing_impaired,visual_impaired,descriptions,dub,original"
        ":format=duration", "-of", "json", str(path)], timeout=timeout)
    if code:
        raise MediaError(tool_failure("Could not read the video", "ffprobe", code, err))
    try:
        data = json.loads(out)
        duration = float(data.get("format", {}).get("duration", 0)) or None
        return {"streams": list(data.get("streams", [])), "duration": duration}
    except (ValueError, TypeError) as exc:
        raise MediaError("Could not read the video's stream list") from exc


def tag(stream: dict, name: str) -> str:
    return next((str(value) for key, value in (stream.get("tags") or {}).items() if key.lower() == name), "")


def language_of(stream: dict) -> str:
    return tag(stream, "language").strip().lower().replace("_", "-").split("-")[0]


def is_english(stream: dict) -> bool:
    return language_of(stream) in _ENGLISH


_NO_LANGUAGE = frozenset({"", "und", "zxx", "mis", "mul"})


def is_commentary(stream: dict) -> bool:
    disposition = stream.get("disposition") or {}
    return bool(disposition.get("comment") or disposition.get("descriptions") or disposition.get("visual_impaired")
                or _COMMENTARY.search(tag(stream, "title")))


def is_dub_situation(streams: list[dict]) -> bool:
    """True when real audio (no commentary or description) has an English track and a tagged non-English one."""
    languages = {language_of(s) for s in streams if s.get("codec_type") == "audio" and not is_commentary(s)}
    english = bool(languages & _ENGLISH)
    return english and bool(languages - _ENGLISH - _NO_LANGUAGE)


def pick_english_audio(streams: list[dict]) -> dict | None:
    """The English dub to caption: the default English stream, never commentary or description."""
    english = [s for s in streams if s.get("codec_type") == "audio" and is_english(s) and not is_commentary(s)]
    return next((s for s in english if (s.get("disposition") or {}).get("default")), english[0] if english else None)


def english_text_tracks(streams: list[dict]) -> list[dict]:
    """Embedded English (or unlabelled) text subtitle tracks, except forced ones."""
    return [s for s in streams
            if s.get("codec_type") == "subtitle" and s.get("codec_name") in TEXT_CODECS
            and (is_english(s) or language_of(s) in {"", "und"})
            and not (s.get("disposition") or {}).get("forced")]


def sidecar_subtitles(video: Path) -> list[Path]:
    """English (or unlabelled) subtitle files beside the video: ``<base>.en.srt``, ``<base>.srt`` ..."""
    prefix, found = video.stem + ".", []
    try:
        entries = sorted(os.scandir(video.parent), key=lambda entry: entry.name)
    except OSError:
        return []
    for entry in entries:
        name = entry.name
        suffix = Path(name).suffix.lower()
        if name.startswith(".") or not name.startswith(prefix) or suffix not in SIDECAR_SUFFIXES or not entry.is_file():
            continue
        tags = [tag.lower() for tag in name[len(prefix):len(name) - len(suffix)].split(".") if tag]
        if "forced" in tags or (tags and not _ENGLISH.intersection(tags)) or entry.stat().st_size > MAX_SIDECAR_BYTES:
            continue
        found.append(video.parent / name)
    return found


async def extract(video: Path, audio: dict, tracks: list[dict], workdir: Path, duration: float | None,
                  progress: Callable[[float], Awaitable[None]]) -> tuple[Path, dict[int, Path], str | None]:
    """Read the file once: the English audio as 16 kHz mono PCM plus each text track as SRT/ASS.

    Returns (pcm path, {stream index: subtitle file}, note). If a subtitle track cannot be
    converted the audio is extracted alone and the note says so; no pass is repeated otherwise.
    """
    pcm = workdir / "audio.pcm"

    def command(with_tracks: list[dict]) -> list[str]:
        result = ["ffmpeg", "-nostdin", "-v", "warning", "-y", "-i", str(video), "-map", f"0:{audio['index']}",
                  "-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", str(pcm)]
        for track in with_tracks:
            kind = TEXT_CODECS[track["codec_name"]]
            result += ["-map", f"0:{track['index']}", "-vn", "-an", "-f", kind, str(workdir / f"sub{track['index']}.{kind}")]
        return result

    async def watch() -> None:
        size = pcm.stat().st_size if pcm.exists() else 0
        await progress(min(1.0, size / (duration * RATE * 2)) if duration else 0.0)

    code, _, err = await _run(command(tracks), watch)
    note = None
    if code and tracks:
        note = "a subtitle track could not be read; its text was not used"
        code, _, err = await _run(command([]), watch)
        tracks = []
    if code:
        raise MediaError(tool_failure("Could not decode the English audio", "ffmpeg", code, err))
    files = {}
    for track in tracks:
        path = workdir / f"sub{track['index']}.{TEXT_CODECS[track['codec_name']]}"
        if path.is_file() and path.stat().st_size:
            files[track["index"]] = path
    return pcm, files, note
