"""Bounded PCM processing and subtitle handling."""

import asyncio
import io
import json
import re
import wave
from dataclasses import dataclass
from pathlib import Path

RATE = 16000
BYTES_PER_SECOND = RATE * 2
MAX_CHUNK_SECONDS = 300
MAX_BYTES = 6 * 60 * 60 * BYTES_PER_SECOND


@dataclass(frozen=True)
class Cue:
    start_ms: int
    end_ms: int
    text: str


async def normalize_audio(source: Path, destination: Path, encode: bool) -> float:
    """Write mono 16 kHz signed 16-bit PCM and return its duration."""
    source, destination = Path(source), Path(destination)
    if encode:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
            "-vn", "-ac", "1", "-ar", str(RATE), "-t", "21600.000063",
            "-f", "s16le", str(destination),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await process.communicate()
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
            await process.wait()
            destination.unlink(missing_ok=True)
            raise
        if process.returncode:
            destination.unlink(missing_ok=True)
            raise ValueError("Could not decode audio: " + stderr.decode(errors="replace").strip()[-1000:])
    else:
        if source.stat().st_size > MAX_BYTES:
            raise ValueError("Audio exceeds the 6-hour limit")
        if source.stat().st_size % 2:
            raise ValueError("Raw PCM must contain complete 16-bit samples")
        try:
            with source.open("rb") as src, destination.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
                    await asyncio.sleep(0)
        except (asyncio.CancelledError, OSError):
            destination.unlink(missing_ok=True)
            raise
    size = destination.stat().st_size
    if size > MAX_BYTES:
        destination.unlink(missing_ok=True)
        raise ValueError("Audio exceeds the 6-hour limit")
    if not size or size % 2:
        destination.unlink(missing_ok=True)
        raise ValueError("Audio is empty or contains incomplete PCM samples")
    return size / BYTES_PER_SECOND


def iter_wav_chunks(pcm_path: Path, chunk_seconds: int = MAX_CHUNK_SECONDS):
    if not isinstance(chunk_seconds, int) or not 1 <= chunk_seconds <= MAX_CHUNK_SECONDS:
        raise ValueError("Chunk duration must be between 1 and 300 seconds")
    size = Path(pcm_path).stat().st_size
    if size % 2:
        raise ValueError("Raw PCM must contain complete 16-bit samples")
    with open(pcm_path, "rb") as source:
        offset = 0
        while pcm := source.read(chunk_seconds * BYTES_PER_SECOND):
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(RATE)
                wav.writeframes(pcm)
            yield offset / BYTES_PER_SECOND, buffer.getvalue()
            offset += len(pcm)


_TIMESTAMP = r"(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})"
_TIMING = re.compile(rf"^{_TIMESTAMP}\s*-->\s*{_TIMESTAMP}\s*$")


def _milliseconds(parts: tuple[str, ...]) -> int:
    h, m, s, ms = map(int, parts)
    if m >= 60 or s >= 60:
        raise ValueError("Invalid subtitle timestamp")
    return ((h * 60 + m) * 60 + s) * 1000 + ms


def _validate(cues: list[Cue]) -> None:
    previous_end = 0
    for cue in cues:
        if cue.start_ms < previous_end or cue.end_ms <= cue.start_ms or not cue.text.strip():
            raise ValueError("Invalid or overlapping subtitle timings")
        previous_end = cue.end_ms


def parse_srt(text: str, offset_seconds: float = 0) -> list[Cue]:
    if offset_seconds < 0:
        raise ValueError("Subtitle offset cannot be negative")
    if not text.strip():
        return []
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n").strip()):
        lines = block.split("\n")
        if len(lines) < 3 or not lines[0].strip().isdigit():
            raise ValueError("Invalid SRT response from speech engine")
        match = _TIMING.fullmatch(lines[1].strip())
        if not match:
            raise ValueError("Invalid SRT timestamp from speech engine")
        start = _milliseconds(match.groups()[:4])
        end = _milliseconds(match.groups()[4:])
        offset = round(offset_seconds * 1000)
        cues.append(Cue(start + offset, end + offset, "\n".join(lines[2:]).strip()))
    _validate(cues)
    return cues


def _timestamp(ms: int, separator: str) -> str:
    seconds, milli = divmod(ms, 1000)
    minutes, second = divmod(seconds, 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour:02}:{minute:02}:{second:02}{separator}{milli:03}"


def render(cues: list[Cue], format: str) -> str:
    _validate(cues)
    if format == "txt":
        return "\n".join(cue.text for cue in cues)
    if format == "json":
        return json.dumps({"text": " ".join(c.text.replace("\n", " ") for c in cues), "segments": [{"start": c.start_ms / 1000, "end": c.end_ms / 1000, "text": c.text} for c in cues]}, ensure_ascii=False)
    if format not in {"srt", "vtt"}:
        raise ValueError("Unsupported subtitle format")
    sep = "," if format == "srt" else "."
    blocks = [f"{i}\n{_timestamp(c.start_ms, sep)} --> {_timestamp(c.end_ms, sep)}\n{c.text}" for i, c in enumerate(cues, 1)]
    body = "\n\n".join(blocks)
    if format == "vtt":
        return "WEBVTT\n\n" + (body + "\n" if body else "")
    return body + "\n" if body else ""
