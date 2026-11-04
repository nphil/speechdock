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


def tool_failure(what: str, tool: str, code: int, err: bytes) -> str:
    """``what: <tool> exited with code N: <last stderr lines>``; says so when the tool printed nothing."""
    lines = [line.strip() for line in err.decode(errors="replace").splitlines() if line.strip()]
    tail = "\n".join(lines)[-300:]
    return f"{what}: {tool} exited with code {code}" + (f": {tail}" if tail else " and no message")


async def normalize_audio(source: Path, destination: Path, encode: bool) -> float:
    """Write mono 16 kHz signed 16-bit PCM and return its duration."""
    source, destination = Path(source), Path(destination)
    if encode:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-nostdin", "-v", "warning", "-y", "-i", str(source),
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
            raise ValueError(tool_failure("Could not decode audio", "ffmpeg", process.returncode, stderr))
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


class SpeechDetector:
    """CPU-only Silero v5 detector with recording-local streaming state."""

    def __init__(self, model_path: Path):
        import numpy as np
        import onnxruntime as ort

        self.np = np
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)
        self.sample_rate = np.array(RATE, dtype=np.int64)

    def probabilities(self, pcm: bytes):
        np = self.np
        samples = np.frombuffer(pcm, dtype="<i2")
        for start in range(0, len(samples), 512):
            frame = np.zeros((1, 512), dtype=np.float32)
            part = samples[start:start + 512]
            frame[0, :len(part)] = part.astype(np.float32) / 32768
            data = np.concatenate((self.context, frame), axis=1)
            result, self.state = self.session.run(None, {"input": data, "state": self.state, "sr": self.sample_rate})
            self.context = data[:, -64:]
            yield float(result[0, 0])


def _speech_windows(probabilities, sample_count: int):
    """Find speech with hysteresis and preserve context without overlapping windows."""
    spans, quiet = [], []
    start = silence = None
    for index, probability in enumerate(probabilities):
        position = index * 512
        if probability < 0.35:
            quiet.append(position)
        if probability >= 0.5:
            if start is None:
                start = position
            silence = None
        elif start is not None and probability < 0.35:
            if silence is None:
                silence = position
            if position - silence >= RATE // 2:
                if silence - start >= RATE // 10:
                    spans.append((start, silence))
                start = silence = None
    if start is not None and sample_count - start >= RATE // 10:
        spans.append((start, sample_count))

    windows = []
    previous_speech_end = 0
    limit = 28 * RATE
    for speech_start, speech_end in spans:
        start, end = max(0, speech_start - RATE), min(sample_count, speech_end + RATE)
        if windows and start <= windows[-1][1] + 2 * RATE and end - windows[-1][0] <= limit:
            windows[-1] = (windows[-1][0], max(end, windows[-1][1]))
            previous_speech_end = speech_end
            continue
        if windows and start < windows[-1][1]:
            boundary = ((previous_speech_end + speech_start) // 2) // (RATE // 100) * (RATE // 100)
            windows[-1] = (windows[-1][0], boundary)
            start = boundary
        while end - start > limit:
            # Prefer a short pause near the limit; only split speech when none exists.
            candidates = [point for point in quiet if start + 20 * RATE <= point <= start + limit]
            boundary = candidates[-1] if candidates else start + limit
            windows.append((start, boundary))
            start = boundary
        if end > start:
            windows.append((start, end))
        previous_speech_end = speech_end
    # Align PCM cuts to Whisper's 10 ms timestamp grid.
    tick = RATE // 100
    return [(start // tick * tick, end if end == sample_count else end // tick * tick)
            for start, end in windows]


def _wav_pcm(wav_data: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_data), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, RATE, "NONE"):
            raise ValueError("Expected mono 16 kHz 16-bit PCM WAV")
        if wav.getnframes() > MAX_CHUNK_SECONDS * RATE:
            raise ValueError("Speech detection chunk exceeds 300 seconds")
        return wav.readframes(wav.getnframes())


def _pcm_wav(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        wav.writeframes(pcm)
    return buffer.getvalue()


def audio_wav_chunks(wav_data: bytes):
    """Keep all non-digital-silent audio in bounded, non-overlapping WAVs."""
    pcm = _wav_pcm(wav_data)
    chunk_bytes = 28 * BYTES_PER_SECOND
    chunks = []
    for start in range(0, len(pcm), chunk_bytes):
        part = pcm[start:start + chunk_bytes]
        if any(part):
            chunks.append((start / BYTES_PER_SECOND, _pcm_wav(part)))
    return chunks


def speech_wav_chunks(wav_data: bytes, detector: SpeechDetector):
    """Return bounded speech WAVs with offsets relative to one 300-second WAV."""
    pcm = _wav_pcm(wav_data)
    chunks = []
    for start, end in _speech_windows(detector.probabilities(pcm), len(pcm) // 2):
        chunks.append((start / RATE, _pcm_wav(pcm[start * 2:end * 2])))
    return chunks


def iter_speech_wav_chunks(pcm_path: Path, vad_model_path: Path):
    """Yield speech-only WAVs and their original offsets, using Silero v5 ONNX."""
    detector = SpeechDetector(vad_model_path)
    for chunk_offset, wav_data in iter_wav_chunks(pcm_path):
        for offset, speech_wav in speech_wav_chunks(wav_data, detector):
            yield chunk_offset + offset, speech_wav



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
