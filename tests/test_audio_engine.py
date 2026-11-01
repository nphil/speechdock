import asyncio
import io
import json
import wave

import httpx
import pytest

from speechdock import audio
from speechdock.engine import Engine, EngineError
from speechdock.languages import normalize_language


def test_pcm_chunks_are_bounded_and_wav_wrapped(tmp_path):
    source = tmp_path / "audio.pcm"
    source.write_bytes(b"\x00\x00" * (audio.RATE * 301))
    chunks = list(audio.iter_wav_chunks(source))
    assert [offset for offset, _ in chunks] == [0, 300]
    for (_, data), seconds in zip(chunks, [300, 1]):
        with wave.open(io.BytesIO(data)) as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
            assert wav.getnframes() == 16000 * seconds
    with pytest.raises(ValueError):
        list(audio.iter_wav_chunks(source, 301))


def test_raw_pcm_validation_and_limit(tmp_path):
    source, dest = tmp_path / "in", tmp_path / "out"
    source.write_bytes(b"a")
    with pytest.raises(ValueError, match="complete"):
        asyncio.run(audio.normalize_audio(source, dest, False))
    with source.open("wb") as f:
        f.truncate(audio.MAX_BYTES + 2)
    with pytest.raises(ValueError, match="6-hour"):
        asyncio.run(audio.normalize_audio(source, dest, False))
    assert not dest.exists()


def test_ffmpeg_decodes_wav(tmp_path):
    source, dest = tmp_path / "in.wav", tmp_path / "out.pcm"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x00\x00" * 16000)
    assert asyncio.run(audio.normalize_audio(source, dest, True)) == 1
    assert dest.stat().st_size == 32000


def test_subtitle_offset_and_formats():
    source = "1\n00:00:00,000 --> 00:00:01,500\nHello\nworld\n\n2\n00:00:01,500 --> 00:00:02,000\nBye\n"
    cues = audio.parse_srt(source, 300)
    assert cues[0] == audio.Cue(300000, 301500, "Hello\nworld")
    assert "00:05:00,000 --> 00:05:01,500" in audio.render(cues, "srt")
    assert audio.render(cues, "vtt").startswith("WEBVTT\n\n1\n00:05:00.000")
    assert audio.render(cues, "txt") == "Hello\nworld\nBye"
    assert json.loads(audio.render(cues, "json"))["segments"][0]["start"] == 300
    assert audio.parse_srt("") == []


@pytest.mark.parametrize("text", ["not srt", "1\n00:00:00,000 --> 00:00:00,000\nhello", "1\n00:00:01,000 --> 00:00:02,000\na\n\n2\n00:00:01,999 --> 00:00:03,000\nb", "1\n00:60:00,000 --> 01:01:00,000\na"])
def test_invalid_srt_rejected(text):
    with pytest.raises(ValueError):
        audio.parse_srt(text)


def _wav(seconds=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setparams((1, 2, audio.RATE, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * int(audio.RATE * seconds))
    return buffer.getvalue()


def test_engine_protocol_and_detection(monkeypatch):
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "whisper-1"}]})
        if b'detect_language' in request.content:
            return httpx.Response(200, json={"language": "english", "detected_language": "english", "language_probabilities": {"en": 0.99}})
        return httpx.Response(200, text="1\n00:00:00,000 --> 00:00:01,000\nhello\n")
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    engine = Engine({"engine_url": "http://example.test/v1/"})
    assert asyncio.run(engine.probe()) == {"reachable": True, "models": ["whisper-1"], "error": None}
    assert asyncio.run(engine.detect(b"wav", "whisper-1")) == {"language_code": "en", "detected_language": "english"}
    result = asyncio.run(engine.transcribe(_wav(), "full", "fr", "translate"))
    assert "hello" in result
    assert normalize_language("cantonese") == "yue"


def test_engine_rejects_unknown_detection_and_bad_srt(monkeypatch):
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"language": "unknown"}) if b'detect_language' in request.content else httpx.Response(200, text="bad")), **kwargs))
    engine = Engine({"engine_url": "http://example.test"})
    with pytest.raises(EngineError, match="recognized"):
        asyncio.run(engine.detect(b"wav", "whisper-1"))
    with pytest.raises(EngineError, match="SRT"):
        asyncio.run(engine.transcribe(b"wav", "whisper-1", "auto", "transcribe"))


class _SampleSpeechDetector:
    """Deterministic speech decisions for synthetic PCM; no movie audio is stored."""

    def __init__(self, _model=None):
        pass

    def probabilities(self, pcm):
        for start in range(0, len(pcm), 1024):
            yield 0.9 if any(pcm[start:start + 1024]) else 0.0


def test_speech_windows_keep_original_pcm_offsets_and_silence(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "SpeechDetector", _SampleSpeechDetector)
    source = tmp_path / "audio.pcm"
    source.write_bytes(b"\0\0" * (audio.RATE * 8) + b"\1\0" * (audio.RATE * 2) + b"\0\0" * (audio.RATE * 5))
    chunks = list(audio.iter_speech_wav_chunks(source, "unused"))
    assert len(chunks) == 1
    offset, data = chunks[0]
    assert 6.9 <= offset <= 7.1
    with wave.open(io.BytesIO(data)) as wav:
        pcm = wav.readframes(wav.getnframes())
    first_speech_sample = pcm.index(b"\1\0") // 2
    assert offset + first_speech_sample / audio.RATE == pytest.approx(8, abs=.001)
    source.write_bytes(b"\0\0" * audio.RATE * 3)
    assert list(audio.iter_speech_wav_chunks(source, "unused")) == []


def test_speech_windows_preserve_offsets_across_outer_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "SpeechDetector", _SampleSpeechDetector)
    source = tmp_path / "long.pcm"
    source.write_bytes(b"\0\0" * (audio.RATE * 300) + b"\1\0" * (audio.RATE * 2))
    chunks = list(audio.iter_speech_wav_chunks(source, "unused"))
    assert len(chunks) == 1
    offset, data = chunks[0]
    assert offset == 300
    with wave.open(io.BytesIO(data)) as wav:
        assert wav.getnframes() == 2 * audio.RATE


def test_long_speech_splits_at_pause_without_losing_samples():
    # Continuous speech with a 96 ms pause near the end of the first window.
    probabilities = [0.9] * 1400
    probabilities[800:803] = [0.0] * 3
    windows = audio._speech_windows(probabilities, len(probabilities) * 512)
    assert windows[0][1] == (802 * 512) // 160 * 160
    assert all(end - start <= 28 * audio.RATE for start, end in windows)
    assert all(left[1] == right[0] for left, right in zip(windows, windows[1:]))
    # The engine reports timestamps in 10 ms increments. Real PCM cuts must
    # use the same grid or a rounded-up cue can overlap the next window.
    assert all(start % 160 == 0 and end % 160 == 0 for start, end in windows)
    cues = [audio.Cue(start // 16, end // 16, "speech") for start, end in windows]
    assert len(audio.parse_srt(audio.render(cues, "srt"))) == len(windows)
    assert windows[0][0] == 0 and windows[-1][1] == len(probabilities) * 512


def test_short_speech_still_becomes_a_valid_timed_cue():
    probabilities = [0.0] * 64 + [0.9] * 8 + [0.0] * 40
    (start, end), = audio._speech_windows(probabilities, len(probabilities) * 512)
    assert start < 64 * 512 and end > 72 * 512
    cue = audio.parse_srt("1\n00:00:00,000 --> 00:00:00,500\nEnjoy.\n", start / audio.RATE)[0]
    assert cue.end_ms > cue.start_ms
    assert cue.text == "Enjoy."



@pytest.mark.parametrize("end_ms", [1000, 1001, 1010])
def test_engine_keeps_speech_within_real_audio(monkeypatch, end_ms):
    async def response(*_args):
        return httpx.Response(200, text=f"1\n00:00:00,200 --> 00:00:01,{end_ms - 1000:03d}\nEnjoy.\n")
    engine = Engine({"engine_url": "http://example.test"})
    monkeypatch.setattr(engine, "_request", response)
    result = asyncio.run(engine.transcribe(_wav(), "whisper-1", "en", "transcribe"))
    assert audio.parse_srt(result) == [audio.Cue(200, 1000, "Enjoy.")]


@pytest.mark.parametrize("srt", [
    "1\n00:00:00,200 --> 00:00:01,011\nToo late\n",
    "1\n00:00:01,000 --> 00:00:01,010\nOutside\n",
    "1\n00:00:01,000 --> 00:00:01,000\nZero\n",
    "1\n00:00:00,500 --> 00:00:01,001\nfirst\n\n2\n00:00:01,000 --> 00:00:01,010\nsecond\n",
    "1\n00:00:00,800 --> 00:00:00,900\nfirst\n\n2\n00:00:00,700 --> 00:00:00,800\nsecond\n",
])
def test_engine_rejects_unrecoverable_timing(monkeypatch, srt):
    async def response(*_args):
        return httpx.Response(200, text=srt)
    engine = Engine({"engine_url": "http://example.test"})
    monkeypatch.setattr(engine, "_request", response)
    with pytest.raises(EngineError):
        asyncio.run(engine.transcribe(_wav(), "whisper-1", "en", "transcribe"))

