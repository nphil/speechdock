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


def test_engine_protocol_and_detection(monkeypatch):
    requests = []
    def handler(request):
        requests.append(request)
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
    result = asyncio.run(engine.transcribe(b"wav", "full", "fr", "translate"))
    assert "hello" in result
    assert requests[-1].url.path == "/v1/audio/transcriptions"
    assert b'translate' in requests[-1].content and b'no_timestamps' in requests[-1].content
    assert normalize_language("cantonese") == "yue"


def test_engine_rejects_unknown_detection_and_bad_srt(monkeypatch):
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"language": "unknown"}) if b'detect_language' in request.content else httpx.Response(200, text="bad")), **kwargs))
    engine = Engine({"engine_url": "http://example.test"})
    with pytest.raises(EngineError, match="recognized"):
        asyncio.run(engine.detect(b"wav", "whisper-1"))
    with pytest.raises(EngineError, match="SRT"):
        asyncio.run(engine.transcribe(b"wav", "whisper-1", "auto", "transcribe"))
