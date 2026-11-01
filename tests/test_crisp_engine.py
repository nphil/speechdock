import asyncio
import io
import wave

import httpx
import pytest

from speechdock.audio import parse_srt
from speechdock.engine import Engine, EngineError, _canary_word_cues, _native_word_cues


def wav(seconds=60):
    out = io.BytesIO()
    with wave.open(out, "wb") as source:
        source.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        source.writeframes(b"\0\0" * (seconds * 16000))
    return out.getvalue()


def payload(words):
    text = " ".join(w["word"] for w in words)
    return {"text": text, "segments": [{"text": text, "words": words}]}


def test_native_words_split_readably_without_losing_text():
    words = [{"word": f"word{i}", "start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(80)]
    data = payload(words)
    cues = _native_word_cues(data, 40)
    assert len(cues) > 5
    assert " ".join(c.text.replace("\n", " ") for c in cues) == data["text"]
    for cue in cues:
        assert 0 <= cue.start_ms < cue.end_ms <= 40000
        assert cue.end_ms - cue.start_ms <= 7000
        assert len(cue.text.replace("\n", " ")) <= 84
        assert len(cue.text.splitlines()) <= 2
    assert all(a.end_ms <= b.start_ms for a, b in zip(cues, cues[1:]))


def test_original_punctuation_and_spacing_preserved():
    data = {"text": "Hello, world!", "segments": [{"text": "Hello, world!", "words": [
        {"word": " Hello,", "start": 0, "end": 0.5},
        {"word": "world!", "start": 0.5, "end": 1}]}]}
    assert _native_word_cues(data, 1)[0].text == "Hello, world!"


@pytest.mark.parametrize("words", [
    [{"word": "hello", "start": 0, "end": 0}],
    [{"word": "hello", "start": -1, "end": 1}],
    [{"word": "hello", "start": True, "end": 1}],
    [{"word": "hello", "start": 0, "end": float("nan")}],
    [{"word": "hello", "start": 0, "end": 1.01}],
    [{"word": "hello", "start": 0, "end": 0.7}, {"word": "world", "start": 0.6, "end": 1}],
])
def test_native_word_times_fail_closed(words):
    with pytest.raises(ValueError, match="timings"):
        _native_word_cues(payload(words), 1)


def test_text_omission_and_coarse_timing_are_not_substituted():
    with pytest.raises(ValueError, match="word timings"):
        _native_word_cues({"segments": [{"text": "hello", "start": 0, "end": 1}]}, 1)
    data = payload([{"word": "hello", "start": 0, "end": 1}])
    data["segments"][0]["text"] += " world"
    with pytest.raises(ValueError, match="complete transcript"):
        _native_word_cues(data, 1)
    data["segments"][0]["text"] = "hello"
    data["text"] = "hello omitted"
    with pytest.raises(ValueError, match="complete transcript"):
        _native_word_cues(data, 1)


def test_cross_segment_overlaps_rejected():
    segment = {"text": "hello", "words": [{"word": "hello", "start": 0, "end": 1}]}
    with pytest.raises(ValueError, match="overlapping"):
        _native_word_cues({"segments": [segment, segment]}, 2)


def test_unreadable_single_word_fails_instead_of_inventing_times():
    with pytest.raises(ValueError, match="readable subtitle"):
        _native_word_cues(payload([{"word": "hello", "start": 0, "end": 9}]), 10)


def test_canary_tied_frame_union_and_exact_padded_cell_support():
    words = [{"word": "Esta", "start": 1.68, "end": 1.76},
             {"word": "revolucionaria", "start": 1.68, "end": 2.64},
             {"word": "televisión.", "start": 8.0, "end": 9.12}]
    data = payload(words)
    cues = _canary_word_cues(data, 9.088)
    assert cues[0].start_ms == 1680
    assert cues[0].end_ms == 2640
    assert cues[0].text == "Esta revolucionaria"
    assert cues[-1].end_ms == 9088
    assert " ".join(c.text for c in cues) == data["text"]
    assert words[-1]["end"] == 9.12  # Never mutate the raw response.
    with pytest.raises(ValueError, match="overlapping"):
        _native_word_cues(data, 9.088)


@pytest.mark.parametrize("words", [
    [{"word": "bad", "start": 0, "end": 1.12}],  # Beyond known padded end1.04.
    [{"word": "bad", "start": 0, "end": 1.01}],  # Not an80ms frame.
    [{"word": "bad", "start": True, "end": 0.08}],
    [{"word": "bad", "start": 0, "end": float("nan")}],
    [{"word": "bad", "start": 1.04, "end": 1.12}],
    [{"word": "bad", "start": 0, "end": 0.48},
     {"word": "overlap", "start": 0.08, "end": 0.56}],
    [{"word": "bad", "start": 0.16, "end": 0.24},
     {"word": "reversed", "start": 0.08, "end": 0.32}],
])
def test_canary_does_not_repair_unrelated_invalid_times(words):
    with pytest.raises(ValueError, match="Canary"):
        _canary_word_cues(payload(words), 1)


def test_canary_does_not_split_an_unreadable_inseparable_cluster():
    words = [{"word": "word" * 12, "start": 0, "end": 0.08}] * 2
    with pytest.raises(ValueError, match="readable subtitle"):
        _canary_word_cues(payload(words), 1)


@pytest.mark.parametrize("architecture,language", [("parakeet", "auto"), ("canary", "es")])
def test_crisp_protocol(monkeypatch, architecture, language):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=payload([{"word": "Hola", "start": 0, "end": 0.96}]))
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    engine = Engine({"engine_url": "http://engine"}, runtime="crispasr", architecture=architecture)
    cues = parse_srt(asyncio.run(engine.transcribe(wav(1), "owned", language, "transcribe")))
    assert cues[0].text == "Hola"
    request = requests[0]
    assert request.url.path == "/v1/audio/transcriptions"
    body = request.content
    for field in ("require_word_timestamps", "no_auto_aligner", "lid_backend", "response_format"):
        assert f'name="{field}"'.encode() in body
    assert b"token_timestamps" not in body
    if architecture == "canary":
        assert b'name="source_lang"\r\n\r\nes' in body
        assert b'name="target_lang"\r\n\r\nes' in body


def test_capabilities_fail_before_request():
    engine = Engine({"engine_url": "http://engine"}, runtime="crispasr", architecture="canary")
    with pytest.raises(EngineError, match="explicit source"):
        asyncio.run(engine.transcribe(wav(1), "owned", "auto", "transcribe"))
    with pytest.raises(EngineError, match="not been validated"):
        asyncio.run(engine.transcribe(wav(1), "owned", "es", "translate"))
    with pytest.raises(EngineError, match="cannot detect"):
        asyncio.run(engine.detect(wav(1), "owned"))
    with pytest.raises(EngineError, match="Unsupported speech runtime"):
        Engine({"engine_url": "http://engine"}, runtime="crispasr", architecture="llama")
