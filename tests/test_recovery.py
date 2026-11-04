"""A window whose timings the strict checks reject is recovered by cutting it, then by a fallback model."""
import io
import wave
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from speechdock import recovery
from speechdock.app import make_processor
from speechdock.audio import RATE, Cue, _pcm_wav, _wav_pcm, parse_srt, render
from speechdock.engine import Engine, EngineUnavailable, ResponseRejected, _parakeet_token_cues
from speechdock.jobs import JobManager
from speechdock.recovery import Fallback, cut_candidates, transcribe_window
from speechdock.store import Store

REJECTION = "Speech engine returned invalid, overlapping, or out-of-bounds word timings"


def payload(text, tokens):
    return {"backend": "parakeet", "text": text, "segments": [{"text": text, "tokens": [
        {"text": token, "t0": start, "t1": end} for token, start, end in tokens]}]}


# The classes of rejection seen on real episodes (invented words): one engine reply each.
REJECTED = {
    "a word with no duration in the middle": (
        payload("Tovi runo kesa.", [(" Tovi", 1200, 1280), (" runo", 1280, 1440), (" kesa", 1440, 1440), (".", 1440, 1440)]),
        28, REJECTION),
    "words piled on the last frame of the window": (
        payload("Nelo tavi va sori.", [(" Nelo", 27520, 27600), (" tavi", 27600, 27840), (" va", 27840, 27840), (" sori", 27840, 27840), (".", 27840, 27840)]),
        28, REJECTION),
    "a word starting before the previous one ends": (
        payload("Kemo sori", [(" Kemo", 3280, 3440), (" sori", 3400, 3520)]),
        28, REJECTION),
    "a token overlapping the previous token of its word": (
        payload("Lenabo", [(" Lena", 5000, 5200), ("bo", 5160, 5300)]),
        28, "Invalid native Parakeet token timings"),
    "punctuation listed after the next word": (
        payload("Kemo sori. Tavu", [(" Kemo", 3280, 3360), (" sori", 3360, 3440), (" Tavu", 3920, 4000), (".", 3920, 4000)]),
        28, "Native Parakeet tokens do not preserve the complete transcript"),
    "one word stretched over ten seconds": (
        payload("Kela Baro", [(" Kela", 15440, 15600), (" Ba", 16560, 16720), ("ro", 26320, 26640)]),
        28, "Speech engine word cannot form a readable subtitle without invented timing"),
}


@pytest.mark.parametrize("name", REJECTED)
def test_the_checks_still_reject_each_real_class_of_bad_timing(name):
    data, seconds, message = REJECTED[name]
    with pytest.raises(ValueError, match=message):
        _parakeet_token_cues(data, seconds)


def audio(seconds, quiet=()):
    """Loud 16 kHz noise except for the (start, end) second spans, as int16 samples."""
    samples = np.random.default_rng(1).integers(-3000, 3000, round(seconds * RATE)).astype("<i2")
    for start, end in quiet:
        samples[round(start * RATE):round(end * RATE)] = 0
    return samples


def test_cuts_go_to_the_quietest_places_near_the_middle_and_keep_both_sides_long_enough():
    samples = audio(28, quiet=[(13.9, 14.1), (11.0, 11.05), (3.0, 3.5), (25.0, 25.5)])
    cuts = cut_candidates(samples)
    assert 1 < len(cuts) <= recovery.CANDIDATES
    assert abs(cuts[0] / RATE - 14.0) < 0.1  # the quietest of the places within +-20 % of the middle
    assert all(0.3 * 28 <= cut / RATE <= 0.7 * 28 for cut in cuts)  # never the quiet places near the edges
    assert all(abs(a - b) >= recovery.MIN_GAP * RATE for index, a in enumerate(cuts) for b in cuts[index + 1:])


def test_a_short_window_is_not_cut():
    assert cut_candidates(audio(7.9)) == []
    assert cut_candidates(audio(8.0)) != []


class Stub:
    """An engine whose replies are decided by ``rejects(samples)``; a passing piece yields one cue 0.1-0.5 s in."""

    def __init__(self, rejects, label="piece"):
        self.rejects, self.label, self.calls, self.passed = rejects, label, [], []

    async def transcribe(self, wav, model, language, task):
        samples = np.frombuffer(_wav_pcm(wav), dtype="<i2")
        self.calls.append(len(samples) / RATE)
        if self.rejects(samples):
            raise ResponseRejected(REJECTION)
        self.passed.append(len(samples) / RATE)
        return render([Cue(100, 500, self.label)], "srt")


def window(seconds=28, **kwargs):
    return _pcm_wav(audio(seconds, **kwargs).tobytes())


async def test_a_rejected_window_is_cut_at_quiet_points_and_every_piece_passes_the_same_check():
    engine = Stub(lambda samples: len(samples) > 10 * RATE)  # long pieces are the ones the engine cannot time
    recovered = []
    text = await transcribe_window(engine, window(quiet=[(13.9, 14.1), (6.9, 7.1), (20.9, 21.1)]), "m", "en", "transcribe",
                                   at=300.0, recovered=recovered)
    cues = parse_srt(text)
    assert len(cues) == 4 and engine.calls[0] > 27  # the whole window was tried first, once
    starts = np.cumsum([0, *engine.passed[:-1]])
    assert [cue.start_ms for cue in cues] == [round(float(start) * 1000) + 100 for start in starts]
    assert abs(float(starts[2]) - 14.0) < 0.1
    assert recovered == [{"start": 300.0, "end": 328.0, "how": "cut", "reason": REJECTION}]
    assert len(engine.calls) == 7


async def test_an_accepted_window_is_passed_through_untouched():
    engine = Stub(lambda samples: False)
    recovered = []
    text = await transcribe_window(engine, window(), "m", "en", "transcribe", recovered=recovered)
    assert parse_srt(text) == [Cue(100, 500, "piece")] and engine.calls == [28.0] and recovered == []


async def test_a_second_cut_is_tried_when_the_first_leaves_a_piece_that_still_fails(monkeypatch):
    monkeypatch.setattr(recovery, "cut_candidates", lambda piece: [8 * RATE, 12 * RATE] if len(piece) == 20 * RATE else [])
    calls = []

    async def strict(piece, offset):
        calls.append((offset, len(piece) / RATE))
        if (offset, len(piece)) == (8.0, 12 * RATE):  # the right-hand piece of the first cut cannot be timed, and is too short to cut
            raise ResponseRejected(REJECTION)
        return [Cue(round(offset * 1000), round(offset * 1000) + 500, f"from {offset:g}")]

    cues = await recovery._recover(strict, audio(20), 0.0, [40], known_failed=True)
    assert [cue.text for cue in cues] == ["from 0", "from 12"]
    assert calls == [(0.0, 8.0), (8.0, 12.0), (0.0, 12.0), (12.0, 8.0)]


async def test_the_work_per_failing_window_is_bounded_and_the_error_says_where_it_was():
    engine = Stub(lambda samples: True)
    with pytest.raises(ResponseRejected, match=r"word timings \(audio 5:00-5:28; cutting it at quiet points did not help\)"):
        await transcribe_window(engine, window(), "m", "en", "transcribe", at=300.0)
    assert 1 < len(engine.calls) <= recovery.BUDGET + 1  # the first, whole-window call plus the budget


async def test_the_fallback_model_gets_the_window_only_after_cutting_fails():
    primary, whisper = Stub(lambda samples: True), Stub(lambda samples: False, label="whisper")
    recovered = []

    async def fallback():
        return Fallback(whisper, "large")

    text = await transcribe_window(primary, window(), "m", "en", "transcribe", fallback=fallback, at=3725.0, recovered=recovered)
    assert parse_srt(text) == [Cue(100, 500, "whisper")]
    assert whisper.calls == [28.0]  # the whole window, once
    assert recovered[0]["how"] == "fallback" and recovered[0]["start"] == 3725.0
    assert len(primary.calls) > 1  # cutting was tried first


async def test_the_fallback_model_is_cut_too_when_its_own_reply_is_rejected():
    primary, whisper = Stub(lambda samples: True), Stub(lambda samples: len(samples) > 20 * RATE, label="whisper")

    async def fallback():
        return Fallback(whisper, "large")

    text = await transcribe_window(primary, window(quiet=[(13.9, 14.1)]), "m", "en", "transcribe", fallback=fallback)
    assert [cue.text for cue in parse_srt(text)] == ["whisper", "whisper"]


async def test_no_fallback_model_means_a_clear_failure_not_a_guess():
    primary = Stub(lambda samples: True)

    async def fallback():
        return None

    with pytest.raises(ResponseRejected, match="cutting it at quiet points did not help"):
        await transcribe_window(primary, window(), "m", "en", "transcribe", fallback=fallback)


async def test_an_engine_that_goes_away_during_recovery_is_not_taken_for_a_bad_reply():
    class Down(Stub):
        async def transcribe(self, wav, model, language, task):
            if self.calls:
                raise EngineUnavailable("The speech engine was not available for 1 s (HTTP 502)")
            return await super().transcribe(wav, model, language, task)

    with pytest.raises(EngineUnavailable):
        await transcribe_window(Down(lambda samples: True), window(), "m", "en", "transcribe")


# The same thing end to end: a Bazarr job whose first window the engine answers with one of the bad replies.

class Upload:
    def __init__(self, data):
        self.file = io.BytesIO(data)

    async def read(self, size):
        return self.file.read(size)


class Models:
    async def list_models(self):
        return [{"model_id": "parakeet", "registered": True, "translation_capable": False, "runtime": "crispasr", "architecture": "parakeet"},
                {"model_id": "large", "registered": True, "translation_capable": True, "runtime": "whisper.cpp", "architecture": "whisper"}]


def served_wav(request):
    body = request.content
    start = body.index(b"RIFF")
    with wave.open(io.BytesIO(body[start:body.rindex(b"\r\n--")]), "rb") as source:
        return source.getnframes() / source.getframerate()


async def test_a_bazarr_job_survives_the_bad_reply_that_used_to_fail_it(tmp_path, monkeypatch):
    bad = REJECTED["a word with no duration in the middle"][0]
    good = payload("Hello there.", [(" Hello", 100, 400), (" there", 450, 800), (".", 800, 800)])
    seen = []

    def handler(request):
        length = served_wav(request)
        seen.append(length)
        return httpx.Response(200, json=bad if length > 20 else good)

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    store = Store(tmp_path / "data")
    store.update_settings({"engine_url": "http://engine", "transcription_model": "parakeet", "translation_model": "large"})
    app = SimpleNamespace(state=SimpleNamespace(models=Models()))
    jobs = JobManager(store, make_processor(app, store))
    pcm = audio(28, quiet=[(13.9, 14.1), (6.9, 7.1), (20.9, 21.1)]).tobytes()
    try:
        job = await jobs.submit(Upload(pcm), source="bazarr", filename="ep.mkv", task="transcribe", language="en", model="parakeet", encode=False)
        await jobs.start()
        result = await jobs.wait(job["id"])
    finally:
        await jobs.close()
    assert result["status"] == "completed", result["error"]
    cues = parse_srt(result["result_text"])
    assert [cue.text for cue in cues] == ["Hello there."] * len(cues) and len(cues) >= 2
    assert all(a.end_ms <= b.start_ms for a, b in zip(cues, cues[1:]))
    assert result["recovery"] == [{"start": 0.0, "end": 28.0, "how": "cut", "reason": REJECTION}]
    assert seen[0] > 27
    store.close()
