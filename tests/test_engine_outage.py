"""A restarting or reloading engine is waited for with a bound, never turned into failed jobs."""
import asyncio
import io
import wave
from types import SimpleNamespace

import httpx
import pytest

from speechdock.app import make_processor, model_for, ready_model
from speechdock.audio import parse_srt
from speechdock.engine import Engine, EngineError, EngineUnavailable, make_wait
from speechdock.jobs import JobManager
from speechdock.store import Store

FAST = {"engine_wait_seconds": 5, "engine_retry_seconds": 0.01}
NATIVE = {"backend": "parakeet", "text": "Hello there.", "segments": [{"text": "Hello there.", "tokens": [
    {"text": " Hello", "t0": 100, "t1": 400}, {"text": " there", "t0": 450, "t1": 800}, {"text": ".", "t0": 800, "t1": 800}]}]}


def wav(seconds=1):
    out = io.BytesIO()
    with wave.open(out, "wb") as source:
        source.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        source.writeframes(b"\x01\x00" * (seconds * 16000))
    return out.getvalue()


def serve(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


def parakeet(**limits):
    return Engine({"engine_url": "http://engine", **FAST, **limits}, runtime="crispasr", architecture="parakeet")


def outages(*failures, then=None):
    """A handler that fails once per entry (an exception to raise or an HTTP status), then answers normally."""
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) <= len(failures):
            failure = failures[len(calls) - 1]
            if isinstance(failure, Exception):
                raise failure
            return httpx.Response(failure, text="upstream down")
        return httpx.Response(200, json=then or NATIVE)
    handler.calls = calls
    return handler


async def test_a_request_waits_out_a_restart_and_then_succeeds(monkeypatch):
    handler = outages(httpx.ConnectError("refused"), 502, 503, httpx.RemoteProtocolError("dropped"), 504)
    serve(monkeypatch, handler)
    waits = []

    async def on_wait(reason, waited):
        waits.append(reason)

    engine = parakeet()
    engine.on_wait = on_wait
    cues = parse_srt(await engine.transcribe(wav(), "parakeet", "en", "transcribe"))
    assert [cue.text for cue in cues] == ["Hello there."]
    assert len(handler.calls) == 6
    assert waits == ["ConnectError", "HTTP 502", "HTTP 503", "RemoteProtocolError", "HTTP 504"]


async def test_the_wait_is_bounded_and_says_why(monkeypatch):
    handler = outages(*[502] * 500)
    serve(monkeypatch, handler)
    with pytest.raises(EngineUnavailable, match=r"not available for 0.2 s \(HTTP 502\)"):
        await parakeet(engine_wait_seconds=0.2).transcribe(wav(), "parakeet", "en", "transcribe")
    assert 2 <= len(handler.calls) < 50  # it did wait and retry, and it did stop


def test_an_unavailable_engine_is_still_an_engine_error():
    assert issubclass(EngineUnavailable, EngineError)


@pytest.mark.parametrize("status", [400, 404, 500])
async def test_other_http_failures_are_not_retried(monkeypatch, status):
    handler = outages(*[status] * 5)
    serve(monkeypatch, handler)
    with pytest.raises(EngineError, match=f"returned HTTP {status}") as caught:
        await parakeet().transcribe(wav(), "parakeet", "en", "transcribe")
    assert not isinstance(caught.value, EngineUnavailable) and len(handler.calls) == 1


async def test_a_hung_engine_is_not_retried_after_its_long_timeout(monkeypatch):
    handler = outages(*[httpx.ReadTimeout("silent")] * 5)
    serve(monkeypatch, handler)
    with pytest.raises(EngineError, match="ReadTimeout"):
        await parakeet().transcribe(wav(), "parakeet", "en", "transcribe")
    assert len(handler.calls) == 1


async def test_a_job_cancelled_while_waiting_stops_waiting(monkeypatch):
    handler = outages(*[502] * 5)
    serve(monkeypatch, handler)

    async def on_wait(reason, waited):
        raise asyncio.CancelledError

    engine = parakeet()
    engine.on_wait = on_wait
    with pytest.raises(asyncio.CancelledError):
        await engine.transcribe(wav(), "parakeet", "en", "transcribe")
    assert len(handler.calls) == 1


class Models:
    """list_models() as the engine saw it on each call: (reachable, registered) per call, the last one repeating."""

    def __init__(self, *states):
        self.states, self.calls = states, 0

    async def list_models(self):
        reachable, registered = self.states[min(self.calls, len(self.states) - 1)]
        self.calls += 1
        return [{"model_id": "parakeet", "registered": registered, "engine_reachable": reachable, "translation_capable": False,
                 "runtime": "crispasr", "architecture": "parakeet"}]


def fake_app(models):
    return SimpleNamespace(state=SimpleNamespace(models=models))


async def test_a_job_waits_for_the_engine_before_checking_the_model():
    models = Models((False, False), (False, False), (False, False), (True, True))
    item = await ready_model(fake_app(models), "parakeet", "transcribe", make_wait(FAST))
    assert item["model_id"] == "parakeet" and models.calls == 4


async def test_a_model_the_engine_does_not_list_still_fails_at_once():
    models = Models((True, False))
    with pytest.raises(ValueError, match="is not available") as caught:
        await ready_model(fake_app(models), "parakeet", "transcribe", make_wait(FAST))
    assert not isinstance(caught.value, EngineUnavailable) and models.calls == 1


async def test_a_job_gives_up_when_the_engine_never_comes_back():
    models = Models((False, False))
    with pytest.raises(EngineUnavailable, match="not available for 0.05 s"):
        await ready_model(fake_app(models), "parakeet", "transcribe", make_wait({**FAST, "engine_wait_seconds": 0.05}))
    assert models.calls > 1


async def test_a_request_is_accepted_while_the_engine_restarts_but_a_setting_is_not():
    app = fake_app(Models((False, False)))
    assert (await model_for(app, "parakeet", "transcribe"))["model_id"] == "parakeet"
    with pytest.raises(EngineUnavailable, match="cannot be reached"):
        await model_for(app, "parakeet", "transcribe", strict=True)
    with pytest.raises(ValueError, match="is not available"):
        await model_for(app, "not-installed", "transcribe")


class Upload:
    def __init__(self, data):
        self.file = io.BytesIO(data)

    async def read(self, size):
        return self.file.read(size)


async def test_a_queue_waits_through_an_engine_outage_instead_of_failing_in_one_second(tmp_path, monkeypatch):
    """The 2026-11-04 dropouts: twenty queued jobs all failed within a second because the engine was down."""
    store = Store(tmp_path / "data")
    store.update_settings({"engine_url": "http://engine", "transcription_model": "parakeet", **FAST})
    models = Models(*[(False, False)] * 6, (True, True))
    handler = outages()
    serve(monkeypatch, handler)
    jobs = JobManager(store, make_processor(fake_app(models), store))
    try:
        queued = [await jobs.submit(Upload(b"\x01\x00" * 16000), source="bazarr", filename=f"ep{number}.mkv", task="transcribe",
                                    language="en", model="parakeet", encode=False) for number in range(4)]
        await jobs.start()
        results = [await asyncio.wait_for(jobs.wait(job["id"]), 20) for job in queued]
    finally:
        await jobs.close()
        store.close()
    assert [job["status"] for job in results] == ["completed"] * 4
    assert models.calls >= 7  # the first job polled until the engine answered; the others never saw it down
    assert all(parse_srt(job["result_text"])[0].text == "Hello there." for job in results)
