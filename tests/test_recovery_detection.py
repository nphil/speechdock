import asyncio
import io
import threading
import uuid
import wave

import httpx
import pytest

from speechdock import app as application
from speechdock.jobs import JobManager
from speechdock.store import Store


@pytest.mark.asyncio
async def test_restart_removes_orphan_pcm_but_preserves_unowned_files(tmp_path):
    store = Store(tmp_path)
    async def unused(job, path, progress):
        return {}
    manager = JobManager(store, unused)
    orphan_id = str(uuid.uuid4())
    pcm = manager.upload_dir / f"{orphan_id}.pcm"
    pcm.write_bytes(b"audio")
    unrelated = manager.upload_dir / "notes.pcm"
    unrelated.write_bytes(b"keep")
    await manager.start()
    try:
        assert not pcm.exists()
        assert unrelated.read_bytes() == b"keep"
    finally:
        await manager.close()
        store.close()


@pytest.mark.asyncio
async def test_detection_uses_first_voiced_window(tmp_path, monkeypatch):
    monkeypatch.setattr(application, "DATA_DIR", tmp_path / "data")
    monkeypatch.setenv("SPDOCK_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("SPDOCK_LLAMA_CONFIG", str(tmp_path / "config.yaml"))
    async def normalize(path, pcm, encode):
        pcm.write_bytes(b"pcm")
        return 600
    async def model_for(*args, **kwargs):
        return {"runtime": "whisper.cpp", "architecture": "whisper"}
    calls = []
    class Engine:
        def __init__(self, settings, *, runtime, architecture, on_wait=None):
            assert (runtime, architecture) == ("whisper.cpp", "whisper")
            pass
        async def detect(self, wav, model):
            calls.append(wav)
            return {"language_code": "en", "detected_language": "english"}
    monkeypatch.setattr(application, "normalize_audio", normalize)
    monkeypatch.setattr(application, "model_for", model_for)
    monkeypatch.setattr(application, "Engine", Engine)
    monkeypatch.setattr(application, "SpeechDetector", lambda path: object())
    monkeypatch.setattr(application, "iter_wav_chunks", lambda path: iter([(0, b"silent"), (300, b"speech")]))
    monkeypatch.setattr(application, "speech_wav_chunks", lambda wav, detector: [] if wav == b"silent" else [(2, b"voice"), (5, b"later")])
    async with application.lifespan(application.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://test") as client:
            response = await client.post("/api/jobs", data={"task": "detect"}, files={"file": ("clip.wav", b"audio", "audio/wav")})
            job_id = response.json()["id"]
            assert (await application.app.state.jobs.wait(job_id))["status"] == "completed"
            detail = await client.get(f"/api/jobs/{job_id}")
            assert detail.json()["detection"]["language_code"] == "en"
            assert calls == [b"voice"]


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["transcribe", "translate", "detect"])
async def test_no_speech_jobs_never_call_engine(tmp_path, monkeypatch, task):
    monkeypatch.setattr(application, "DATA_DIR", tmp_path / "data")
    monkeypatch.setenv("SPDOCK_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("SPDOCK_LLAMA_CONFIG", str(tmp_path / "config.yaml"))
    async def normalize(path, pcm, encode):
        pcm.write_bytes(b"pcm")
        return 10
    async def model_for(*args, **kwargs):
        return {"runtime": "whisper.cpp", "architecture": "whisper"}
    class NoEngineCalls:
        def __init__(self, settings, *, runtime, architecture, on_wait=None):
            pass
        async def transcribe(self, *args):
            pytest.fail("No speech should reach the engine")
        async def detect(self, *args):
            pytest.fail("No speech should reach the engine")
    monkeypatch.setattr(application, "normalize_audio", normalize)
    monkeypatch.setattr(application, "model_for", model_for)
    monkeypatch.setattr(application, "Engine", NoEngineCalls)
    monkeypatch.setattr(application, "SpeechDetector", lambda path: object())
    monkeypatch.setattr(application, "iter_wav_chunks", lambda path: iter([(0, b"wav")]))
    monkeypatch.setattr(application, "speech_wav_chunks", lambda wav, detector: [])
    monkeypatch.setattr(application, "audio_wav_chunks", lambda wav: [])
    async with application.lifespan(application.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://test") as client:
            response = await client.post("/api/jobs", data={"task": task}, files={"file": ("silent.wav", b"audio", "audio/wav")})
            assert response.status_code == 202
            job_id = response.json()["id"]
            result = await application.app.state.jobs.wait(job_id)
            if task == "detect":
                assert result["status"] == "failed"
                assert "no speech" in result["error"]
            else:
                assert result["status"] == "completed"
                output = await client.get(f"/api/jobs/{job_id}/result")
                assert output.status_code == 200 and output.text == ""
            assert not list((tmp_path / "data" / "uploads").glob("*.pcm"))


@pytest.mark.asyncio
@pytest.mark.parametrize("end,valid", [("01,000", True), ("02,001", False)])
@pytest.mark.parametrize("task", ["transcribe", "translate"])
async def test_voice_windows_keep_original_offsets(tmp_path, monkeypatch, end, valid, task):
    monkeypatch.setattr(application, "DATA_DIR", tmp_path / "data")
    monkeypatch.setenv("SPDOCK_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("SPDOCK_LLAMA_CONFIG", str(tmp_path / "config.yaml"))
    async def normalize(path, pcm, encode):
        pcm.write_bytes(b"pcm")
        return 600
    async def model_for(*args, **kwargs):
        return {"runtime": "whisper.cpp", "architecture": "whisper"}
    calls = []
    class Engine:
        def __init__(self, settings, *, runtime, architecture, on_wait=None):
            pass
        async def transcribe(self, wav, model, language, current_task):
            assert current_task == task
            calls.append(wav)
            return f"1\n00:00:00,500 --> 00:00:{end}\nHello\n"
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes(32000 * 2))
    voice = buffer.getvalue()
    monkeypatch.setattr(application, "normalize_audio", normalize)
    monkeypatch.setattr(application, "model_for", model_for)
    monkeypatch.setattr(application, "Engine", Engine)
    monkeypatch.setattr(application, "SpeechDetector", lambda path: pytest.fail("Subtitles must not require VAD"))
    monkeypatch.setattr(application, "iter_wav_chunks", lambda path: iter([(0, b"first"), (300, b"second")]))
    def windows(wav):
        return [] if wav == b"first" else [(12.25, voice)]
    monkeypatch.setattr(application, "audio_wav_chunks", windows)
    async with application.lifespan(application.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://test") as client:
            response = await client.post("/api/jobs", data={"task": task}, files={"file": ("clip.wav", b"audio", "audio/wav")})
            job_id = response.json()["id"]
            job = await application.app.state.jobs.wait(job_id)
            if valid:
                assert job["status"] == "completed"
                result = await client.get(f"/api/jobs/{job_id}/result")
                assert "00:05:12,750 --> 00:05:13,250" in result.text
            else:
                assert job["status"] == "failed"
                assert "beyond its audio window" in job["error"]
            assert calls == [voice]


@pytest.mark.asyncio
async def test_cancel_during_voice_detection_cleans_temporary_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(application, "DATA_DIR", tmp_path / "data")
    monkeypatch.setenv("SPDOCK_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("SPDOCK_LLAMA_CONFIG", str(tmp_path / "config.yaml"))
    started, release = threading.Event(), threading.Event()
    async def normalize(path, pcm, encode):
        pcm.write_bytes(b"pcm")
        return 10
    async def model_for(*args, **kwargs):
        return {"runtime": "whisper.cpp", "architecture": "whisper"}
    def windows(wav):
        started.set()
        assert release.wait(10)
        return [(0, b"voice")]
    class Engine:
        def __init__(self, settings, *, runtime, architecture, on_wait=None):
            pass
        async def transcribe(self, *args):
            pytest.fail("Cancelled speech must not reach the engine")
    monkeypatch.setattr(application, "normalize_audio", normalize)
    monkeypatch.setattr(application, "model_for", model_for)
    monkeypatch.setattr(application, "Engine", Engine)
    monkeypatch.setattr(application, "SpeechDetector", lambda path: pytest.fail("Subtitles must not require VAD"))
    monkeypatch.setattr(application, "iter_wav_chunks", lambda path: iter([(0, b"wav")]))
    monkeypatch.setattr(application, "audio_wav_chunks", windows)
    async with application.lifespan(application.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://test") as client:
            response = await client.post("/api/jobs", files={"file": ("clip.wav", b"audio", "audio/wav")})
            job_id = response.json()["id"]
            try:
                assert await asyncio.to_thread(started.wait, 10)
                assert (await client.post(f"/api/jobs/{job_id}/cancel")).status_code == 200
            finally:
                release.set()
            assert (await application.app.state.jobs.wait(job_id))["status"] == "cancelled"
            assert not list((tmp_path / "data" / "uploads").glob("*.pcm"))
