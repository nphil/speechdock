import uuid

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
async def test_completed_detection_is_accessible_from_job_detail(tmp_path, monkeypatch):
    monkeypatch.setattr(application, "DATA_DIR", tmp_path / "data")
    monkeypatch.setenv("SPDOCK_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("SPDOCK_LLAMA_CONFIG", str(tmp_path / "config.yaml"))
    async with application.lifespan(application.app):
        async def detect(job, path, progress):
            return {"result_json": {"language_code": "en", "detected_language": "english"}}
        application.app.state.jobs.processor = detect
        class Upload:
            async def read(self, size):
                if hasattr(self, "done"):
                    return b""
                self.done = True
                return b"audio"
        job = await application.app.state.jobs.submit(Upload(), source="api", filename="clip", task="detect", language="auto", model="whisper-1")
        await application.app.state.jobs.wait(job["id"])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://test") as client:
            response = await client.get(f'/api/jobs/{job["id"]}')
            assert response.status_code == 200
            assert response.json()["detection"] == {"language_code": "en", "detected_language": "english"}
