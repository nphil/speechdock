from types import SimpleNamespace
import json
import sqlite3

import httpx

import pytest

from speechdock.app import clean_filename, model_for, normalize_url
from speechdock.app import app
from speechdock.store import Store


class Models:
    async def list_models(self):
        return [
            {"model_id": "whisper-1", "registered": True, "translation_capable": False},
            {"model_id": "speechdock-large-v3", "registered": False, "translation_capable": True},
        ]


@pytest.mark.asyncio
async def test_translation_never_uses_turbo_or_unregistered_model():
    app = SimpleNamespace(state=SimpleNamespace(models=Models()))
    with pytest.raises(ValueError, match="cannot translate"):
        await model_for(app, "whisper-1", "translate")
    with pytest.raises(ValueError, match="not available"):
        await model_for(app, "speechdock-large-v3", "translate")
    assert (await model_for(app, "whisper-1", "transcribe"))["model_id"] == "whisper-1"


def test_url_and_display_filename_are_safe():
    assert normalize_url("https://host.example/v1/") == "https://host.example"
    with pytest.raises(ValueError):
        normalize_url("https://user:password@host.example")
    assert clean_filename("/private/video\nname.mp4") == "videoname.mp4"


def test_detection_setting_migrates_once_and_stays_independent(tmp_path):
    db = sqlite3.connect(tmp_path / "speechdock.sqlite3")
    db.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("INSERT INTO settings VALUES (?, ?)", ("transcription_model", json.dumps("old-model")))
    db.commit()
    db.close()
    store = Store(tmp_path)
    assert store.get_settings()["detection_model"] == "old-model"
    store.update_settings({"transcription_model": "new-model"})
    store.close()
    store = Store(tmp_path)
    assert store.get_settings()["detection_model"] == "old-model"
    store.close()


@pytest.mark.asyncio
async def test_import_api_stream_and_validation_privacy(monkeypatch):
    class Manager:
        runtimes = {"whisper.cpp": {"available": True}}
        async def list_models(self):
            return []
        async def downloads(self):
            return []
        async def create_import(self, **fields):
            assert fields["source"] == "upload"
            return {"model": {"id": "owned"}, "download": {"status": "awaiting_upload"}}
        async def receive_upload(self, catalog_id, stream, content_length=None):
            data = b"".join([chunk async for chunk in stream])
            assert (catalog_id, data, content_length) == ("owned", b"abc", 3)
            return {"status": "completed"}
        async def remove_import(self, catalog_id, confirm=False):
            return {"id": catalog_id, "confirmed": confirm}
    monkeypatch.setattr(app.state, "models", Manager(), raising=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/models/imports", json={"name": "My model", "source": "upload"})
        assert response.status_code == 202
        assert (await client.put("/api/models/owned/upload", content=b"abc")).json()["status"] == "completed"
        assert (await client.delete("/api/models/owned/import?confirm=true")).json()["confirmed"]
        assert (await client.get("/api/models")).json()["runtimes"]["whisper.cpp"]["available"]
        secret = "https://example.com/model?token=secret"
        response = await client.post("/api/models/imports", json={"name": "x", "source": "url", "url": secret, "expected_size_bytes": "bad"})
        assert response.status_code == 422 and secret not in response.text and "secret" not in response.text


@pytest.mark.asyncio
async def test_detection_capability_is_enforced():
    app_state = SimpleNamespace(state=SimpleNamespace(models=Models()))
    with pytest.raises(ValueError, match="cannot detect"):
        await model_for(app_state, "whisper-1", "detect")


@pytest.mark.asyncio
async def test_task_defaults_and_explicit_canary_language(monkeypatch, tmp_path):
    class Manager:
        async def list_models(self):
            return [
                {"model_id": "detector", "registered": True, "language_detection_capable": True, "translation_capable": False},
                {"model_id": "canary", "registered": True, "language_detection_capable": False, "translation_capable": False, "requires_language": True},
            ]
    class Jobs:
        async def submit(self, upload, **fields):
            return fields
    store = Store(tmp_path)
    store.update_settings({"transcription_model": "canary", "detection_model": "detector"})
    monkeypatch.setattr(app.state, "store", store, raising=False)
    monkeypatch.setattr(app.state, "models", Manager(), raising=False)
    monkeypatch.setattr(app.state, "jobs", Jobs(), raising=False)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            audio = {"file": ("clip.wav", b"audio", "audio/wav")}
            response = await client.post("/api/jobs", data={"task": "detect"}, files=audio)
            assert response.status_code == 202 and response.json()["model"] == "detector"
            response = await client.post("/api/jobs", files=audio)
            assert response.status_code == 422 and "explicit source language" in response.text
            response = await client.post("/api/jobs", data={"language": "fr"}, files=audio)
            assert response.status_code == 202 and response.json()["language"] == "fr"
            response = await client.patch("/api/settings", json={"detection_model": "canary"})
            assert response.status_code == 422
    finally:
        store.close()
