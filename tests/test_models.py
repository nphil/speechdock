"""Integration coverage for app-owned model downloads and configuration edits."""

import asyncio
import copy
import hashlib
import json
import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from speechdock.catalog import CATALOG
from speechdock.models import ModelError, ModelManager


CONFIG = """# Keep the operator's comments and existing models.
healthCheckTimeout: 300
models:
  whisper-large-v3-turbo:
    cmd: whisper-server --port ${PORT}
    aliases: [whisper-1]
    ttl: 0
  vision:
    cmd: llama-server --port ${PORT}
groups:
  daily:
    swap: false
    exclusive: true
    members: [whisper-large-v3-turbo, vision]
hooks:
  on_startup:
    preload: [whisper-large-v3-turbo, vision]
"""


@pytest.fixture
def upstream():
    class Handler(BaseHTTPRequestHandler):
        payload = b"official-test-model-bytes"
        ids = ["whisper-large-v3-turbo", "vision"]
        config = None

        def do_GET(self):
            if self.path == "/v1/models":
                if self.config is not None:
                    models = YAML(typ="safe").load(self.config.read_text())["models"]
                    data = [{"id": ident, "meta": {"llamaswap": {"aliases": definition.get("aliases", [])}}}
                            for ident, definition in models.items()]
                else:
                    data = [{"id": ident} for ident in self.ids]
                body = json.dumps({"data": data}).encode()
            else:
                body = self.payload
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
async def manager(tmp_path, monkeypatch, upstream):
    base, handler = upstream
    config = tmp_path / "llama" / "config.yaml"
    config.parent.mkdir()
    config.write_text(CONFIG)
    handler.config = config
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("SPDOCK_LLAMA_CONFIG", str(config))
    monkeypatch.setenv("SPDOCK_MODELS_DIR", str(model_dir))
    monkeypatch.setenv("SPDOCK_ENGINE_MODELS_DIR", "/models/whisper/speechdock")
    # The validator must be the real v268 binary, not a fake that accepts bad YAML.
    validator = os.environ.get("SPDOCK_TEST_LLAMA_VALIDATOR") or shutil.which("llama-swap")
    monkeypatch.setenv("SPDOCK_LLAMA_VALIDATOR", validator or str(tmp_path / "missing-validator"))
    item = copy.deepcopy(CATALOG["tiny"])
    item.update(url=f"{base}/model.bin", size_bytes=len(handler.payload),
                sha256=hashlib.sha256(handler.payload).hexdigest())
    monkeypatch.setitem(CATALOG, "tiny", item)
    instance = ModelManager(data_dir, lambda: {"engine_url": base})
    try:
        yield instance, item, config, model_dir, validator
    finally:
        await instance.close()


async def wait_for_download(manager, ident="tiny"):
    async with asyncio.timeout(10):
        while True:
            items = await manager.downloads()
            record = next((item for item in items if item["id"] == ident), None)
            if record and record["status"] in {"completed", "complete", "failed", "cancelled"}:
                return record
            await asyncio.sleep(0.02)


async def test_download_verifies_and_installs_only_catalog_file(manager):
    instance, item, config, model_dir, _ = manager
    await instance.install("tiny")
    result = await wait_for_download(instance)
    assert result["status"] in {"completed", "complete"}, result
    assert (model_dir / item["filename"]).read_bytes() == b"official-test-model-bytes"
    assert config.read_text() == CONFIG
    assert not list(model_dir.glob("*.part"))


async def test_digest_mismatch_never_publishes_model(manager, monkeypatch):
    instance, item, config, model_dir, _ = manager
    monkeypatch.setitem(item, "sha256", "0" * 64)
    await instance.install("tiny")
    result = await wait_for_download(instance)
    assert result["status"] == "failed"
    assert not (model_dir / item["filename"]).exists()
    assert not list(model_dir.glob("*.part"))
    assert config.read_text() == CONFIG


async def test_unowned_file_is_not_overwritten_or_deleted(manager):
    instance, item, config, model_dir, _ = manager
    target = model_dir / item["filename"]
    target.write_bytes(b"operator-owned")
    try:
        await instance.install("tiny")
    except ModelError:
        pass
    else:
        result = await wait_for_download(instance)
        assert result["status"] == "failed"
    with pytest.raises(ModelError):
        await instance.delete("tiny", confirm=True)
    assert target.read_bytes() == b"operator-owned"
    assert config.read_text() == CONFIG


async def test_unknown_catalog_id_is_rejected(manager):
    instance, _, config, _, _ = manager
    with pytest.raises(ModelError):
        await instance.install("../../config.yaml")
    assert config.read_text() == CONFIG


async def test_registration_requires_confirmation_and_validator(manager):
    instance, _, config, _, _ = manager
    await instance.install("tiny")
    result = await wait_for_download(instance)
    assert result["status"] in {"completed", "complete"}, result
    with pytest.raises(ModelError):
        await instance.register("tiny", confirm_reload=False)
    assert config.read_text() == CONFIG
    # A missing validator must never result in an unvalidated live config.
    instance.validator = Path("/nonexistent/speechdock-validator")
    with pytest.raises(ModelError):
        await instance.register("tiny", confirm_reload=True)
    assert config.read_text() == CONFIG


async def test_register_and_unregister_preserve_existing_config(manager):
    instance, _, config, _, validator = manager
    if not validator:
        pytest.skip("Set SPDOCK_TEST_LLAMA_VALIDATOR to the official v268 llama-swap binary")
    await instance.install("tiny")
    result = await wait_for_download(instance)
    assert result["status"] in {"completed", "complete"}, result
    await instance.register("tiny", confirm_reload=True)
    text = config.read_text()
    assert "# Keep the operator's comments" in text
    parsed = YAML().load(text)
    original = YAML().load(CONFIG)
    for name, model in original["models"].items():
        assert parsed["models"][name] == model
    assert parsed["groups"]["daily"] == original["groups"]["daily"]
    assert parsed["hooks"] == original["hooks"]
    assert "speechdock-tiny" in parsed["models"]
    assert parsed["groups"]["speechdock"]["swap"] is True
    assert parsed["groups"]["speechdock"]["exclusive"] is True
    assert parsed["groups"]["speechdock"]["members"] == ["speechdock-tiny"]
    with pytest.raises(ModelError):
        await instance.delete("tiny", confirm=True)
    await instance.unregister("tiny", confirm_reload=True)
    parsed = YAML().load(config.read_text())
    assert "speechdock-tiny" not in parsed["models"]
    for name, model in original["models"].items():
        assert parsed["models"][name] == model
    await instance.delete("tiny", confirm=True)
    assert not (instance.models_dir / CATALOG["tiny"]["filename"]).exists()


async def test_external_edits_to_owned_model_are_not_overwritten(manager):
    instance, _, config, _, validator = manager
    if not validator:
        pytest.skip("Set SPDOCK_TEST_LLAMA_VALIDATOR to the official v268 llama-swap binary")
    await instance.install("tiny")
    result = await wait_for_download(instance)
    assert result["status"] in {"completed", "complete"}, result
    await instance.register("tiny", confirm_reload=True)
    altered = config.read_text().replace("ttl: 180", "ttl: 181")
    config.write_text(altered)
    with pytest.raises(ModelError):
        await instance.unregister("tiny", confirm_reload=True)
    assert config.read_text() == altered


async def test_shared_turbo_alias_is_read_from_engine_metadata(manager):
    instance, _, _, _, _ = manager
    shared = (await instance.list_models())[0]
    assert shared["model_id"] == "whisper-1"
    assert shared["registered"] is True
    assert shared["translation_capable"] is False
    assert shared["verified"] is False


async def test_cancel_immediately_and_refuse_symlink(manager):
    instance, item, _, model_dir, _ = manager
    await instance.install("tiny")
    result = await instance.cancel_download("tiny")
    assert result["status"] == "cancelled"
    target = model_dir / item["filename"]
    outside = model_dir.parent / "outside"
    outside.write_bytes(b"untouched")
    target.symlink_to(outside)
    with pytest.raises(ModelError):
        await instance.install("tiny")
    with pytest.raises(ModelError):
        await instance.delete("tiny", confirm=True)
    assert outside.read_bytes() == b"untouched"


@pytest.mark.parametrize("phase", ["partial", "linked", "published", "unrelated"])
async def test_restart_recovers_only_owned_interrupted_publication(manager, phase):
    instance, item, _, model_dir, _ = manager
    stage = model_dir / ".speechdock-download-abcdefgh"
    stage.mkdir(mode=0o700)
    part = stage / "model.part"
    part.write_bytes(b"official-test-model-bytes")
    info = stage.stat()
    pending = {"directory": stage.name, "stage_identity": [info.st_dev, info.st_ino]}
    final = model_dir / item["filename"]
    if phase in ("linked", "published", "unrelated"):
        info = part.stat()
        pending["file_identity"] = [info.st_dev, info.st_ino]
        if phase == "unrelated":
            final.write_bytes(b"operator file")
        else:
            os.link(part, final)
            if phase == "published":
                part.unlink()
                stage.rmdir()
    instance._state["pending"]["tiny"] = pending
    instance._state["downloads"]["tiny"] = {"id": "tiny", "status": "verifying", "received_bytes": item["size_bytes"], "total_bytes": item["size_bytes"], "progress": 1.0, "error": None}
    instance._save()
    restarted = ModelManager(instance.data_dir, instance.settings_provider)
    try:
        assert not stage.exists()
        model = next(x for x in await restarted.list_models() if x["id"] == "tiny")
        if phase in ("linked", "published"):
            assert model["installed"] and model["verified"]
            assert final.read_bytes() == b"official-test-model-bytes"
        else:
            assert not model["installed"]
            if phase == "unrelated":
                assert final.read_bytes() == b"operator file"
            else:
                assert not final.exists()
    finally:
        await restarted.close()
