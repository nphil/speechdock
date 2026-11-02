"""Integration coverage for app-owned model downloads and configuration edits."""

import asyncio
import copy
import hashlib
import json
import os
import shlex
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
                    data = [{"id": ident, "meta": {"llamaswap": {**definition.get("metadata", {}), "aliases": definition.get("aliases", [])}}}
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


async def test_registration_validates_exact_candidate_with_accessible_runtime_store(manager, tmp_path, monkeypatch):
    instance, _, config, _, validator = manager
    if not validator:
        pytest.skip("Set SPDOCK_TEST_LLAMA_VALIDATOR to the official v268 llama-swap binary")

    # The real validator checks store.path in its own filesystem, even with -validate.
    store_dir = tmp_path / "engine-runtime"
    original = CONFIG + f"store:\n  path: {store_dir / 'state.db'}\n"
    config.write_text(original)
    await instance.install("tiny")
    assert (await wait_for_download(instance))["status"] in {"completed", "complete"}

    with pytest.raises(ModelError, match="store.path"):
        await instance.register("tiny", confirm_reload=True)
    assert config.read_text() == original

    # Map the engine namespace to the same config directory, not a copy.
    store = config.parent / "state.db"
    store.write_bytes(b"existing engine store")
    store_dir.symlink_to(config.parent, target_is_directory=True)
    captured = tmp_path / "validated.yaml"
    wrapper = tmp_path / "validator"
    wrapper.write_text(f'#!/bin/sh\ncp "$2" {shlex.quote(str(captured))}\nexec {shlex.quote(str(validator))} "$@"\n')
    wrapper.chmod(0o755)
    instance.validator = wrapper
    definition = instance._definition
    monkeypatch.setattr(instance, "_definition", lambda entry: {**definition(entry), "ttl": -2})
    with pytest.raises(ModelError, match="invalid TTL"):
        await instance.register("tiny", confirm_reload=True)
    assert config.read_text() == original
    assert YAML().load(captured.read_text())["models"]["speechdock-tiny"]["ttl"] == -2

    monkeypatch.setattr(instance, "_definition", definition)
    await instance.register("tiny", confirm_reload=True)
    assert captured.read_bytes() == config.read_bytes()
    assert (instance.data_dir / "llama-swap-config.backup.yaml").read_text() == original
    assert YAML().load(config.read_text())["store"]["path"] == str(store_dir / "state.db")
    assert store.read_bytes() == b"existing engine store"


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


async def test_register_keep_loaded_transaction_via_api(manager, monkeypatch, tmp_path):
    import httpx
    from speechdock.app import app

    instance, tiny, config, _, _ = manager
    large = copy.deepcopy(CATALOG["large-v3"])
    large.update(url=tiny["url"], size_bytes=tiny["size_bytes"], sha256=tiny["sha256"])
    monkeypatch.setitem(CATALOG, "large-v3", large)
    for ident in ("tiny", "large-v3"):
        await instance.install(ident)
        assert (await wait_for_download(instance, ident))["status"] in {"completed", "complete"}
    downloads = copy.deepcopy(await instance.downloads())
    validator = tmp_path / "validator"
    validator.write_text("#!/bin/sh\nexit 0\n")
    validator.chmod(0o755)
    instance.validator = validator
    instance.runtimes["whisper.cpp"] = {"available": True}
    monkeypatch.setattr(app.state, "models", instance, raising=False)

    async def ids():
        return set(YAML(typ="safe").load(config.read_text())["models"])

    async def no_wait(_seconds):
        pass

    monkeypatch.setattr(instance, "_engine_ids", ids)
    monkeypatch.setattr("speechdock.models.asyncio.sleep", no_wait)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        url = "/api/models/tiny/register"
        for invalid in ("true", 1, None, [], {}):
            response = await client.post(url, json={"confirm_reload": True, "keep_loaded": invalid})
            assert response.status_code == 422
        assert (await client.post(url, json={"confirm_reload": "true"})).status_code == 422
        assert (await client.post(url, json={"keep_loaded": True})).status_code == 409
        response = await client.post(url, json={"confirm_reload": True, "keep_loaded": True})
        assert response.status_code == 200 and response.json()["keep_loaded"] is True
        initial = config.read_bytes()
        state = instance.state_path.read_bytes()
        for body in ({"keep_loaded": True}, {}, {"confirm_reload": True, "keep_loaded": True}):
            response = await client.post(url, json=body)
            assert response.status_code == 200 and response.json()["keep_loaded"] is True
            assert config.read_bytes() == initial and instance.state_path.read_bytes() == state
        assert (await client.post(url, json={"keep_loaded": False})).status_code == 409
        assert (await client.post(url, json={"confirm_reload": True, "keep_loaded": False})).json()["keep_loaded"] is False
        parsed = YAML(typ="safe").load(config.read_text())
        assert parsed["models"][tiny["model_id"]]["ttl"] == 180
        assert parsed["groups"]["speechdock"]["exclusive"] is True
        assert (await client.post(url, json={"confirm_reload": True, "keep_loaded": True})).json()["keep_loaded"] is True
        response = await client.post("/api/models/large-v3/register", json={"confirm_reload": True})
        assert response.status_code == 200 and response.json()["keep_loaded"] is False
        parsed = YAML(typ="safe").load(config.read_text())
        assert parsed["models"][tiny["model_id"]]["ttl"] == 0
        assert parsed["models"][large["model_id"]]["ttl"] == 180
        assert parsed["groups"]["speechdock"] == {"swap": False, "exclusive": False, "persistent": True, "members": [tiny["model_id"], large["model_id"]]}
        original = YAML(typ="safe").load(CONFIG)
        assert parsed["groups"]["daily"] == original["groups"]["daily"]
        assert parsed["hooks"] == original["hooks"]
        listed = (await client.get("/api/models")).json()["items"]
        assert next(x for x in listed if x["id"] == "tiny")["keep_loaded"] is True
        assert next(x for x in listed if x["id"] == "large-v3")["keep_loaded"] is False
        assert (await client.delete("/api/models/tiny/registration?confirm_reload=true")).status_code == 200
        parsed = YAML(typ="safe").load(config.read_text())
        assert parsed["groups"]["speechdock"] == {"swap": True, "exclusive": True, "persistent": False, "members": [large["model_id"]]}
        assert await instance.downloads() == downloads

        before_config = config.read_bytes()
        before_state = instance.state_path.read_bytes()
        before_owned = copy.deepcopy(instance._state)
        stale_models = await instance._engine_models()
        async def stale_engine():
            return stale_models
        monkeypatch.setattr(instance, "_engine_models", stale_engine)
        async def no_engine():
            return None
        monkeypatch.setattr(instance, "_engine_ids", no_engine)
        offline = (await client.get("/api/models")).json()["items"]
        assert next(x for x in offline if x["id"] == "large-v3")["keep_loaded"] is False
        assert next(x for x in offline if x["id"] == "tiny")["keep_loaded"] is None
        response = await client.post("/api/models/large-v3/register", json={"confirm_reload": True, "keep_loaded": True})
        assert response.status_code == 503
        assert config.read_bytes() == before_config
        assert instance.state_path.read_bytes() == before_state
        assert instance._state == before_owned


async def test_keep_loaded_rejects_external_definition(manager, monkeypatch, tmp_path):
    instance, tiny, config, _, _ = manager
    await instance.install("tiny")
    assert (await wait_for_download(instance))["status"] in {"completed", "complete"}
    validator = tmp_path / "validator"
    validator.write_text("#!/bin/sh\nexit 0\n")
    validator.chmod(0o755)
    instance.validator = validator
    instance.runtimes["whisper.cpp"] = {"available": True}
    async def ids():
        return {tiny["model_id"]}
    async def no_wait(_seconds):
        pass
    monkeypatch.setattr(instance, "_engine_ids", ids)
    monkeypatch.setattr("speechdock.models.asyncio.sleep", no_wait)
    original = config.read_text()
    config.write_text(original.replace("models:\n", f"models:\n  {tiny['model_id']}:\n    cmd: external\n", 1))
    with pytest.raises(ModelError, match="Model ID or alias already exists"):
        await instance.register("tiny", True, True)
    config.write_text(original)
    await instance.register("tiny", True, True)
    config.write_text(config.read_text().replace("ttl: 0", "ttl: 12"))
    with pytest.raises(ModelError, match="changed externally"):
        await instance.register("tiny", True, False)
