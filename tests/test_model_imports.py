"""Consumer-visible safety and lifecycle coverage for manual model imports."""

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager

import pytest

from speechdock import model_formats, model_transfer
from speechdock.models import ModelError, ModelManager


LIMIT = 8 * 1024**3
PAYLOAD = b"example model payload"
PROFILE = {
    "runtime": "whisper.cpp", "format": "ggml", "architecture": "whisper",
    "multilingual": True, "translation_capable": True,
    "language_detection_capable": True, "requires_language": False,
}


@pytest.fixture
async def manager(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setenv("SPDOCK_MODELS_DIR", str(models))
    instance = ModelManager(tmp_path / "data", lambda: {})

    async def engine_ids():
        return set()

    monkeypatch.setattr(instance, "_engine_ids", engine_ids)
    try:
        yield instance
    finally:
        await instance.close()


async def chunks(*parts):
    for part in parts:
        yield part


async def create_upload(manager, **options):
    result = await manager.create_import(name="My model", source="upload", **options)
    assert result["download"]["status"] == "awaiting_upload"
    return result["model"]["id"]


async def model(manager, ident):
    return next(item for item in await manager.list_models() if item["id"] == ident)


def path_for(manager, ident):
    return manager.models_dir / manager._entry(ident)["filename"]


def assert_no_staging(manager):
    assert not list(manager.models_dir.glob(".speechdock-download-*"))


async def test_upload_publishes_verified_profile_and_can_be_removed(manager, monkeypatch):
    monkeypatch.setattr(model_formats, "inspect_model", lambda path: PROFILE.copy())
    digest = hashlib.sha256(PAYLOAD).hexdigest()
    ident = await create_upload(manager, expected_sha256=digest.upper(), expected_size_bytes=len(PAYLOAD))
    result = await manager.receive_upload(ident, chunks(PAYLOAD[:7], PAYLOAD[7:]), len(PAYLOAD))
    assert result["status"] == "completed"
    installed = await model(manager, ident)
    assert installed["installed"] and installed["verified"]
    assert installed["sha256"] == digest
    assert installed["size_bytes"] == len(PAYLOAD)
    assert installed["checksum_status"] == "matched"
    assert all(installed[key] == value for key, value in PROFILE.items())
    assert path_for(manager, ident).read_bytes() == PAYLOAD
    assert_no_staging(manager)
    target = path_for(manager, ident)
    assert (await manager.remove_import(ident, confirm=True))["deleted"]
    assert not target.exists()
    assert all(item["id"] != ident for item in await manager.list_models())
    assert not list(manager.models_dir.glob("import-*.model"))


@pytest.mark.parametrize(
    ("options", "payload", "length", "message"),
    [
        ({"expected_sha256": "0" * 64}, PAYLOAD, len(PAYLOAD), "checksum"),
        ({}, b"short", 10, "truncated"),
        ({}, b"not a supported format", None, "Unsupported"),
    ],
)
async def test_bad_upload_is_failed_and_never_published(manager, options, payload, length, message):
    ident = await create_upload(manager, **options)
    with pytest.raises(ModelError, match=message):
        await manager.receive_upload(ident, chunks(payload), length)
    record = next(item for item in await manager.downloads() if item["id"] == ident)
    assert record["status"] == "failed"
    assert message.lower() in record["error"].lower()
    assert not (await model(manager, ident))["installed"]
    assert not path_for(manager, ident).exists()
    assert_no_staging(manager)


async def test_declared_size_and_stream_are_bounded_at_8_gib(manager):
    with pytest.raises(ModelError):
        await create_upload(manager, expected_size_bytes=LIMIT + 1)
    ident = await create_upload(manager)
    with pytest.raises(ModelError, match="8 GiB"):
        await manager.receive_upload(ident, chunks(b"unused"), LIMIT + 1)
    assert not path_for(manager, ident).exists()
    assert_no_staging(manager)

    # Simulate an oversized chunk without allocating eight gigabytes. The receiver
    # must reject its reported length before trying to write its contents.
    class OversizedChunk(bytes):
        def __len__(self):
            return LIMIT + 1

    ident = await create_upload(manager)
    with pytest.raises(ModelError, match="8 GiB"):
        await manager.receive_upload(ident, chunks(OversizedChunk(b"x")))
    record = next(item for item in await manager.downloads() if item["id"] == ident)
    assert record["status"] == "failed"
    assert not path_for(manager, ident).exists()
    assert_no_staging(manager)


async def test_cancel_awaiting_upload_survives_restart_and_can_be_removed(manager, monkeypatch):
    ident = await create_upload(manager)
    assert (await manager.cancel_download(ident))["status"] == "cancelled"
    with pytest.raises(ModelError):
        await manager.receive_upload(ident, chunks(PAYLOAD))
    restarted = ModelManager(manager.data_dir, lambda: {})
    monkeypatch.setattr(restarted, "_engine_ids", manager._engine_ids)
    try:
        record = next(item for item in await restarted.downloads() if item["id"] == ident)
        assert record["status"] == "cancelled"
        await restarted.remove_import(ident, confirm=True)
        assert all(item["id"] != ident for item in await restarted.list_models())
    finally:
        await restarted.close()


async def test_interrupted_upload_is_marked_failed_after_restart(manager, monkeypatch):
    ident = await create_upload(manager)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def interrupted_stream():
        yield b"partial"
        entered.set()
        await release.wait()

    upload = asyncio.create_task(manager.receive_upload(ident, interrupted_stream()))
    await asyncio.wait_for(entered.wait(), 5)
    # Save the in-progress state as a crash would leave it, then stop this instance.
    manager._save()
    saved = manager.state_path.read_bytes()
    upload.cancel()
    with pytest.raises(asyncio.CancelledError):
        await upload
    manager.state_path.write_bytes(saved)
    restarted = ModelManager(manager.data_dir, lambda: {})
    monkeypatch.setattr(restarted, "_engine_ids", manager._engine_ids)
    try:
        record = next(item for item in await restarted.downloads() if item["id"] == ident)
        assert record["status"] == "failed"
        assert "restart" in record["error"].lower()
        assert not (await model(restarted, ident))["installed"]
        assert not path_for(restarted, ident).exists()
        assert_no_staging(restarted)
    finally:
        await restarted.close()


@pytest.mark.parametrize("kind", ["foreign", "symlink"])
async def test_foreign_destination_is_never_overwritten_or_removed(manager, tmp_path, kind):
    ident = await create_upload(manager)
    target = path_for(manager, ident)
    outside = tmp_path / "outside"
    outside.write_bytes(b"do not touch")
    if kind == "symlink":
        target.symlink_to(outside)
    else:
        target.write_bytes(b"foreign")
    with pytest.raises(ModelError):
        await manager.receive_upload(ident, chunks(PAYLOAD))
    with pytest.raises(ModelError):
        await manager.remove_import(ident, confirm=True)
    assert outside.read_bytes() == b"do not touch"
    assert target.is_symlink() if kind == "symlink" else target.read_bytes() == b"foreign"
    assert_no_staging(manager)


async def test_registered_or_modified_owned_model_cannot_be_removed(manager, monkeypatch):
    monkeypatch.setattr(model_formats, "inspect_model", lambda path: PROFILE.copy())
    ident = await create_upload(manager)
    await manager.receive_upload(ident, chunks(PAYLOAD))
    target = path_for(manager, ident)
    manager._state["registrations"][ident] = "owned registration"
    with pytest.raises(ModelError, match="[Uu]nregister"):
        await manager.remove_import(ident, confirm=True)
    assert target.read_bytes() == PAYLOAD
    del manager._state["registrations"][ident]
    target.write_bytes(b"externally modified")
    with pytest.raises(ModelError, match="owned"):
        await manager.remove_import(ident, confirm=True)
    assert target.read_bytes() == b"externally modified"


async def test_url_and_query_secret_are_not_persisted_or_exposed(manager, monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    secret_url = "https://models.example.org/private/model.bin?token=secret-token-value"

    @asynccontextmanager
    async def transfer(url):
        assert url == secret_url
        started.set()
        await release.wait()
        yield None

    monkeypatch.setattr(model_transfer, "stream_model", transfer)
    result = await manager.create_import(name="Remote model", source="url", url=secret_url)
    ident = result["model"]["id"]
    await asyncio.wait_for(started.wait(), 5)
    visible = json.dumps({"created": result, "models": await manager.list_models(),
                          "downloads": await manager.downloads()})
    persisted = manager.state_path.read_text()
    for text in (visible, persisted):
        assert secret_url not in text
        assert "secret-token-value" not in text
        assert "/private/model.bin" not in text
    assert (await manager.cancel_download(ident))["status"] == "cancelled"


async def test_url_import_accepts_h11_bytearray_chunks(manager, monkeypatch):
    monkeypatch.setattr(model_formats, "inspect_model", lambda path: PROFILE.copy())

    class Transfer:
        total_bytes = len(PAYLOAD)

        async def aiter_bytes(self):
            yield bytearray(PAYLOAD[:7])
            yield bytearray(PAYLOAD[7:])

    @asynccontextmanager
    async def transfer(_url):
        yield Transfer()

    monkeypatch.setattr(model_transfer, "stream_model", transfer)
    result = await manager.create_import(name="Remote", source="url", url="https://models.example.org/model.bin")
    ident = result["model"]["id"]
    await manager._tasks[ident]
    record = next(item for item in await manager.downloads() if item["id"] == ident)
    assert record["status"] == "completed"
    assert path_for(manager, ident).read_bytes() == PAYLOAD
    assert (await model(manager, ident))["verified"]


async def test_interrupted_url_reports_safe_reason_and_does_not_publish(manager, monkeypatch):
    class Transfer:
        total_bytes = 20

        async def aiter_bytes(self):
            yield b"partial"
            raise model_transfer.ModelTransferError("Model download response was incomplete or invalid")

    @asynccontextmanager
    async def transfer(_url):
        yield Transfer()

    monkeypatch.setattr(model_transfer, "stream_model", transfer)
    result = await manager.create_import(name="Remote", source="url", url="https://models.example.org/file?token=secret")
    ident = result["model"]["id"]
    await manager._tasks[ident]
    record = next(item for item in await manager.downloads() if item["id"] == ident)
    assert record["status"] == "failed"
    assert record["error"] == "Model download response was incomplete or invalid"
    assert "secret" not in manager.state_path.read_text()
    assert not path_for(manager, ident).exists()
    assert_no_staging(manager)
