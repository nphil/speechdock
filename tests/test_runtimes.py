import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from speechdock import runtimes


def bundle(tmp_path, monkeypatch):
    source = tmp_path / "bundle"
    source.mkdir()
    files = {}
    for name in runtimes._REQUIRED:
        path = source / name
        path.write_bytes(name.encode())
        path.chmod(0o555)
        files[name] = {"sha256": hashlib.sha256(name.encode()).hexdigest(), "mode": 0o555}
    manifest = {"version": runtimes.CRISP_VERSION, "archive_sha256": runtimes.CRISP_ARCHIVE_SHA256,
                "files": files}
    (source / "manifest.json").write_text(json.dumps(manifest))
    (source / "manifest.json").chmod(0o444)
    monkeypatch.setattr(runtimes, "BUNDLED_RUNTIME", source)
    owned = tmp_path / "owned"
    owned.mkdir()
    return source, owned


def test_absent_package_preserves_whisper_without_creating_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(runtimes, "BUNDLED_RUNTIME", tmp_path / "absent")
    result = runtimes.provision_runtimes(tmp_path)
    assert result["whisper.cpp"]["available"] is True
    assert result["crispasr"]["available"] is False
    assert "unavailable" in result["crispasr"]["error"]
    assert not (tmp_path / ".runtimes").exists()


def test_provision_is_immutable_and_starts_no_process(tmp_path, monkeypatch):
    source, owned = bundle(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("Provisioning must not start processes")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    first = runtimes.provision_runtimes(owned)
    assert first["crispasr"]["available"]
    installed = owned / ".runtimes" / runtimes.CRISP_VERSION
    inode = (installed / "crispasr").stat().st_ino
    assert installed.stat().st_mode & 0o777 == 0o555
    assert runtimes.provision_runtimes(owned) == first
    assert (installed / "crispasr").stat().st_ino == inode
    assert (installed / "crispasr").read_bytes() == (source / "crispasr").read_bytes()


def test_corrupt_existing_runtime_is_not_overwritten(tmp_path, monkeypatch):
    _, owned = bundle(tmp_path, monkeypatch)
    assert runtimes.provision_runtimes(owned)["crispasr"]["available"]
    target = owned / ".runtimes" / runtimes.CRISP_VERSION / "crispasr"
    target.chmod(0o755)
    target.write_bytes(b"corrupt")
    target.chmod(0o555)
    result = runtimes.provision_runtimes(owned)
    assert not result["crispasr"]["available"]
    assert "checksum mismatch" in result["crispasr"]["error"]
    assert target.read_bytes() == b"corrupt"


def test_symlink_runtime_root_rejected(tmp_path, monkeypatch):
    _, owned = bundle(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (owned / ".runtimes").symlink_to(elsewhere, target_is_directory=True)
    assert not runtimes.provision_runtimes(owned)["crispasr"]["available"]
    assert not list(elsewhere.iterdir())


def test_untrusted_manifest_path_rejected(tmp_path, monkeypatch):
    source, owned = bundle(tmp_path, monkeypatch)
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../escape"] = {"mode": 0o555, "sha256": "a" * 64}
    path.chmod(0o644)
    path.write_text(json.dumps(manifest))
    assert not runtimes.provision_runtimes(owned)["crispasr"]["available"]
    assert not (owned / ".runtimes").exists()


@pytest.mark.parametrize("value", [None, [], 1, "invalid"])
def test_non_object_manifest_reports_unavailable(tmp_path, monkeypatch, value):
    source, owned = bundle(tmp_path, monkeypatch)
    path = source / "manifest.json"
    path.chmod(0o644)
    path.write_text(json.dumps(value))
    result = runtimes.provision_runtimes(owned)["crispasr"]
    assert not result["available"]
    assert result["error"] == "Invalid runtime manifest"


def test_commands_preserve_whisper_and_scope_cuda_guard():
    root = Path("/models/whisper/speechdock")
    whisper = runtimes.model_command({"filename": "ggml-small.bin"}, root)
    assert whisper == ["whisper-server", "--host", "127.0.0.1", "--port", "${PORT}", "-m",
                       str(root / "ggml-small.bin"), "--request-path", "/v1",
                       "--inference-path", "/audio/transcriptions", "--language", "auto"]
    for architecture in ("parakeet", "canary"):
        command = runtimes.model_command({"filename": "owned.gguf", "runtime": "crispasr", "architecture": architecture}, root)
        assert "${PORT}" in command
        assert command[command.index("--gpu-backend") + 1] == "cuda"
        assert any(value.endswith("/cuda-launch") for value in command)
        assert not any(value.startswith("LD_PRELOAD=") for value in command)
        assert "--no-auto-aligner" in command
    with pytest.raises(ValueError, match="Unsupported"):
        runtimes.model_command({"filename": "owned.gguf", "runtime": "crispasr", "architecture": "llama"}, root)
    with pytest.raises(ValueError, match="filename"):
        runtimes.model_command({"filename": "../foreign.gguf"}, root)


def test_native_launcher_rejects_missing_or_unloadable_guard(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler required for native launch-guard test")
    source = Path(__file__).parents[1] / "runtime" / "cuda_launch.c"
    launcher = tmp_path / "cuda-launch"
    subprocess.run([compiler, "-O2", "-Wall", "-Wextra", "-Werror", "-o", str(launcher), str(source), "-ldl"], check=True)
    preload_before = os.environ.get("LD_PRELOAD")
    for content, error in [(None, "missing or unsafe"), (b"not an ELF library", "cannot load")]:
        if content is not None:
            (tmp_path / "cuda-required.so").write_bytes(content)
        result = subprocess.run([str(launcher), "--version"], capture_output=True, text=True)
        assert result.returncode == 78
        assert error in result.stderr
    assert os.environ.get("LD_PRELOAD") == preload_before
