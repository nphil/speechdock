"""Provision pinned executables onto the model mount; never start a model here."""

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path

CRISP_VERSION = "crispasr-0.8.48-cuda12-1"
CRISP_ARCHIVE_SHA256 = "772160483fe5fb43024237fac43563f270bed7877d72b3333282024d070bced4"
BUNDLED_RUNTIME = Path("/opt/speechdock/runtime") / CRISP_VERSION
_REQUIRED = {"crispasr", "libggml-cuda.so", "libggml.so.0", "libggml-base.so.0",
             "libggml-cpu-x64.so", "cuda-required.so", "cuda-launch", "LICENSE", "THIRD_PARTY_NOTICES.txt"}


def _directory(path: Path) -> None:
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError("Runtime directory must be a real directory, not a symbolic link")


def _manifest(source: Path) -> dict:
    _directory(source)
    path = source / "manifest.json"
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("Runtime manifest must be a regular file")
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Invalid runtime manifest")
    files = manifest.get("files")
    if (manifest.get("version") != CRISP_VERSION
            or manifest.get("archive_sha256") != CRISP_ARCHIVE_SHA256
            or not isinstance(files, dict) or not _REQUIRED <= files.keys()):
        raise ValueError("Incompatible or incomplete bundled CrispASR runtime")
    for name, entry in files.items():
        if (not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", "..", "manifest.json"}
                or not isinstance(entry, dict) or entry.get("mode") not in {0o444, 0o555}
                or not re.fullmatch(r"[a-f0-9]{64}", str(entry.get("sha256", "")))):
            raise ValueError("Invalid runtime manifest entry")
    return manifest


def _verify(directory: Path, manifest: dict) -> None:
    _directory(directory)
    if {p.name for p in directory.iterdir()} != {*manifest["files"], "manifest.json"}:
        raise ValueError("Runtime contains missing or unexpected files; it was not overwritten")
    if _manifest(directory) != manifest:
        raise ValueError("Runtime manifest differs from bundled version; it was not overwritten")
    for name, entry in manifest["files"].items():
        path = directory / name
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode) or stat.S_IMODE(mode) != entry["mode"]:
            raise ValueError("Runtime contains an unsafe file or permissions")
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError("Runtime checksum mismatch; it was not overwritten")


def provision_runtimes(owned_dir: Path) -> dict:
    """Copy validated image assets atomically. Availability means files are ready.

    CUDA/driver compatibility is checked when llama-swap launches the executable,
    in its own container. SpeechDock itself does not need or initialize a GPU.
    Existing Whisper executables remain owned by the llama-swap image.
    """
    result = {"whisper.cpp": {"available": True, "version": None, "error": None},
              "crispasr": {"available": False, "version": CRISP_VERSION, "error": None}}
    scratch = None
    try:
        manifest = _manifest(BUNDLED_RUNTIME)
        _verify(BUNDLED_RUNTIME, manifest)
        owned_dir = Path(owned_dir)
        _directory(owned_dir)
        root = owned_dir / ".runtimes"
        root.mkdir(mode=0o755, exist_ok=True)
        _directory(root)
        # A lock prevents two application starts from publishing the same version.
        lock_fd = os.open(root / ".provision.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(lock_fd, "rb+") as lock:
            if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
                raise ValueError("Unsafe runtime provisioning lock")
            fcntl.flock(lock, fcntl.LOCK_EX)
            destination = root / CRISP_VERSION
            if destination.exists() or destination.is_symlink():
                _verify(destination, manifest)
            else:
                scratch = Path(tempfile.mkdtemp(prefix=".provision-", dir=root))
                for name in (*manifest["files"], "manifest.json"):
                    shutil.copyfile(BUNDLED_RUNTIME / name, scratch / name)
                    (scratch / name).chmod(0o444 if name == "manifest.json" else manifest["files"][name]["mode"])
                _verify(scratch, manifest)
                scratch.chmod(0o555)
                os.rename(scratch, destination)
                scratch = None
            result["crispasr"]["available"] = True
    except (OSError, ValueError, TypeError, KeyError) as exc:
        # Do not expose filesystem internals or silently fall back to Whisper/CPU.
        result["crispasr"]["error"] = (str(exc) if isinstance(exc, ValueError)
            else "CrispASR runtime package unavailable; install the CUDA 12 SpeechDock image and check the owned model mount")
    finally:
        if scratch is not None:
            scratch.chmod(0o700)
            shutil.rmtree(scratch)
    return result


def model_command(entry: dict, engine_dir: Path) -> list[str]:
    """The manager adds its existing free-VRAM guard and shell quoting."""
    filename = entry.get("filename")
    if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", filename) or filename in {".", ".."}:
        raise ValueError("Invalid owned model filename")
    model = str(engine_dir / filename)
    profile = (entry.get("runtime", "whisper.cpp"), entry.get("architecture", "whisper"))
    if profile == ("whisper.cpp", "whisper"):
        return ["whisper-server", "--host", "127.0.0.1", "--port", "${PORT}", "-m", model,
                "--request-path", "/v1", "--inference-path", "/audio/transcriptions", "--language", "auto"]
    if profile not in {("crispasr", "parakeet"), ("crispasr", "canary")}:
        raise ValueError("Unsupported speech runtime or architecture")
    runtime = engine_dir / ".runtimes" / CRISP_VERSION
    return ["env", f"GGML_BACKEND_PATH={runtime / 'libggml-cuda.so'}",
            "OPENBLAS_NUM_THREADS=2", str(runtime / "cuda-launch"), "--server", "--host", "127.0.0.1",
            "--port", "${PORT}", "-m", model, "--backend", profile[1], "--gpu-backend", "cuda",
            "--device", "0", "--threads", "2", "--no-auto-aligner", "--lid-backend", "off"]
