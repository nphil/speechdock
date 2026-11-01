"""Build-only packaging of the checksum-pinned, flat upstream CUDA archive."""
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

ARCHIVE_SHA256 = "772160483fe5fb43024237fac43563f270bed7877d72b3333282024d070bced4"
VERSION = "crispasr-0.8.48-cuda12-1"


def package(archive: Path, guard: Path, launcher: Path, destination: Path) -> None:
    with archive.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("CrispASR archive checksum mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            parts = Path(member.name).parts
            if member.isdir() and parts == ("crispasr-linux-x86_64-cuda",):
                continue
            if (not member.isfile() or len(parts) != 2
                    or parts[0] != "crispasr-linux-x86_64-cuda"
                    or parts[1] in {".", "..", "manifest.json", "cuda-required.so", "cuda-launch"}):
                raise ValueError("Unexpected CrispASR archive member")
            target = destination / parts[1]
            with source.extractfile(member) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            target.chmod(0o555 if member.mode & 0o111 else 0o444)
    shutil.copyfile(guard, destination / "cuda-required.so")
    (destination / "cuda-required.so").chmod(0o555)
    shutil.copyfile(launcher, destination / "cuda-launch")
    (destination / "cuda-launch").chmod(0o555)
    files = {}
    for path in sorted(destination.iterdir()):
        with path.open("rb") as source:
            files[path.name] = {"sha256": hashlib.file_digest(source, "sha256").hexdigest(),
                                "mode": path.stat().st_mode & 0o777}
    required = {"crispasr", "libggml-cuda.so", "libggml.so.0", "libggml-base.so.0",
                "libggml-cpu-x64.so", "cuda-required.so", "cuda-launch", "LICENSE", "THIRD_PARTY_NOTICES.txt"}
    if not required <= files.keys():
        raise ValueError("Incomplete CrispASR runtime")
    manifest = {"version": VERSION, "archive_sha256": ARCHIVE_SHA256,
                "source_revision": "cc114d2b4a5a9413738f390093e9131576743cb6", "files": files}
    (destination / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    (destination / "manifest.json").chmod(0o444)
    destination.chmod(0o555)


if __name__ == "__main__":
    package(*(Path(arg) for arg in sys.argv[1:]))
