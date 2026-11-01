"""Bake the explicitly pinned CPU alignment assets; no downloads occur at service runtime.
Usage: python runtime/prepare_alignment.py DESTINATION
Source model license: Apache-2.0 (facebook/wav2vec2-base-960h; Xenova ONNX export).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import httpx

REPOSITORY = "Xenova/wav2vec2-base-960h"
REVISION = "a19f851b3d42865797e410752b4c570c871e4825"
ASSETS = {
    "onnx/model_quantized.onnx": "cd5040c147381580ed73258143dd8e0c28e800a09e74ee42ee2b3e8cb4d760a3",
    "vocab.json": "4178db26b3c7570f6a47f14ac6a1c7b32950b8c2800fb097287e53776934f1c5",
    "config.json": "ceff8f92e6670ad3633779bd43b3559628c1a64b0868ef6046b23d62b85c69d5",
    "preprocessor_config.json": "8cdfd65ff4115423185a1512bdae100e2e0cd744f5b322417429944aaafd0827",
}


def prepare(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=180) as client:
        for name, expected in ASSETS.items():
            target = destination / Path(name).name
            if target.is_file():
                with target.open("rb") as existing:
                    if hashlib.file_digest(existing, "sha256").hexdigest() == expected:
                        continue
            descriptor, temporary = tempfile.mkstemp(prefix=".alignment-", dir=destination)
            try:
                digest = hashlib.sha256()
                size = 0
                with os.fdopen(descriptor, "wb") as output:
                    with client.stream("GET", f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}") as response:
                        response.raise_for_status()
                        for chunk in response.iter_bytes():
                            size += len(chunk)
                            if size > 128 * 1024 * 1024:
                                raise ValueError("Alignment artifact exceeds pinned size budget")
                            digest.update(chunk)
                            output.write(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError("Alignment artifact checksum mismatch")
                os.chmod(temporary, 0o644)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    shutil.copyfile(Path(__file__).with_name("alignment-LICENSE.txt"), destination / "alignment-LICENSE.txt")
    (destination / "alignment-provenance.json").write_text(json.dumps({
        "repository": REPOSITORY,
        "revision": REVISION,
        "source_model": "facebook/wav2vec2-base-960h",
        "source_model_url": "https://huggingface.co/facebook/wav2vec2-base-960h",
        "export": "Xenova ONNX export",
        "sha256": ASSETS,
        "license": "Apache-2.0",
        "license_file": "alignment-LICENSE.txt",
        "runtime": "onnxruntime CPUExecutionProvider",
        "languages": ["en"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: prepare_alignment.py DESTINATION")
    prepare(Path(sys.argv[1]))
