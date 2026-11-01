"""Pinned official multilingual whisper.cpp model artifacts."""

REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"
BASE_URL = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{REVISION}"

# LFS SHA-256 and size from the pinned upstream repository, not downloaded metadata.
_DATA = (
    ("tiny", 77691713, "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21", "Basic", "Fastest", 2048),
    ("base", 147951465, "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe", "Basic", "Fast", 2048),
    ("small", 487601967, "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b", "Good", "Moderate", 3072),
    ("medium", 1533763059, "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208", "Better", "Slower", 6144),
    ("large-v3-turbo-q8_0", 874188075, "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1", "High", "Fast", 8192),
    ("large-v3-q5_0", 1081140203, "d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1", "High", "Slower", 8192),
    ("large-v3", 3095033483, "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2", "Highest", "Slowest", 8192),
)

CATALOG = {
    ident: {
        "id": ident, "name": f"Whisper {ident}", "model_id": f"speechdock-{ident}",
        "filename": f"ggml-{ident}.bin", "size_bytes": size, "sha256": digest,
        "url": f"{BASE_URL}/ggml-{ident}.bin", "multilingual": True,
        "translation_capable": "turbo" not in ident, "quality": quality,
        "speed": speed, "free_mib": free,
        "description": "Official multilingual Whisper model.",
    }
    for ident, size, digest, quality, speed, free in _DATA
}
