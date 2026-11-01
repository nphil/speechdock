"""Structural validation of untrusted model files without large fixtures."""

import struct

import pytest

from speechdock.model_formats import MAX_MODEL_BYTES, inspect_model


def string(value):
    value = value.encode()
    return struct.pack("<Q", len(value)) + value


def gguf(*, architecture="parakeet", tensors=None, data=b"", version=3):
    """Build a small GGUF with actual tensor descriptors and payload bytes."""
    if tensors is None:
        tensors = [("weight", (1,), 0, 0)]
    header = b"GGUF" + struct.pack("<IQQ", version, len(tensors), 1)
    header += string("general.architecture") + struct.pack("<I", 8) + string(architecture)
    for name, dimensions, tensor_type, offset in tensors:
        header += string(name) + struct.pack("<I", len(dimensions))
        header += struct.pack("<" + "Q" * len(dimensions), *dimensions)
        header += struct.pack("<IQ", tensor_type, offset)
    header += b"\0" * (-len(header) % 32)
    return header + data


def inspect_bytes(tmp_path, contents):
    path = tmp_path / "model.bin"
    path.write_bytes(contents)
    return inspect_model(path)


@pytest.mark.parametrize("contents", [b"", b"xyz", b"not a model", b"GGUF", b"lmgg"])
def test_rejects_unknown_empty_and_truncated_headers(tmp_path, contents):
    with pytest.raises(ValueError):
        inspect_bytes(tmp_path, contents)


@pytest.mark.parametrize("version", [0, 1, 4, 0xffffffff])
def test_rejects_unsupported_gguf_versions(tmp_path, version):
    with pytest.raises(ValueError, match="GGUF"):
        inspect_bytes(tmp_path, gguf(version=version))


def test_rejects_non_speech_gguf_architecture(tmp_path):
    with pytest.raises(ValueError, match="architecture"):
        inspect_bytes(tmp_path, gguf(architecture="llama"))


def test_rejects_speech_architecture_without_required_schema(tmp_path):
    # Claiming to be a speech model is not sufficient: real metadata and
    # mandatory tensors must be present even when the tensor data fits.
    with pytest.raises(ValueError):
        inspect_bytes(tmp_path, gguf(data=b"\0" * 4))


@pytest.mark.parametrize("dimensions,tensor_type", [
    ((0,), 0),                 # zero-sized tensor
    ((1,), 999),               # unknown data type
    ((31,), 2),                # quantized blocks require a multiple of 32
    ((MAX_MODEL_BYTES, 2), 0), # tensor is larger than the entire allowed file
])
def test_rejects_invalid_tensor_shapes_and_types(tmp_path, dimensions, tensor_type):
    with pytest.raises(ValueError):
        inspect_bytes(tmp_path, gguf(tensors=[("weight", dimensions, tensor_type, 0)]))


def test_rejects_tensor_data_beyond_end_of_file(tmp_path):
    with pytest.raises(ValueError, match="truncated"):
        inspect_bytes(tmp_path, gguf(tensors=[("weight", (2,), 0, 0)], data=b"\0" * 4))


def test_rejects_overlapping_tensor_ranges(tmp_path):
    tensors = [("first", (16,), 0, 0), ("second", (16,), 0, 32)]
    with pytest.raises(ValueError, match="Overlapping"):
        inspect_bytes(tmp_path, gguf(tensors=tensors, data=b"\0" * 96))


def test_rejects_unaligned_and_duplicate_tensors(tmp_path):
    for tensors in ([('weight', (1,), 0, 1)],
                    [('weight', (1,), 0, 0), ('weight', (1,), 0, 32)]):
        with pytest.raises(ValueError):
            inspect_bytes(tmp_path, gguf(tensors=tensors, data=b"\0" * 64))


def test_rejects_unbounded_metadata_string_length(tmp_path):
    contents = b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + struct.pack("<Q", 2**63)
    with pytest.raises(ValueError):
        inspect_bytes(tmp_path, contents)


def test_rejects_oversized_sparse_model_before_reading(tmp_path):
    path = tmp_path / "oversized.model"
    with path.open("wb") as output:
        output.truncate(MAX_MODEL_BYTES + 1)
    with pytest.raises(ValueError, match="8 GiB"):
        inspect_model(path)


def test_rejects_truncated_whisper_metadata(tmp_path):
    # A supported Whisper magic and dimensions do not make incomplete mel data valid.
    header = struct.pack("<11i", 51865, 1500, 384, 6, 4, 448, 384, 6, 4, 80, 1)
    with pytest.raises(ValueError):
        inspect_bytes(tmp_path, b"lmgg" + header + struct.pack("<2i", 80, 201))


@pytest.mark.parametrize("value,value_type", [(None, None), (8, 4), (7, 4), (8, 5)])
def test_canary_frame_duration_metadata(tmp_path, value, value_type):
    metadata = {
        "d_model": 64, "enc_n_layers": 1, "dec_n_layers": 1, "n_heads": 1,
        "ff_dim": 128, "vocab_size": 1, "max_dec_ctx": 100,
        "sample_rate": 16000, "n_mels": 128, "n_fft": 512,
        "win_length": 400, "hop_length": 160, "head_dim": 64,
        "subsampling_factor": 8, "subsampling_channels": 1, "conv_kernel": 3,
    }
    entries = [("general.architecture", 8, string("canary"))]
    entries += [("canary." + key, 4, struct.pack("<I", number)) for key, number in metadata.items()]
    entries.append(("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 1) + string("token")))
    if value is not None:
        entries.append(("canary.frame_dur_cs", value_type, struct.pack("<I", value)))
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, len(entries))
    for key, typ, encoded in entries:
        header += string(key) + struct.pack("<I", typ) + encoded
    header += string("weight") + struct.pack("<I Q I Q", 1, 1, 0, 0)
    contents = header + b"\0" * (-len(header) % 32) + b"\0" * 4
    # Valid duration proceeds to tensor-schema validation; this intentionally
    # incomplete fixture cannot be mistaken for a fully valid model.
    expected = "Missing or invalid speech tensor" if value in (None, 8) and value_type in (None, 4) else "frame duration"
    with pytest.raises(ValueError, match=expected):
        inspect_bytes(tmp_path, contents)
