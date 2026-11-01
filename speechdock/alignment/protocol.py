"""Versioned wire contract; client data is identity, never a fetch location."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Identity(StrictModel):
    server_machine_id: str = Field(min_length=1, max_length=128)
    rating_key: str = Field(pattern=r"^[0-9]{1,20}$")
    media_id: str = Field(pattern=r"^[0-9]{1,20}$")
    part_id: str = Field(pattern=r"^[0-9]{1,20}$")
    part_key: str = Field(min_length=1, max_length=512)
    audio_stream_id: str = Field(pattern=r"^[0-9]{1,20}$")
    subtitle_stream_id: str = Field(pattern=r"^[0-9]{1,20}$")
    subtitle_codec: str = Field(min_length=1, max_length=32)


class Cue(StrictModel):
    cue_id: str = Field(min_length=1, max_length=160)
    ordinal: int = Field(ge=0, le=2**31 - 1)
    authored_start_ms: int = Field(ge=-86400000, le=86400000)
    authored_end_ms: int = Field(ge=-86400000, le=86400000)
    start_ms: int = Field(ge=-86400000, le=86400000)
    end_ms: int = Field(ge=-86400000, le=86400000)
    text: str = Field(min_length=1, max_length=2048)


class Window(StrictModel):
    protocol: Literal[1]
    request_id: str = Field(min_length=1, max_length=128)
    generation: int = Field(ge=0, le=2**63 - 1)
    playback_id: str = Field(min_length=1, max_length=128)
    identity: Identity
    subtitle_source_kind: Literal["sidecar", "embedded"]
    subtitle_hash_kind: Literal["source_bytes_sha256", "window_manifest_sha256"]
    subtitle_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    language: str = Field(min_length=1, max_length=16)
    timeline: str = Field(min_length=1, max_length=32)
    media_origin_pts_ms: int = Field(ge=-86400000, le=86400000)
    subtitle_offset_ms: int = Field(ge=-3600000, le=3600000)
    window_start_ms: int = Field(ge=-86400000, le=86400000)
    window_end_ms: int = Field(ge=-86400000, le=86400000)
    cues: list[Cue] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def bounds(self):
        if not 0 < self.window_end_ms - self.window_start_ms <= 30000:
            raise ValueError("window bounds")
        if sum(len(cue.text.encode()) for cue in self.cues) > 4096:
            raise ValueError("text budget")
        if len({cue.cue_id for cue in self.cues}) != len(self.cues):
            raise ValueError("duplicate cue id")
        for cue in self.cues:
            if not 0 < cue.authored_end_ms - cue.authored_start_ms <= 15000:
                raise ValueError("authored bounds")
            if not self.window_start_ms <= cue.start_ms < cue.end_ms <= self.window_end_ms:
                raise ValueError("cue outside window")
            if not cue.authored_start_ms + self.subtitle_offset_ms <= cue.start_ms < cue.end_ms <= cue.authored_end_ms + self.subtitle_offset_ms:
                raise ValueError("visibility exceeds authored offset bounds")
        expected = "source_bytes_sha256" if self.subtitle_source_kind == "sidecar" else "window_manifest_sha256"
        if self.subtitle_hash_kind != expected:
            raise ValueError("hash kind")
        return self


def manifest_hash(cues: list[Cue]) -> str:
    manifest = [[cue.ordinal, cue.authored_start_ms, cue.authored_end_ms, cue.text] for cue in cues]
    return hashlib.sha256(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
