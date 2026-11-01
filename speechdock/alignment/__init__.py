"""Alignment integration: attach_alignment_routes(app), then in the host lifespan use
`async with alignment_lifespan(app): ... yield ...`. Never load models on import.

Configuration: SPDOCK_ALIGNMENT_PLEX_URL (one trusted base URL),
SPDOCK_ALIGNMENT_PLEX_TOKEN_FILE, SPDOCK_ALIGNMENT_ROOTS (JSON list of
{"plex":"/PMS/root","local":"/read-only/container/root"}),
SPDOCK_ALIGNMENT_MODEL_DIR (checksummed baked assets). Missing configuration is
an explicit estimated response, not a startup failure. Mount media read-only.
One CPU worker, one waiting request, no result cache or persistent request data.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from .core import ALGORITHM, MODEL_SHA256, PROVENANCE, AcousticAligner, Rejected
from .protocol import Window
from .source import Configuration, PlexSource, decode_audio, open_media, probe, revision, stream_index


def response(request: Window, reason: str) -> dict:
    body = request.model_dump()
    body["request_cues"] = body.pop("cues")
    body.update(source_revision=None, algorithm=ALGORITHM, model_sha256=MODEL_SHA256, timing_provenance=PROVENANCE,
                cues=[{"cue_id": cue.cue_id, "status": "estimated", "reason": reason, "words": []} for cue in request.cues])
    return body


class AlignmentService:
    def __init__(self, config: Configuration | None = None):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speechdock-alignment")
        self.pending = 0
        self.closed = False
        self.aligner = None

    async def submit(self, request: Window) -> dict:
        if self.closed or self.pending >= 2:
            return response(request, "alignment_busy")
        self.pending += 1
        # Shield cancellation: a disconnected caller must not free a slot while its CPU work runs.
        future = asyncio.get_running_loop().run_in_executor(self.executor, self.process, request)
        future.add_done_callback(lambda _: setattr(self, "pending", self.pending - 1))
        return await asyncio.shield(future)

    async def close(self):
        self.closed = True
        await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)
        self.aligner = None

    def process(self, request: Window) -> dict:
        result = response(request, "alignment_unavailable")
        try:
            if request.language not in {"en", "eng"}:
                raise Rejected("unsupported_language")
            if request.timeline != "container_pts":
                raise Rejected("timeline_unverified")
            if sum(cue.end_ms - cue.start_ms for cue in request.cues) > 60000:
                raise Rejected("audio_work_budget")
            config = self.config or Configuration.environment()
            source = PlexSource(config)
            resolved = source.resolve(request)
            with open_media(resolved["part"]["file"], config.roots) as fd:
                initial = revision(fd)
                inventory = probe(fd)
                audio_index = stream_index(resolved["audio"], inventory, "audio")
                subtitle = source.subtitle_bytes(resolved, request, fd, inventory)
                rejected_cues = source.verify_cues(request, subtitle)
                if self.aligner is None:
                    self.aligner = AcousticAligner(config.model_dir)
                for cue, output in zip(request.cues, result["cues"]):
                    try:
                        if cue.cue_id in rejected_cues:
                            raise Rejected(rejected_cues[cue.cue_id])
                        if any(other.cue_id != cue.cue_id and max(other.start_ms, cue.start_ms) < min(other.end_ms, cue.end_ms) for other in request.cues):
                            raise Rejected("overlapping_cues")
                        pcm, origin = decode_audio(fd, audio_index, cue.start_ms, cue.end_ms, request.media_origin_pts_ms, inventory)
                        aligned = self.aligner.align(pcm, cue.text, origin, cue.start_ms, cue.end_ms)
                        output.update(status="aligned", reason="acoustic_match", words=aligned["words"])
                    except Rejected as exc:
                        output.update(reason=str(exc))
                # Fence both opened inode and the current path; replacement is not detected by fstat alone.
                with open_media(resolved["part"]["file"], config.roots) as current_fd:
                    if revision(fd) != initial or revision(current_fd) != initial:
                        raise Rejected("source_revision_changed")
                if source.resolve(request) != resolved:
                    raise Rejected("source_revision_changed")
                if request.subtitle_source_kind == "sidecar" and source.subtitle_bytes(resolved, request, fd, inventory) != subtitle:
                    raise Rejected("source_revision_changed")
                revision_data = json.dumps([resolved, initial, hashlib.sha256(subtitle).hexdigest()], sort_keys=True, separators=(",", ":")).encode()
                result["source_revision"] = hashlib.sha256(revision_data).hexdigest()
                return result
        except Rejected as exc:
            return response(request, str(exc))
        except Exception:
            # Tool/library errors can include tokens or media paths. Never expose their strings.
            return response(request, "alignment_unavailable")


@asynccontextmanager
async def alignment_lifespan(app: FastAPI, config: Configuration | None = None):
    service = AlignmentService(config)
    app.state.alignment = service
    try:
        yield service
    finally:
        await service.close()
        del app.state.alignment


def attach_alignment_routes(app: FastAPI) -> None:
    @app.post("/api/alignment/windows")
    async def align_window(request: Request):
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > 32768:
                raise HTTPException(413, "Alignment request too large")
            raw.extend(chunk)
        try:
            payload = Window.model_validate_json(raw)
        except (ValidationError, ValueError):
            raise HTTPException(422, "Invalid alignment window") from None
        service = getattr(request.app.state, "alignment", None)
        if service is None:
            return response(payload, "alignment_not_started")
        return await service.submit(payload)
