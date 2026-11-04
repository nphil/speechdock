"""HTTP endpoints for dub captions: one video, a batch, the Sonarr/Radarr webhook and the ledger."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from ..jobs import QueueFull
from . import media

MAX_WEBHOOK_BYTES = 2 * 1024 * 1024
WEBHOOK_EVENTS = frozenset({"download", "upgrade"})  # Sonarr/Radarr "On Import" and "On Upgrade"


class CaptionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    force: bool = Field(default=False, strict=True)


class BatchBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paths: list[str] = Field(min_length=1, max_length=5000)
    force: bool = Field(default=False, strict=True)


def arr_paths(payload: dict) -> list[str]:
    """Imported file paths of a Sonarr (``episodeFile``/``episodeFiles``) or Radarr (``movieFile``) event."""
    files = [payload[key] for key in ("episodeFile", "movieFile") if isinstance(payload.get(key), dict)]
    files += [item for item in payload.get("episodeFiles") or [] if isinstance(item, dict)]
    root = (payload.get("series") or {}).get("path") or (payload.get("movie") or {}).get("folderPath")
    paths = []
    for item in files:
        path = item.get("path")
        if not path and isinstance(root, str) and isinstance(item.get("relativePath"), str):
            path = root.rstrip("/") + "/" + item["relativePath"]
        if isinstance(path, str):
            paths.append(path)
    return paths


def attach_caption_routes(app: FastAPI, model_for: Callable[[FastAPI, str, str], Awaitable[dict]]) -> None:
    router = APIRouter(prefix="/api/captions", tags=["captions"])

    async def transcription_model(request: Request) -> str:
        model = request.app.state.store.get_settings()["transcription_model"]
        try:
            await model_for(request.app, model, "transcribe")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return model

    async def queue(request: Request, paths: list[str], *, source: str, force: bool) -> dict:
        model = await transcription_model(request)
        queued, existing, rejected = [], [], []
        for value in dict.fromkeys(paths):
            try:
                path = str(await asyncio.to_thread(media.resolve_media_path, value))
                job, created = await request.app.state.jobs.submit_captions(source=source, path=path, force=force, model=model)
            except (media.MediaError, QueueFull) as exc:
                rejected.append({"path": value, "error": str(exc)})
                continue
            (queued if created else existing).append({"path": path, "job_id": job["id"]})
        return {"queued": queued, "already_queued": existing, "rejected": rejected}

    @router.post("", status_code=202)
    async def create(request: Request, body: CaptionBody):
        """Queue one video; returns its job. A path outside the media mounts is rejected with 422."""
        model = await transcription_model(request)
        try:
            path = str(await asyncio.to_thread(media.resolve_media_path, body.path))
        except media.MediaError as exc:
            raise HTTPException(422, str(exc)) from exc
        job, _ = await request.app.state.jobs.submit_captions(source="captions", path=path, force=body.force, model=model)
        return job

    @router.post("/batch", status_code=202)
    async def batch(request: Request, body: BatchBody):
        """Queue many videos; each is reported as queued, already queued or rejected."""
        return await queue(request, body.paths, source="captions", force=body.force)

    @router.post("/webhook")
    async def webhook(request: Request):
        """Sonarr/Radarr Connect (Webhook) target: caption each file on import or upgrade.

        Answers 200 for every event it understands so the *arr app reports no failure; files are
        queued only while the auto-captions setting is on.
        """
        raw = await request.body()
        if len(raw) > MAX_WEBHOOK_BYTES:
            raise HTTPException(413, "Webhook payload is too large")
        try:
            payload = json.loads(raw)
        except ValueError:
            raise HTTPException(400, "Webhook body must be JSON") from None
        event = str(payload.get("eventType", "")) if isinstance(payload, dict) else ""
        if event.lower() == "test":
            return {"status": "ok", "event": event}
        if event.lower() not in WEBHOOK_EVENTS:
            return {"status": "ignored", "event": event}
        if not request.app.state.store.get_settings()["auto_captions"]:
            return {"status": "disabled", "event": event, "detail": "Turn on auto-captions in Settings to caption imports."}
        # An import means the file at this path is new, so a caption SpeechDock wrote for the old one is stale.
        return {"status": "queued", "event": event, **await queue(request, arr_paths(payload), source="webhook", force=True)}

    @router.get("")
    async def decisions(request: Request, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                        outcome: str | None = None, path: str | None = None):
        """What SpeechDock decided per video (written, matched an existing track, ...), newest first."""
        ledger = request.app.state.ledger
        if path is not None:
            row = ledger.get(path)
            return {"items": [row] if row else [], "total": 1 if row else 0}
        return ledger.listing(limit, offset, outcome)

    app.include_router(router)
