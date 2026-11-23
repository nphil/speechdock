"""Subtitle tidy-up jobs: the queue task, its API, and the alerts for every failed job.

``POST /api/subtitles/tidy`` (one path) and ``/tidy/batch`` queue jobs by hand; the scheduled look in
``watch.py`` queues them for new imports and new Bazarr subtitles. An anime episode with English audio
that has no dub measurement yet gets a dub caption job first and its tidy-up again right behind it.
``POST /api/subtitles/undo/{job_id}`` reverses one job's changes.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from . import plex, tidy
from .captions import media
from .jobs import QueueFull
from .watch import Library, sources

log = logging.getLogger("speechdock.tidy")
TASK_NAMES = {"captions": "Dub caption", "inspect": "Audio inspection", "prune": "Audio pruning", "tidy": "Subtitle tidy-up",
              "transcribe": "Transcription", "translate": "Translation", "detect": "Language detection"}


class TidyBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    anime: bool | None = None


class TidyBatchBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paths: list[str] = Field(min_length=1, max_length=5000)
    anime: bool | None = None


async def anime_for(path: str) -> bool:
    """Whether a video belongs to an anime title, asked from Sonarr/Radarr (False when they cannot say)."""
    import httpx
    found = sources()
    if not ({"sonarr", "radarr"} & found.keys()):
        return False
    try:
        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            folders = await Library(found, client).anime_folders()
    except (httpx.HTTPError, ValueError):
        return False
    roots = media.media_roots()
    for folder in folders:
        for prefix, local in roots.items():
            if folder.startswith(prefix.rstrip("/") + "/"):
                local_folder = local + folder[len(prefix.rstrip("/")):]
                if path.startswith(local_folder + "/"):
                    return True
    return False


async def run_tidy_job(app: FastAPI, job: dict, progress, data_dir: Path) -> dict:
    request = job["request"]
    try:
        video = media.resolve_media_path(request["path"])
    except media.MediaError as exc:
        raise ValueError(str(exc)) from exc
    anime = request.get("anime")
    if anime is None:
        anime = await anime_for(str(video))
    await progress(0.05)
    record = app.state.ledger.get(str(video))
    report = await asyncio.to_thread(tidy.tidy, video, anime=anime, record=record,
                                     quarantine=data_dir / "subtitle-quarantine" / job["id"])
    if {"speechdock", "speechdock-ocr", "speechdock-dub"} & set(report["pending"]) and not request.get("after_captions"):
        # No dub measurement yet, none of the OCR'd text, or no dub caption at all: measure (or write) the dub caption,
        # then tidy this video again behind it. Forced so a reused earlier result cannot skip the new measurement.
        jobs = app.state.jobs
        model = app.state.store.get_settings()["transcription_model"]
        forced = bool({"speechdock-ocr", "speechdock-dub"} & set(report["pending"]))
        await jobs.submit_captions(source="tidy", path=str(video), force=forced, model=model)
        await jobs.submit_tidy(source="tidy", path=str(video), anime=anime, after_captions=True)
        report["notes"].append("Queued a dub caption job and a second tidy-up after it.")
    if report["written"] or report["renamed"] or report["quarantined"]:
        report["plex_refreshed"] = await plex.refresh_folder(str(video.parent))
    await progress(1.0)
    if report["errors"] and not (report["written"] or report["renamed"] or report["quarantined"]):
        raise RuntimeError("; ".join(report["errors"])[:400])
    if report["errors"]:
        await app.state.notifier.problem(f"tidy-partial:{video.parent}", "Subtitle tidy-up incomplete",
                                         f"{video.name}: {'; '.join(report['errors'])[:300]}")
    return {"result_json": report, "duration_seconds": None}


async def job_failed(app: FastAPI, job: dict | None, message: str) -> None:
    """Every failed job is an issue to report; the same kind is repeated at most every few hours."""
    if not job:
        return
    name = TASK_NAMES.get(job["task"], job["task"])
    await app.state.notifier.problem(f"job:{job['task']}", f"{name} failed", f"{job.get('filename') or ''}\n{message}")


def attach_tidy_routes(app: FastAPI) -> None:
    router = APIRouter(prefix="/api/subtitles", tags=["subtitles"])

    async def queue(request: Request, paths: list[str], anime: bool | None, source: str) -> dict:
        queued, existing, rejected = [], [], []
        for value in dict.fromkeys(paths):
            try:
                path = str(await asyncio.to_thread(media.resolve_media_path, value))
                job, created = await request.app.state.jobs.submit_tidy(source=source, path=path, anime=anime)
            except (media.MediaError, QueueFull) as exc:
                rejected.append({"path": value, "error": str(exc)})
                continue
            (queued if created else existing).append({"path": path, "job_id": job["id"]})
        return {"queued": queued, "already_queued": existing, "rejected": rejected}

    @router.post("/tidy", status_code=202)
    async def create(request: Request, body: TidyBody):
        """Queue a tidy-up of one video (anime is looked up in Sonarr/Radarr when not given)."""
        result = await queue(request, [body.path], body.anime, "subtitles")
        if result["rejected"]:
            raise HTTPException(422, result["rejected"][0]["error"])
        return result

    @router.post("/tidy/batch", status_code=202)
    async def batch(request: Request, body: TidyBatchBody):
        return await queue(request, body.paths, body.anime, "subtitles")

    @router.post("/look")
    async def look(request: Request):
        """Run the scheduled look now: queue a tidy-up for everything imported or subtitled since the last one."""
        return await request.app.state.watcher.look()

    @router.get("/status")
    async def status(request: Request):
        return {"auto_tidy": request.app.state.store.get_settings()["auto_tidy"], "sources": sorted(sources()),
                "notifications": request.app.state.notifier.configured, "last_look": request.app.state.watcher.last_result}

    @router.post("/undo/{job_id}")
    async def undo(request: Request, job_id: str):
        job = request.app.state.store.get_job(job_id, internal=True)
        if job is None or job["task"] != "tidy":
            raise HTTPException(404, "No subtitle tidy-up job with that id")
        if job["status"] != "completed" or not job.get("result_json"):
            raise HTTPException(409, "Only a completed tidy-up can be undone")
        entries = job["result_json"].get("journal") or []
        return {"undone": await asyncio.to_thread(tidy.undo, entries), "entries": len(entries)}

    app.include_router(router)
