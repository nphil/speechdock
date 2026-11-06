"""HTTP interface for SpeechDock and Bazarr."""
from __future__ import annotations

import asyncio
import io
import os
import re
import shutil
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import __version__
from .audio import SpeechDetector, audio_wav_chunks, iter_wav_chunks, normalize_audio, parse_srt, render, speech_wav_chunks
from .engine import ENGINE_WAIT_SECONDS, Engine, EngineUnavailable, EngineWait, make_wait
from .jobs import JobConflict, JobManager, JobNotFound, QueueFull, UploadTooLarge
from .languages import normalize_language
from .models import ModelError, ModelManager
from .recovery import Fallback, transcribe_window
from .store import Store
from .alignment import alignment_lifespan, attach_alignment_routes
from .captions import Ledger, attach_caption_routes, run_caption_job
from .inspect import InspectError, clean_source, parse_detect, resolve_inspect_path, run_inspection
from .prune import PruneError, check_writable_root, parse_keep, run_prune

DATA_DIR = Path(os.environ.get("SPDOCK_DATA_DIR", "/data"))
FRONTEND_DIR = Path(os.environ.get("SPDOCK_FRONTEND_DIR", "ui/dist"))


def clean_filename(name: str | None) -> str:
    name = (name or "upload").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip(" .")
    return name[:200] or "upload"


def clean_error(exc: Exception) -> str:
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(exc))[:500] or "Processing failed"


def normalize_url(value: str) -> str:
    try:
        parts = urlsplit(value.strip())
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None or parts.password is not None or parts.query or parts.fragment:
            raise ValueError
        _ = parts.port
        path = parts.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))
    except ValueError:
        raise ValueError("Engine URL must be an HTTP(S) URL without credentials, query, or fragment") from None


class SettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    engine_url: str | None = None
    transcription_model: str | None = None
    detection_model: str | None = None
    translation_model: str | None = None
    inference_timeout_seconds: int | None = Field(default=None, ge=1, le=86400, strict=True)
    connect_timeout_seconds: int | None = Field(default=None, ge=1, le=300, strict=True)
    max_upload_mb: int | None = Field(default=None, ge=1, le=10240, strict=True)
    retention_days: int | None = Field(default=None, ge=1, le=3650, strict=True)
    default_language: str | None = None
    auto_captions: bool | None = Field(default=None, strict=True)

    @field_validator("engine_url")
    @classmethod
    def valid_url(cls, value: str | None) -> str | None:
        return normalize_url(value) if value is not None else None

    @field_validator("default_language")
    @classmethod
    def valid_language(cls, value: str | None) -> str | None:
        return normalize_language(value) if value is not None else None


class RegisterBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirm_reload: bool = Field(default=False, strict=True)
    keep_loaded: bool | None = Field(default=None, strict=True)

    @field_validator("keep_loaded")
    @classmethod
    def valid_keep_loaded(cls, value: bool | None) -> bool:
        if value is None:
            raise ValueError("keep_loaded must be a boolean")
        return value


class ImportBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    source: str
    url: str | None = None
    expected_sha256: str | None = None
    expected_size_bytes: int | None = Field(default=None, strict=True)
    source_filename: str | None = None


async def model_for(app: FastAPI, model: str, task: str, *, strict: bool = False) -> dict:
    """The registered model for a task.

    An engine that cannot be reached cannot say whether a model is registered. With ``strict`` that
    raises ``EngineUnavailable`` (a job waits and asks again); otherwise the model is accepted on what
    SpeechDock knows itself, so a request is queued while the engine restarts instead of being refused.
    A model the engine does list as missing, or one SpeechDock does not know, is refused at once.
    """
    items = await app.state.models.list_models()
    item = next((item for item in items if item["model_id"] == model), None)
    if item is None:
        raise ValueError(f"Model '{model}' is not available. Install and register it in Models first.")
    if not item["registered"]:
        if item.get("engine_reachable", True):
            raise ValueError(f"Model '{model}' is not available. Install and register it in Models first.")
        if strict:
            raise EngineUnavailable("The speech engine cannot be reached, so the model cannot be confirmed yet")
    if task == "translate" and not item["translation_capable"]:
        raise ValueError(f"Model '{model}' cannot translate. Select a translation-capable model.")
    if task == "detect" and not item.get("language_detection_capable", False):
        raise ValueError(f"Model '{model}' cannot detect language. Select a detection-capable model.")
    return item


async def ready_model(app: FastAPI, model: str, task: str, wait: EngineWait) -> dict:
    """``model_for`` for a job about to run: while the engine is down, wait for it (bounded by ``wait``)
    instead of failing the job; a model that is really not registered still fails at once."""
    while True:
        try:
            item = await model_for(app, model, task, strict=True)
        except EngineUnavailable:
            await wait.pause("engine not reachable")
            continue
        wait.recovered()
        return item


FALLBACK_WAIT_SECONDS = 90  # a fallback model that will not start is dropped quickly, unlike the engine itself
INSPECT_WAIT_SECONDS = 720  # longer than the 600 s idle timeout of llama-swap's big chat models; callers allow 30 minutes


async def fallback_for(app: FastAPI, settings: dict, item: dict, task: str, waiting) -> Fallback | None:
    """The model that gets a window the first model's timings could not be recovered for: the translation
    model, which is a full Whisper (a Whisper primary has no second model, and translation is Whisper's job)."""
    if task != "transcribe" or item["runtime"] == "whisper.cpp":
        return None
    try:
        other = await model_for(app, settings["translation_model"], "translate")
    except ValueError:
        return None
    if other["runtime"] != "whisper.cpp":
        return None
    limits = {**settings, "engine_wait_seconds": min(float(settings.get("engine_wait_seconds", ENGINE_WAIT_SECONDS)), FALLBACK_WAIT_SECONDS)}
    return Fallback(Engine(limits, runtime=other["runtime"], architecture=other["architecture"], on_wait=waiting), settings["translation_model"])


def make_processor(app: FastAPI, store: Store):
    """The job processor: wait for the engine, then transcribe, translate, detect or caption."""

    async def process(job: dict, path: Path, progress):
        settings = store.get_settings()

        async def waiting(reason: str, waited: float) -> None:
            await progress(None)  # a job cancelled while it waits for the engine stops waiting

        if job["task"] == "prune":
            return await run_prune(job=job, progress=progress)  # mkvmerge and ffmpeg only, no speech engine
        if job["task"] == "inspect":
            gate_settings = {**settings, "engine_wait_seconds": min(float(settings.get("engine_wait_seconds", ENGINE_WAIT_SECONDS)), INSPECT_WAIT_SECONDS)}

            async def engine_for_detection() -> Engine:
                item = await ready_model(app, job["model"], "detect", make_wait(gate_settings, waiting))
                return Engine(gate_settings, runtime=item["runtime"], architecture=item["architecture"], on_wait=waiting)
            return await run_inspection(job=job, progress=progress, engine_for_detection=engine_for_detection, model=job["model"],
                                        vad_model=Path(os.environ.get("SPDOCK_VAD_MODEL", "/opt/speechdock/models/silero-vad-v5.1.onnx")))
        item = await ready_model(app, job["model"], "transcribe" if job["task"] == "captions" else job["task"], make_wait(settings, waiting))
        if job["task"] == "captions":
            return await run_caption_job(
                job=job, progress=progress, ledger=app.state.ledger, workdir=DATA_DIR / "captions-work",
                engine=Engine(settings, runtime=item["runtime"], architecture=item["architecture"], on_wait=waiting))
        if item.get("requires_language") and job["language"] == "auto":
            raise ValueError("This model requires an explicit source language.")
        engine = Engine(settings, runtime=item["runtime"], architecture=item["architecture"], on_wait=waiting)
        pcm = path.with_suffix(".pcm")
        try:
            duration = await normalize_audio(path, pcm, job.get("encode", True))
            if duration > 6 * 3600:
                raise ValueError("Audio exceeds the six-hour limit")
            await progress(0, duration)
            cues, recovered = [], []
            fallback_models: list[Fallback | None] = []

            async def fallback() -> Fallback | None:
                if not fallback_models:
                    fallback_models.append(await fallback_for(app, settings, item, job["task"], waiting))
                return fallback_models[0]

            detector = None
            if job["task"] == "detect":
                detector = await asyncio.to_thread(
                    SpeechDetector, Path(os.environ.get("SPDOCK_VAD_MODEL", "/opt/speechdock/models/silero-vad-v5.1.onnx"))
                )
            for chunk_offset, wav in iter_wav_chunks(pcm):
                if detector is None:
                    windows = await asyncio.to_thread(audio_wav_chunks, wav)
                else:
                    windows = await asyncio.to_thread(speech_wav_chunks, wav, detector)
                await progress(min(1, chunk_offset / duration) if duration else 1, duration)
                for relative_offset, speech_wav in windows:
                    offset = chunk_offset + relative_offset
                    if job["task"] == "detect":
                        result = await engine.detect(speech_wav, job["model"])
                        await progress(1, duration)
                        return {"result_json": result, "duration_seconds": duration}
                    text = await transcribe_window(engine, speech_wav, job["model"], job["language"], job["task"],
                                                   fallback=fallback, at=offset, recovered=recovered)
                    next_cues = parse_srt(text, offset)
                    with wave.open(io.BytesIO(speech_wav), "rb") as source:
                        window_ms = source.getnframes() * 1000 // source.getframerate()
                    if next_cues and next_cues[-1].end_ms > round(offset * 1000) + window_ms:
                        raise ValueError("Subtitle extends beyond its audio window")
                    cues.extend(next_cues)
                    await progress(min(1, offset / duration) if duration else 1, duration)
                await progress(min(1, (chunk_offset + 300) / duration) if duration else 1, duration)
            if job["task"] == "detect":
                raise ValueError("Cannot detect language: no speech found in the recording")
            result = {"result_text": render(cues, "srt"), "duration_seconds": duration}
            if recovered:
                result["result_json"] = {"recovery": recovered}
            return result
        finally:
            pcm.unlink(missing_ok=True)

    return process


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    store = Store(DATA_DIR)
    url = store.get_settings()["engine_url"]
    normalized = normalize_url(url)
    if normalized != url:
        store.update_settings({"engine_url": normalized})
    models = ModelManager(DATA_DIR, store.get_settings)
    app.state.store, app.state.models = store, models
    app.state.ledger = Ledger(store.db)
    shutil.rmtree(DATA_DIR / "captions-work", ignore_errors=True)

    jobs = JobManager(store, make_processor(app, store), store.get_settings)
    app.state.jobs = jobs
    await jobs.start()
    try:
        async with alignment_lifespan(app):
            yield
    finally:
        await jobs.close()
        await models.close()
        store.close()


app = FastAPI(title="SpeechDock", version=__version__, lifespan=lifespan)
attach_alignment_routes(app)
attach_caption_routes(app, model_for)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    if request.url.path == "/api/models/imports":
        return JSONResponse({"detail": "Invalid import request. Check the supplied fields."}, status_code=422)
    return await request_validation_exception_handler(request, exc)


@app.middleware("http")
async def same_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin and request.method not in {"GET", "HEAD", "OPTIONS"}:
        try:
            parsed = urlsplit(origin)
            request_url = urlsplit(str(request.base_url))
            if (parsed.scheme, parsed.netloc.lower()) != (request_url.scheme, request_url.netloc.lower()) or parsed.path or parsed.query or parsed.fragment:
                return JSONResponse({"detail": "Cross-origin request denied"}, status_code=403)
        except ValueError:
            return JSONResponse({"detail": "Invalid origin"}, status_code=403)
    return await call_next(request)


@app.exception_handler(ModelError)
async def model_error(request: Request, exc: ModelError):
    return JSONResponse({"detail": clean_error(exc)}, status_code=exc.status_code)


@app.exception_handler(JobNotFound)
async def not_found(request: Request, exc: JobNotFound):
    return JSONResponse({"detail": clean_error(exc)}, status_code=404)


@app.exception_handler(JobConflict)
async def conflict(request: Request, exc: JobConflict):
    return JSONResponse({"detail": clean_error(exc)}, status_code=409)


@app.exception_handler(QueueFull)
async def queue_full(request: Request, exc: QueueFull):
    return JSONResponse({"detail": clean_error(exc)}, status_code=503)


@app.exception_handler(UploadTooLarge)
async def too_large(request: Request, exc: UploadTooLarge):
    return JSONResponse({"detail": clean_error(exc)}, status_code=413)


@app.get("/health")
async def health():
    return {"status": "ok", "version": __version__}


@app.get("/status")
async def status(request: Request):
    result = await Engine(request.app.state.store.get_settings()).probe()
    if not result["reachable"]:
        raise HTTPException(503, "Speech engine is unavailable")
    return {"status": "ok"}


@app.post("/api/engine/test")
async def engine_test(request: Request):
    return await Engine(request.app.state.store.get_settings()).probe()


@app.get("/api/overview")
async def overview(request: Request):
    store = request.app.state.store
    settings = store.get_settings()
    engine = await Engine(settings).probe()
    warnings = []
    try:
        await model_for(request.app, settings["translation_model"], "translate")
    except (ValueError, ModelError) as exc:
        warnings.append(clean_error(exc))
    warnings.append("Loading a SpeechDock model may unload other models in llama-swap.")
    return {"version": __version__, "engine": {"url": settings["engine_url"], **engine}, "queue": request.app.state.jobs.queue_status(), "counts": store.counts(), "storage": {"free_bytes": shutil.disk_usage(DATA_DIR).free}, "warnings": warnings}


@app.get("/api/settings")
async def get_settings(request: Request):
    return request.app.state.store.get_settings()


@app.patch("/api/settings")
async def patch_settings(request: Request, patch: SettingsPatch):
    changes = patch.model_dump(exclude_unset=True)
    if any(value is None for value in changes.values()):
        raise HTTPException(422, "Settings cannot be null")
    for key, task in (("transcription_model", "transcribe"), ("translation_model", "translate"), ("detection_model", "detect")):
        if key in changes:
            try:
                await model_for(request.app, changes[key], task, strict=True)
            except ValueError as exc:
                raise HTTPException(422, clean_error(exc)) from exc
    return request.app.state.store.update_settings(changes)


@app.get("/api/models")
async def get_models(request: Request):
    manager = request.app.state.models
    return {"items": await manager.list_models(), "downloads": await manager.downloads(), "runtimes": manager.runtimes}


@app.post("/api/models/imports", status_code=202)
async def create_model_import(request: Request, body: ImportBody):
    try:
        return await request.app.state.models.create_import(**body.model_dump())
    except ModelError as exc:
        if body.url:
            raise HTTPException(exc.status_code, "Could not start import. Check the source and import settings.") from None
        raise


@app.put("/api/models/{catalog_id}/upload")
async def upload_model(request: Request, catalog_id: str):
    length = request.headers.get("content-length")
    if length is not None:
        try:
            length = int(length)
            if length < 0:
                raise ValueError
        except ValueError:
            raise HTTPException(400, "Invalid Content-Length") from None
    return await request.app.state.models.receive_upload(catalog_id, request.stream(), content_length=length)


@app.delete("/api/models/{catalog_id}/import")
async def remove_model_import(request: Request, catalog_id: str, confirm: bool = False):
    return await request.app.state.models.remove_import(catalog_id, confirm=confirm)


@app.post("/api/models/{catalog_id}/download")
async def install_model(request: Request, catalog_id: str):
    return await request.app.state.models.install(catalog_id)


@app.post("/api/models/{catalog_id}/cancel-download")
async def cancel_download(request: Request, catalog_id: str):
    return await request.app.state.models.cancel_download(catalog_id)


@app.post("/api/models/{catalog_id}/register")
async def register_model(request: Request, catalog_id: str, body: RegisterBody):
    return await request.app.state.models.register(catalog_id, body.confirm_reload, body.keep_loaded)


@app.delete("/api/models/{catalog_id}/registration")
async def unregister_model(request: Request, catalog_id: str, confirm_reload: bool = False):
    return await request.app.state.models.unregister(catalog_id, confirm_reload)


@app.delete("/api/models/{catalog_id}/files")
async def delete_model(request: Request, catalog_id: str, confirm: bool = False):
    return await request.app.state.models.delete(catalog_id, confirm)


@app.get("/api/jobs")
async def list_jobs(request: Request, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), status: str | None = None):
    if status is not None and status not in {"queued", "running", "cancelling", "completed", "failed", "cancelled"}:
        raise HTTPException(422, "Unknown job status")
    return request.app.state.store.list_jobs(limit, offset, status)


async def submit(request: Request, upload: UploadFile, source: str, task: str, language: str | None, model: str | None, encode: bool, filename: str | None = None):
    if task not in {"transcribe", "translate", "detect"}:
        raise HTTPException(422, "Unknown task")
    settings = request.app.state.store.get_settings()
    try:
        language = normalize_language(language or settings["default_language"])
        model = model or settings[{"translate": "translation_model", "detect": "detection_model", "transcribe": "transcription_model"}[task]]
        item = await model_for(request.app, model, task)
        if item.get("requires_language") and language == "auto":
            raise ValueError("This model requires an explicit source language.")
        return await request.app.state.jobs.submit(upload, source=source, filename=clean_filename(filename or upload.filename), task=task, language=language, model=model, encode=encode)
    except ValueError as exc:
        if isinstance(exc, (JobConflict, JobNotFound, QueueFull, UploadTooLarge, ModelError)):
            raise
        raise HTTPException(422, clean_error(exc)) from exc
    finally:
        await upload.close()


@app.post("/api/jobs", status_code=202)
async def create_job(request: Request, file: UploadFile = File(...), task: str = Form("transcribe"), language: str | None = Form(None), model: str | None = Form(None)):
    return await submit(request, file, "upload", task, language, model, True)


def get_job_or_404(request: Request, job_id: str):
    job = request.app.state.store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/jobs/{job_id}")
async def job_detail(request: Request, job_id: str):
    return get_job_or_404(request, job_id)


@app.get("/api/jobs/{job_id}/result")
async def job_result(request: Request, job_id: str, format: str = "srt"):
    if format not in {"srt", "vtt", "txt"}:
        raise HTTPException(422, "Unknown result format")
    job = get_job_or_404(request, job_id)
    if not job["result_available"] or job["task"] == "detect":
        raise HTTPException(409, "Subtitle result is not available")
    content = render(parse_srt(job["result_text"]), format)
    return Response(content, media_type={"srt": "application/x-subrip", "vtt": "text/vtt", "txt": "text/plain"}[format], headers={"Content-Disposition": f'attachment; filename="speechdock-{job_id}.{format}"'})


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str):
    return await request.app.state.jobs.cancel(job_id)


@app.delete("/api/jobs/{job_id}")
async def delete_job(request: Request, job_id: str, confirm: bool = False):
    if not confirm:
        raise HTTPException(400, "Confirm deletion with confirm=true")
    await request.app.state.jobs.delete(job_id)
    return {"deleted": True}


async def sync_job(request: Request, audio_file: UploadFile, task: str, language: str | None, encode: bool, video_file: str | None):
    job = await submit(request, audio_file, "bazarr", task, language, None, encode, video_file)
    result = await request.app.state.jobs.wait(job["id"])
    if result["status"] != "completed":
        raise HTTPException(502, result.get("error") or "Speech processing did not complete")
    return request.app.state.store.get_job(job["id"], internal=True)


class InspectBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    detect: str | list[str] = "untagged"
    source: str = "api"
    note: str | None = Field(default=None, max_length=200)


class NoteBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note: str = Field(min_length=1, max_length=200)


@app.post("/api/inspect")
async def inspect(request: Request, body: InspectBody):
    """Inspect one video from the inspectable mounts and wait for the report: every stream with its
    tags, and for the audio tracks ``detect`` names (``untagged`` by default, ``all``, ``none`` or a
    list of language codes to verify) the language Whisper hears on a sample from the middle of the
    track. The job shows in Activity; the same path is not inspected twice at once."""
    settings = request.app.state.store.get_settings()
    try:
        source = clean_source(body.source)
        detect = parse_detect(body.detect)
        path = str(await asyncio.to_thread(resolve_inspect_path, body.path))
        if detect != "none":
            await model_for(request.app, settings["detection_model"], "detect")
    except (InspectError, ValueError) as exc:
        raise HTTPException(422, clean_error(exc)) from exc
    job, _ = await request.app.state.jobs.submit_inspection(source=source, path=path, detect=detect, note=body.note, model=settings["detection_model"])
    result = await request.app.state.jobs.wait(job["id"])
    if result["status"] != "completed":
        raise HTTPException(502, result.get("error") or "Inspection did not complete")
    return {"job_id": job["id"], **request.app.state.store.get_job(job["id"], internal=True)["result_json"]}


@app.patch("/api/inspect/{job_id}")
async def annotate_inspection(request: Request, job_id: str, body: NoteBody):
    """Record what the caller decided from an inspection or pruning; shown with the job in Activity."""
    return await request.app.state.jobs.annotate(job_id, re.sub(r"[\x00-\x1f\x7f]", " ", body.note).strip())


class PruneBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    keep: list[str] = Field(min_length=1, max_length=20)
    source: str = "api"
    note: str | None = Field(default=None, max_length=200)
    dry_run: bool = Field(default=False, strict=True)


@app.post("/api/prune")
async def prune(request: Request, body: PruneBody):
    """Remove the audio tracks of a Matroska file whose language is not in ``keep`` (codes or names, in
    order of preference) and wait for the report. Stream copy only, verified against the original, then
    renamed over it; untagged tracks stay. ``dry_run`` reports what would go. Allowed only inside the
    folders ``SPDOCK_PRUNE_ROOTS`` names (the media mounts by default)."""
    try:
        source = clean_source(body.source)
        keep = parse_keep(body.keep)
        path = Path(await asyncio.to_thread(resolve_inspect_path, body.path))
        check_writable_root(path)
    except (InspectError, PruneError, ValueError) as exc:
        raise HTTPException(422, clean_error(exc)) from exc
    job, _ = await request.app.state.jobs.submit_prune(source=source, path=str(path), keep=keep, dry_run=body.dry_run, note=body.note)
    result = await request.app.state.jobs.wait(job["id"])
    if result["status"] != "completed":
        raise HTTPException(502, result.get("error") or "Pruning did not complete")
    return {"job_id": job["id"], **request.app.state.store.get_job(job["id"], internal=True)["result_json"]}


@app.post("/detect-language")
async def detect_language(request: Request, audio_file: UploadFile = File(...), encode: bool = True, video_file: str | None = None):
    result = await sync_job(request, audio_file, "detect", "auto", encode, video_file)
    return result["result_json"]


@app.post("/asr")
async def asr(request: Request, audio_file: UploadFile = File(...), task: str = "transcribe", language: str | None = None, output: str = "txt", encode: bool = True, video_file: str | None = None):
    if task not in {"transcribe", "translate"} or output not in {"srt", "vtt", "txt", "json"}:
        raise HTTPException(422, "Invalid task or output format")
    result = await sync_job(request, audio_file, task, language, encode, video_file)
    cues = parse_srt(result["result_text"])
    if output == "json":
        return {"text": render(cues, "txt"), "segments": [{"start": c.start_ms / 1000, "end": c.end_ms / 1000, "text": c.text} for c in cues]}
    return Response(render(cues, output), media_type={"srt": "application/x-subrip", "vtt": "text/vtt", "txt": "text/plain"}[output])


@app.get("/{path:path}", include_in_schema=False)
async def frontend(path: str):
    root = FRONTEND_DIR.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise HTTPException(404)
    if target.is_file():
        return FileResponse(target)
    if not path or ("." not in Path(path).name and not path.startswith(("api/", "health", "status"))):
        index = root / "index.html"
        if index.is_file():
            return FileResponse(index)
    raise HTTPException(404, "Not found")
