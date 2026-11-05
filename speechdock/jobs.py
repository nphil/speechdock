"""Durable, bounded serial queue for local speech requests."""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .store import Store, utcnow


class QueueFull(ValueError):
    pass


class UploadTooLarge(ValueError):
    pass


class JobNotFound(ValueError):
    pass


class JobConflict(ValueError):
    pass


MAX_CAPTION_QUEUE = 5000
MAX_INSPECT_QUEUE = 200  # inspections and prunings each
PATH_TASKS = ("captions", "inspect", "prune")  # jobs whose input is a path in a mount, not a spooled upload
MAX_RESTARTS = 3  # a job that keeps being cut short by restarts stops being retried
RESTART_GAVE_UP = f"Processing was interrupted by a restart {MAX_RESTARTS} times and was not retried again"


class JobManager:
    def __init__(self, store: Store, processor, settings_provider=None):
        self.store = store
        self.processor = processor
        self.settings_provider = settings_provider or store.get_settings
        self.upload_dir = store.data_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._changed = asyncio.Condition()
        self._worker = None
        self._active = None
        self._active_id = None
        self._reservations = 0
        self._closing = False

    def _path(self, job_id: str) -> Path:
        # Only canonical UUIDs are accepted, so callers cannot escape uploads/.
        try:
            if str(uuid.UUID(job_id)) != job_id:
                raise ValueError
        except (ValueError, AttributeError):
            raise JobNotFound("Job not found") from None
        return self.upload_dir / job_id

    def _remove_upload(self, job_id: str) -> None:
        path = self._path(job_id)
        path.unlink(missing_ok=True)
        path.with_suffix(".pcm").unlink(missing_ok=True)

    async def _notify(self) -> None:
        async with self._changed:
            self._changed.notify_all()

    async def start(self) -> None:
        if self._worker is not None:
            return
        self._closing = False
        now = utcnow()
        # A job that was running when the service stopped goes back in the queue: its input (the spooled
        # upload, or the caption path) is still there. A cancel that had not finished stays a cancel.
        with self.store.db:
            self.store.db.execute("UPDATE jobs SET status='cancelled', finished_at=? WHERE status='cancelling'", (now,))
            self.store.db.execute("UPDATE jobs SET status='queued', started_at=NULL, progress=NULL, restarts=restarts+1 WHERE status='running' AND restarts<?", (MAX_RESTARTS,))
            self.store.db.execute("UPDATE jobs SET status='failed', finished_at=?, error=? WHERE status='running'", (now, RESTART_GAVE_UP))
        queued = self.store.db.execute("SELECT id, task FROM jobs WHERE status='queued'").fetchall()
        live = {row["id"] for row in queued}
        for row in queued:
            if row["task"] not in PATH_TASKS and not self._path(row["id"]).is_file():
                with self.store.db:
                    self.store.db.execute("UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=?", (now, "The uploaded audio was lost during the restart", row["id"]))
                live.discard(row["id"])
        for path in self.upload_dir.iterdir():
            job_id = path.name[:-4] if path.name.endswith(".pcm") else path.name
            # A .pcm file is scratch from an interrupted run; an upload stays only while its job is waiting.
            if path.is_file() and (path.name.endswith(".pcm") or job_id not in live):
                try:
                    self._path(job_id)
                except JobNotFound:
                    continue
                path.unlink(missing_ok=True)
        await self.cleanup()
        self._worker = asyncio.create_task(self._run())
        self._wake.set()

    def _interrupted(self, job_id: str) -> bool:
        """A shutdown cut this job short: queue it again (True, its upload must stay) unless it was being
        cancelled or has already been cut short too often."""
        now = utcnow()
        with self.store.db:
            self.store.db.execute("UPDATE jobs SET status='cancelled', finished_at=? WHERE id=? AND status='cancelling'", (now, job_id))
            requeued = self.store.db.execute("UPDATE jobs SET status='queued', started_at=NULL, progress=NULL, restarts=restarts+1 WHERE id=? AND status='running' AND restarts<?", (job_id, MAX_RESTARTS)).rowcount
            self.store.db.execute("UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=? AND status='running'", (now, RESTART_GAVE_UP, job_id))
        return bool(requeued)

    async def close(self) -> None:
        self._closing = True
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        if self._active_id is not None:
            if self._active is not None and not self._active.done():
                self._active.cancel()
                await asyncio.gather(self._active, return_exceptions=True)
            if not self._interrupted(self._active_id):
                self._remove_upload(self._active_id)
            self._active = None
            self._active_id = None
        await self._notify()

    def queue_status(self) -> dict:
        rows = dict(self.store.db.execute("SELECT status,COUNT(*) FROM jobs WHERE status IN ('queued','running','cancelling') GROUP BY status"))
        return {"waiting": rows.get("queued", 0), "running": rows.get("running", 0) + rows.get("cancelling", 0)}

    def _waiting_uploads(self) -> int:
        """Queued Bazarr/UI requests; caption batches and import-gate jobs never count against their limit."""
        return self.store.db.execute("SELECT COUNT(*) FROM jobs WHERE status='queued' AND task NOT IN ('captions','inspect','prune')").fetchone()[0]

    async def submit(self, upload, *, source: str, filename: str, task: str, language: str, model: str, encode: bool = True) -> dict:
        async with self._lock:
            if self._closing:
                raise JobConflict("The job queue is shutting down")
            if self._waiting_uploads() + self._reservations >= 8:
                raise QueueFull("The speech queue is full")
            self._reservations += 1
        job_id = str(uuid.uuid4())
        path = self._path(job_id)
        try:
            limit = int(self.settings_provider()["max_upload_mb"]) * 1024 * 1024
            size = 0
            with path.open("xb") as output:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > limit:
                        raise UploadTooLarge("Upload exceeds the configured size limit")
                    output.write(chunk)
            # User supplied filenames are display-only, never file paths.
            filename = re.sub(r"[\x00-\x1f\x7f]", "", str(filename).replace("\\", "/").split("/")[-1])[:255] or "upload"
            with self.store.db:
                self.store.db.execute("INSERT INTO jobs(id,source,filename,task,language,model,encode,status,created_at,progress) VALUES(?,?,?,?,?,?,?,'queued',?,NULL)", (job_id, source, filename, task, language, model, int(encode), utcnow()))
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            async with self._lock:
                self._reservations -= 1
        self._wake.set()
        await self._notify()
        return self.store.get_job(job_id)

    async def submit_captions(self, *, source: str, path: str, force: bool, model: str) -> tuple[dict, bool]:
        """Queue a dub-caption job for one video; returns (job, created).

        A video already queued or running is not queued twice; a queued job that has not started
        is upgraded to a forced one when a forced request arrives.
        """
        async with self._lock:
            if self._closing:
                raise JobConflict("The job queue is shutting down")
            existing = self.store.db.execute(
                "SELECT id,status,payload FROM jobs WHERE task='captions' AND status IN ('queued','running','cancelling') AND json_extract(payload,'$.path')=?", (path,)).fetchone()
            if existing:
                if force and existing["status"] == "queued" and not json.loads(existing["payload"]).get("force"):
                    with self.store.db:
                        self.store.db.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps({"path": path, "force": True}), existing["id"]))
                return self.store.get_job(existing["id"]), False
            waiting = self.store.db.execute("SELECT COUNT(*) FROM jobs WHERE status='queued' AND task='captions'").fetchone()[0]
            if waiting >= MAX_CAPTION_QUEUE:
                raise QueueFull("The caption queue is full")
            job_id = str(uuid.uuid4())
            with self.store.db:
                self.store.db.execute(
                    "INSERT INTO jobs(id,source,filename,task,language,model,encode,status,created_at,progress,payload) VALUES(?,?,?,?,?,?,1,'queued',?,NULL,?)",
                    (job_id, source, Path(path).name[:255], "captions", "en", model, utcnow(), json.dumps({"path": path, "force": force})))
        self._wake.set()
        await self._notify()
        return self.store.get_job(job_id), True

    async def submit_inspection(self, *, source: str, path: str, detect: str | list[str], note: str | None, model: str) -> tuple[dict, bool]:
        """Queue an inspection of one video; returns (job, created). The same path with the same
        ``detect`` already queued or running is not queued twice."""
        return await self._submit_path_job("inspect", source=source, path=path, model=model, request={"path": path, "detect": detect, **({"note": note} if note else {})}, same="detect")

    async def submit_prune(self, *, source: str, path: str, keep: list[str], dry_run: bool, note: str | None) -> tuple[dict, bool]:
        """Queue an audio pruning of one video; returns (job, created). A path already queued or running
        is not queued twice, whatever its keep list."""
        return await self._submit_path_job("prune", source=source, path=path, model="mkvmerge",
                                           request={"path": path, "keep": keep, "dry_run": dry_run, **({"note": note} if note else {})}, same=None)

    async def _submit_path_job(self, task: str, *, source: str, path: str, model: str, request: dict, same: str | None) -> tuple[dict, bool]:
        async with self._lock:
            if self._closing:
                raise JobConflict("The job queue is shutting down")
            query = "SELECT id FROM jobs WHERE task=? AND status IN ('queued','running') AND json_extract(payload,'$.path')=?"
            args: tuple = (task, path)
            if same:
                query += f" AND json_extract(payload,'$.{same}')=json_extract(?,'$.{same}')"
                args += (json.dumps(request),)
            existing = self.store.db.execute(query, args).fetchone()
            if existing:
                return self.store.get_job(existing["id"]), False
            waiting = self.store.db.execute("SELECT COUNT(*) FROM jobs WHERE status='queued' AND task=?", (task,)).fetchone()[0]
            if waiting >= MAX_INSPECT_QUEUE:
                raise QueueFull(f"The {task} queue is full")
            job_id = str(uuid.uuid4())
            with self.store.db:
                self.store.db.execute(
                    "INSERT INTO jobs(id,source,filename,task,language,model,encode,status,created_at,progress,payload) VALUES(?,?,?,?,?,?,1,'queued',?,NULL,?)",
                    (job_id, source, Path(path).name[:255], task, "auto", model, utcnow(), json.dumps(request)))
        self._wake.set()
        await self._notify()
        return self.store.get_job(job_id), True

    async def annotate(self, job_id: str, note: str) -> dict:
        """Attach a caller's note (its decision, for example) to an inspection or pruning job."""
        async with self._lock:
            job = self.store.get_job(job_id, internal=True)
            if job is None:
                raise JobNotFound("Job not found")
            if job["task"] not in ("inspect", "prune"):
                raise JobConflict("Only inspection and pruning jobs take a note")
            with self.store.db:
                self.store.db.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps({**(job["request"] or {}), "note": note}), job_id))
        await self._notify()
        return self.store.get_job(job_id)

    async def wait(self, job_id: str) -> dict:
        async with self._changed:
            while True:
                job = self.store.get_job(job_id)
                if job is None:
                    raise JobNotFound("Job not found")
                if job["status"] in ("completed", "failed", "cancelled"):
                    return job
                if self._closing:
                    raise JobConflict("The job queue is shutting down")
                await self._changed.wait()

    async def cancel(self, job_id: str) -> dict:
        async with self._lock:
            job = self.store.get_job(job_id)
            if job is None:
                raise JobNotFound("Job not found")
            if job["status"] == "queued":
                with self.store.db:
                    self.store.db.execute("UPDATE jobs SET status='cancelled',finished_at=? WHERE id=?", (utcnow(), job_id))
                self._remove_upload(job_id)
            elif job["status"] == "running":
                with self.store.db:
                    self.store.db.execute("UPDATE jobs SET status='cancelling' WHERE id=?", (job_id,))
            elif job["status"] != "cancelling":
                raise JobConflict("Only queued or running jobs can be cancelled")
        await self._notify()
        return self.store.get_job(job_id)

    async def delete(self, job_id: str) -> None:
        async with self._lock:
            job = self.store.get_job(job_id)
            if job is None:
                raise JobNotFound("Job not found")
            if job["status"] not in ("completed", "failed", "cancelled"):
                raise JobConflict("Only finished jobs can be deleted")
            with self.store.db:
                self.store.db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            self._remove_upload(job_id)
        await self._notify()

    async def cleanup(self) -> None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=int(self.settings_provider()["retention_days"]))).isoformat()
        async with self._lock:
            rows = self.store.db.execute("SELECT id FROM jobs WHERE status IN ('completed','failed','cancelled') AND finished_at < ?", (cutoff,)).fetchall()
            with self.store.db:
                self.store.db.executemany("DELETE FROM jobs WHERE id=?", [(row[0],) for row in rows])
            for row in rows:
                self._remove_upload(row[0])
        if rows:
            await self._notify()

    async def _run(self) -> None:
        while True:
            self._wake.clear()
            async with self._lock:
                row = self.store.db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY (task='captions'), created_at, rowid LIMIT 1").fetchone()
                if row:
                    job_id = row[0]
                    with self.store.db:
                        self.store.db.execute("UPDATE jobs SET status='running',started_at=?,progress=0 WHERE id=?", (utcnow(), job_id))
                    job = self.store.get_job(job_id, internal=True)
                    self._active_id = job_id
                    self._active = asyncio.create_task(self.processor(job, self._path(job_id), self._progress(job_id)))
            if not row:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=3600)
                except TimeoutError:
                    await self.cleanup()
                continue
            await self._notify()
            keep_upload = False
            try:
                result = await self._active
                async with self._lock:
                    current = self.store.get_job(job_id)
                    if current and current["status"] == "cancelling":
                        status, error = "cancelled", None
                    else:
                        status, error = "completed", None
                    with self.store.db:
                        self.store.db.execute("UPDATE jobs SET status=?,finished_at=?,progress=?,duration_seconds=COALESCE(?,duration_seconds),error=?,result_text=?,result_json=? WHERE id=?", (status, utcnow(), 1.0 if status == "completed" else current["progress"], result.get("duration_seconds"), error, result.get("result_text") if status == "completed" else None, json.dumps(result["result_json"]) if status == "completed" and result.get("result_json") is not None else None, job_id))
            except asyncio.CancelledError:
                if self._closing:
                    keep_upload = self._interrupted(job_id)
                    raise
                with self.store.db:
                    self.store.db.execute("UPDATE jobs SET status='cancelled',finished_at=? WHERE id=?", (utcnow(), job_id))
            except Exception as exc:
                # Avoid storing URLs or filesystem paths that can contain private data.
                message = re.sub(r"https?://\S+", "[engine]", str(exc))
                message = re.sub(r"(?:/[\w.~-]+){2,}", "[path]", message)
                message = re.sub(r"[\x00-\x1f\x7f]", " ", message)[:500] or "Processing failed"
                with self.store.db:
                    self.store.db.execute("UPDATE jobs SET status=CASE WHEN status='cancelling' THEN 'cancelled' ELSE 'failed' END,finished_at=?,error=CASE WHEN status='cancelling' THEN NULL ELSE ? END WHERE id=?", (utcnow(), message, job_id))
            finally:
                if not keep_upload:
                    self._remove_upload(job_id)
                self._active = None
                self._active_id = None
                await self._notify()
            await self.cleanup()

    def _progress(self, job_id: str):
        async def update(progress: float | None, duration_seconds: float | None = None) -> None:
            job = self.store.get_job(job_id)
            if job and job["status"] == "cancelling":
                raise asyncio.CancelledError
            if progress is None:
                return  # only a check that the job was not cancelled, e.g. while it waits for the engine
            with self.store.db:
                self.store.db.execute("UPDATE jobs SET progress=?,duration_seconds=COALESCE(?,duration_seconds) WHERE id=? AND status='running'", (max(0.0, min(1.0, float(progress))), duration_seconds, job_id))
            await self._notify()
        return update
