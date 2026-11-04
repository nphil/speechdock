import asyncio
import io
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from speechdock.jobs import MAX_RESTARTS, RESTART_GAVE_UP, JobConflict, JobManager, QueueFull, UploadTooLarge
from speechdock.store import Store


class Upload:
    def __init__(self, data=b"audio"):
        self.file = io.BytesIO(data)

    async def read(self, size):
        return self.file.read(size)


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name))
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = []

        async def processor(job, path, progress):
            self.calls.append((job["id"], job["encode"], path.read_bytes()))
            self.started.set()
            await progress(0.5, 3)
            await self.release.wait()
            await progress(1.0, 3)
            return {"result_text": "subtitle", "duration_seconds": 3}

        self.manager = JobManager(self.store, processor)
        await self.manager.start()

    async def asyncTearDown(self):
        await self.manager.close()
        self.store.close()
        self.temp.cleanup()

    async def submit(self, data=b"audio", **kwargs):
        return await self.manager.submit(Upload(data), source="api", filename="../../clip.wav", task="transcribe", language="auto", model="whisper-1", **kwargs)

    async def test_serial_bounded_queue_and_cancellation(self):
        first = await self.submit(encode=False)
        await self.started.wait()
        waiting = [await self.submit() for _ in range(8)]
        self.assertEqual(self.manager.queue_status(), {"waiting": 8, "running": 1})
        with self.assertRaises(QueueFull):
            await self.submit()
        self.assertEqual((await self.manager.cancel(waiting[0]["id"]))["status"], "cancelled")
        self.assertFalse((self.manager.upload_dir / waiting[0]["id"]).exists())
        with self.assertRaises(JobConflict):
            await self.manager.delete(first["id"])
        self.assertEqual((await self.manager.cancel(first["id"]))["status"], "cancelling")
        self.assertEqual(self.store.get_job(first["id"])["status"], "cancelling")
        self.release.set()
        self.assertEqual((await self.manager.wait(first["id"]))["status"], "cancelled")
        self.assertEqual((await self.manager.wait(waiting[-1]["id"]))["status"], "completed")
        self.assertEqual(len(self.calls), 8)
        self.assertFalse(self.calls[0][1])
        self.assertEqual(first["filename"], "clip.wav")
        self.assertNotIn("result_text", self.store.list_jobs()["items"][0])
        await self.manager.delete(first["id"])
        self.assertIsNone(self.store.get_job(first["id"]))

    async def test_upload_limit_removes_partial_file(self):
        self.store.update_settings({"max_upload_mb": 1})
        with self.assertRaises(UploadTooLarge):
            await self.submit(b"x" * (1024 * 1024 + 1))
        self.assertEqual(list(self.manager.upload_dir.iterdir()), [])
        self.assertEqual(self.store.list_jobs()["total"], 0)

    async def test_restart_preserves_queued_and_settings(self):
        first = await self.submit()
        await self.started.wait()
        queued = await self.submit(encode=False)
        self.store.update_settings({"retention_days": 3})
        await self.manager.close()
        # The job the shutdown cut short goes back in the queue with its audio instead of failing.
        cut_short = self.store.get_job(first["id"])
        self.assertEqual((cut_short["status"], cut_short["restarts"]), ("queued", 1))
        self.assertTrue((self.manager.upload_dir / first["id"]).exists())
        self.assertEqual(self.store.get_job(queued["id"])["status"], "queued")
        self.store.close()
        self.store = Store(Path(self.temp.name))
        self.manager = JobManager(self.store, self.manager.processor)
        self.release.set()
        await self.manager.start()
        self.assertEqual((await self.manager.wait(first["id"]))["result_text"], "subtitle")
        result = await self.manager.wait(queued["id"])
        self.assertEqual(result["result_text"], "subtitle")
        self.assertEqual([call[0] for call in self.calls], [first["id"], first["id"], queued["id"]])
        self.assertEqual(self.calls[-1][1:], (False, b"audio"))
        self.assertEqual(self.store.get_settings()["retention_days"], 3)

    def plant(self, job_id, status, *, restarts=0, task="transcribe", payload=None, audio=True):
        """A job row (and its spooled audio) as a killed service would have left it."""
        with self.store.db:
            self.store.db.execute(
                "INSERT INTO jobs(id,source,filename,task,language,model,encode,status,created_at,started_at,progress,restarts,payload) VALUES(?,?,?,?,?,?,1,?,?,?,0.4,?,?)",
                (job_id, "bazarr", "ep.mkv", task, "en", "whisper-1", status, "2026-11-04T00:00:00+00:00", "2026-11-04T00:00:01+00:00", restarts, json.dumps(payload) if payload else None))
        if audio:
            (self.manager.upload_dir / job_id).write_bytes(b"audio")

    async def test_a_crash_leaves_work_that_is_queued_again_but_never_forever(self):
        await self.manager.close()
        ids = {name: f"00000000-0000-4000-8000-00000000000{number}" for number, name in enumerate(
            ("running", "tired", "cancelling", "lost", "caption", "stray"), 1)}
        self.plant(ids["running"], "running")
        self.plant(ids["tired"], "running", restarts=MAX_RESTARTS)
        self.plant(ids["cancelling"], "cancelling")
        self.plant(ids["lost"], "running", audio=False)
        self.plant(ids["caption"], "running", task="captions", payload={"path": "/plex-media/tv/Show/Ep 1.mkv", "force": False}, audio=False)
        self.plant(ids["stray"], "completed")  # an upload nothing waits for any more
        (self.manager.upload_dir / f"{ids['running']}.pcm").write_bytes(b"half-decoded scratch")
        gate = asyncio.Event()

        async def hold(job, path, progress):
            await gate.wait()
            return {"result_text": "subtitle", "duration_seconds": 1}

        self.manager = JobManager(self.store, hold)
        await self.manager.start()
        status = {name: self.store.get_job(job_id) for name, job_id in ids.items()}
        self.assertEqual(status["running"]["restarts"], 1)
        self.assertIn(status["running"]["status"], ("queued", "running"))
        self.assertEqual((status["tired"]["status"], status["tired"]["error"]), ("failed", RESTART_GAVE_UP))
        self.assertEqual((status["cancelling"]["status"], status["cancelling"]["error"]), ("cancelled", None))
        self.assertEqual(status["lost"]["status"], "failed")
        self.assertIn("audio was lost", status["lost"]["error"])
        self.assertIn(status["caption"]["status"], ("queued", "running"))  # its input is just a path
        self.assertEqual(status["caption"]["captions"]["path"], "/plex-media/tv/Show/Ep 1.mkv")
        self.assertEqual(sorted(path.name for path in self.manager.upload_dir.iterdir()), [ids["running"]])
        gate.set()
        for name in ("running", "caption"):
            self.assertEqual((await self.manager.wait(ids[name]))["status"], "completed")

    async def test_a_job_cancelled_during_shutdown_is_not_run_again(self):
        job = await self.submit()
        await self.started.wait()
        await self.manager.cancel(job["id"])
        await self.manager.close()
        self.assertEqual(self.store.get_job(job["id"])["status"], "cancelled")
        self.assertFalse((self.manager.upload_dir / job["id"]).exists())

    async def test_waiting_for_the_engine_checks_for_cancellation_without_touching_progress(self):
        seen = []

        async def waiting(job, path, progress):
            await progress(0.5, 3)
            await progress(None)  # what a job does each time it waits for the engine
            seen.append(self.store.get_job(job["id"])["progress"])
            self.started.set()
            await self.release.wait()
            await progress(None)
            seen.append("not cancelled")
            return {"result_text": "subtitle", "duration_seconds": 3}

        self.manager.processor = waiting
        job = await self.submit()
        await self.started.wait()
        self.assertEqual(seen, [0.5])
        await self.manager.cancel(job["id"])
        self.release.set()
        self.assertEqual((await self.manager.wait(job["id"]))["status"], "cancelled")
        self.assertEqual(seen, [0.5])

    async def test_retention_and_failure(self):
        async def broken(job, path, progress):
            raise ValueError("bad https://user:secret@example.com/file /private/folder")
        self.manager.processor = broken
        job = await self.submit()
        result = await self.manager.wait(job["id"])
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("secret", result["error"])
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        with self.store.db:
            self.store.db.execute("UPDATE jobs SET finished_at=? WHERE id=?", (old, job["id"]))
        await self.manager.cleanup()
        self.assertIsNone(self.store.get_job(job["id"]))

    async def test_waiter_cancellation_does_not_cancel_job(self):
        job = await self.submit()
        await self.started.wait()
        waiter = asyncio.create_task(self.manager.wait(job["id"]))
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        self.release.set()
        self.assertEqual((await self.manager.wait(job["id"]))["status"], "completed")


class MigrationTests(unittest.TestCase):
    def test_a_job_history_from_before_restart_counting_gains_the_column(self):
        with tempfile.TemporaryDirectory() as folder:
            old = sqlite3.connect(Path(folder) / "speechdock.sqlite3")
            old.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, source TEXT NOT NULL, filename TEXT NOT NULL, task TEXT NOT NULL, language TEXT NOT NULL, "
                        "model TEXT NOT NULL, encode INTEGER NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, "
                        "progress REAL, duration_seconds REAL, error TEXT, result_text TEXT, result_json TEXT, payload TEXT)")
            old.execute("INSERT INTO jobs(id,source,filename,task,language,model,encode,status,created_at) VALUES('a','bazarr','x.mkv','transcribe','en','m',1,'failed','2026-11-01T00:00:00+00:00')")
            old.commit()
            old.close()
            store = Store(Path(folder))
            self.assertEqual(store.get_job("a")["restarts"], 0)
            store.close()
