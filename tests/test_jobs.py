import asyncio
import io
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from speechdock.jobs import JobConflict, JobManager, QueueFull, UploadTooLarge
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
        self.assertEqual(self.store.get_job(first["id"])["status"], "failed")
        self.assertEqual(self.store.get_job(queued["id"])["status"], "queued")
        self.store.close()
        self.store = Store(Path(self.temp.name))
        self.manager = JobManager(self.store, self.manager.processor)
        self.release.set()
        await self.manager.start()
        result = await self.manager.wait(queued["id"])
        self.assertEqual(result["result_text"], "subtitle")
        self.assertEqual(self.calls[-1][1:], (False, b"audio"))
        self.assertEqual(self.store.get_settings()["retention_days"], 3)

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
