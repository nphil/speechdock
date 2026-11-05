"""SQLite-backed settings and speech job history."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


DEFAULT_SETTINGS = {
    "engine_url": os.environ.get("SPDOCK_ENGINE_URL", "https://llama.nateshome.net").rstrip("/"),
    "transcription_model": "whisper-1",
    "detection_model": "whisper-1",
    "translation_model": "speechdock-large-v3",
    "inference_timeout_seconds": 3600,
    "connect_timeout_seconds": 10,
    "max_upload_mb": 1024,
    "retention_days": 14,
    "default_language": "auto",
    "auto_captions": False,
}


class Store:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.data_dir / "speechdock.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, source TEXT NOT NULL, filename TEXT NOT NULL,
                task TEXT NOT NULL, language TEXT NOT NULL, model TEXT NOT NULL,
                encode INTEGER NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
                started_at TEXT, finished_at TEXT, progress REAL, duration_seconds REAL,
                error TEXT, result_text TEXT, result_json TEXT, payload TEXT, restarts INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at);
        """)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(jobs)")}
        if "payload" not in columns:
            self.db.execute("ALTER TABLE jobs ADD COLUMN payload TEXT")
        if "restarts" not in columns:
            self.db.execute("ALTER TABLE jobs ADD COLUMN restarts INTEGER NOT NULL DEFAULT 0")
        self.db.execute(
            "INSERT OR IGNORE INTO settings(key,value) VALUES ('detection_model', COALESCE((SELECT value FROM settings WHERE key='transcription_model'), ?))",
            (json.dumps(DEFAULT_SETTINGS["detection_model"]),),
        )
        self.db.commit()

    def get_settings(self) -> dict:
        settings = dict(DEFAULT_SETTINGS)
        settings.update({row["key"]: json.loads(row["value"]) for row in self.db.execute("SELECT key,value FROM settings")})
        return settings

    def update_settings(self, values: dict) -> dict:
        with self.db:
            self.db.executemany("INSERT INTO settings(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [(key, json.dumps(value)) for key, value in values.items()])
        return self.get_settings()

    @staticmethod
    def _job(row: sqlite3.Row | None, detail: bool = True, internal: bool = False) -> dict | None:
        if row is None:
            return None
        job = dict(row)
        job["result_available"] = job["status"] == "completed" and (job["result_text"] is not None or job["result_json"] is not None)
        result_json = job.pop("result_json")
        payload = job.pop("payload")
        request = json.loads(payload) if payload else None
        if job["task"] == "captions":
            # The report lives in result_json; there is downloadable text only when captions were produced.
            job["result_available"] = job["status"] == "completed" and job["result_text"] is not None
        if job["task"] == "inspect":
            job["result_available"] = False  # the report is structured; there is nothing to download
        if internal:
            job["encode"] = bool(job["encode"])
            job["result_json"] = json.loads(result_json) if result_json else None
            job["request"] = request
        else:
            job.pop("encode")
            if detail and job["task"] == "detect":
                job["detection"] = json.loads(result_json) if result_json else None
            if detail and job["task"] in ("transcribe", "translate") and result_json:
                # Windows whose engine reply failed the timing checks and how each was recovered (only if any were).
                job["recovery"] = json.loads(result_json).get("recovery")
            if job["task"] == "captions":
                report = json.loads(result_json) if result_json else {}
                job["captions"] = {**(request or {}), **(report if detail else {key: report[key] for key in ("outcome", "cached", "matched") if key in report})}
            if job["task"] == "inspect":
                # The list shows the caller's note; the detail carries the whole report.
                job["note"] = (request or {}).get("note")
                if detail:
                    job["inspection"] = json.loads(result_json) if result_json else None
        if not detail:
            job.pop("result_text")
        return job

    def get_job(self, job_id: str, detail: bool = True, *, internal: bool = False) -> dict | None:
        return self._job(self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone(), detail, internal)

    def list_jobs(self, limit: int = 50, offset: int = 0, status: str | None = None) -> dict:
        where = " WHERE status=?" if status else ""
        args = (status,) if status else ()
        total = self.db.execute("SELECT COUNT(*) FROM jobs" + where, args).fetchone()[0]
        rows = self.db.execute("SELECT * FROM jobs" + where + " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?", (*args, limit, offset)).fetchall()
        return {"items": [self._job(row, False) for row in rows], "total": total}

    def counts(self) -> dict:
        counts = {"completed": 0, "failed": 0}
        counts.update({row[0]: row[1] for row in self.db.execute("SELECT status,COUNT(*) FROM jobs WHERE status IN ('completed','failed') GROUP BY status")})
        return counts

    def close(self) -> None:
        self.db.close()
