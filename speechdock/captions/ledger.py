"""The record of what SpeechDock decided for each video, and which files it wrote.

Ownership rule: a caption file is SpeechDock's only if the ledger says it wrote that path
and the file still has exactly the bytes it wrote. Anything else - another tool's file, or
one a person edited afterwards - is never overwritten, not even when a rerun is forced.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import sqlite3
import uuid
from pathlib import Path

from ..store import utcnow

SKIPPED = ("matched_existing", "no_english_audio", "no_speech")  # decisions that stay valid while the video is unchanged

_SCHEMA = """
CREATE TABLE IF NOT EXISTS captions (
    video TEXT PRIMARY KEY, outcome TEXT NOT NULL, sidecar TEXT, sha256 TEXT,
    video_size INTEGER, video_mtime_ns INTEGER, details TEXT NOT NULL, job_id TEXT, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS captions_sidecar ON captions(sidecar);
"""


def sidecar_for(video: Path) -> Path:
    """Plex lists ``<base>.en.cc.srt`` as English (CC); Bazarr never writes ``.cc``."""
    return video.with_name(video.stem + ".en.cc.srt")


def fingerprint(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            sha.update(block)
    return sha.hexdigest()


def publish(path: Path, data: bytes, *, replace: bool) -> None:
    """Atomically create ``path`` with ``data``; without ``replace`` an existing file is never touched."""
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temp, path)
            return
        try:
            os.link(temp, path)  # fails when the name exists: never clobbers
        except OSError as exc:
            if exc.errno in (errno.EEXIST,):
                raise FileExistsError(path.name) from None
            if exc.errno not in (errno.EPERM, errno.ENOTSUP, errno.ENOSYS, errno.EMLINK, errno.EXDEV):
                raise
            if os.path.lexists(path):  # filesystem without hard links: check, then rename
                raise FileExistsError(path.name) from None
            os.rename(temp, path)
    finally:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass


class Ledger:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.db.executescript(_SCHEMA)
        self.db.commit()

    def get(self, video: Path | str) -> dict | None:
        row = self.db.execute("SELECT * FROM captions WHERE video=?", (str(video),)).fetchone()
        return self._row(row) if row else None

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        item = dict(row)
        item["details"] = json.loads(item["details"])
        return item

    def is_owned(self, sidecar: Path) -> bool:
        """True only for a file this ledger wrote that still holds the bytes it wrote."""
        row = self.db.execute("SELECT sha256 FROM captions WHERE sidecar=? AND sha256 IS NOT NULL", (str(sidecar),)).fetchone()
        try:
            return bool(row) and sidecar.is_file() and digest(sidecar) == row["sha256"]
        except OSError:
            return False

    def reusable(self, row: dict, video: Path, sidecar: Path) -> bool:
        """Whether an earlier decision still stands, so a non-forced rerun can return it instantly."""
        if row["outcome"] == "written":
            return self.is_owned(sidecar)
        try:
            return row["outcome"] in SKIPPED and fingerprint(video) == (row["video_size"], row["video_mtime_ns"])
        except OSError:
            return False

    def record(self, video: Path, outcome: str, details: dict, *, sidecar: Path | None = None,
               sha256: str | None = None, job_id: str | None = None) -> None:
        size, mtime = fingerprint(video)
        with self.db:
            self.db.execute(
                "INSERT INTO captions(video,outcome,sidecar,sha256,video_size,video_mtime_ns,details,job_id,updated_at) VALUES(?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(video) DO UPDATE SET outcome=excluded.outcome,sidecar=COALESCE(excluded.sidecar,captions.sidecar),"
                "sha256=COALESCE(excluded.sha256,captions.sha256),video_size=excluded.video_size,video_mtime_ns=excluded.video_mtime_ns,"
                "details=excluded.details,job_id=excluded.job_id,updated_at=excluded.updated_at",
                (str(video), outcome, str(sidecar) if sidecar else None, sha256, size, mtime, json.dumps(details), job_id, utcnow()))

    def listing(self, limit: int = 50, offset: int = 0, outcome: str | None = None) -> dict:
        where, args = (" WHERE outcome=?", (outcome,)) if outcome else ("", ())
        total = self.db.execute("SELECT COUNT(*) FROM captions" + where, args).fetchone()[0]
        rows = self.db.execute("SELECT * FROM captions" + where + " ORDER BY updated_at DESC, rowid DESC LIMIT ? OFFSET ?", (*args, limit, offset)).fetchall()
        return {"items": [self._row(row) for row in rows], "total": total}
