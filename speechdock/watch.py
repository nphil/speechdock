"""Scheduled subtitle tidy-up: every ``SPDOCK_TIDY_INTERVAL_MINUTES`` (default 30), ask Sonarr and Radarr
which files they imported and Bazarr which subtitles it downloaded since the last look, and queue a
tidy-up job for each of those videos only. Nothing else in the library is read.

Configuration (environment; each API key in a file, never in the template):
``SPDOCK_SONARR_URL`` + ``/data/sonarr-key``, ``SPDOCK_RADARR_URL`` + ``/data/radarr-key``,
``SPDOCK_BAZARR_URL`` + ``/data/bazarr-key`` (``SPDOCK_<NAME>_KEY_FILE`` to move a key). A source
without a URL or key is skipped. Paths come back as the *arr apps see them (/tv/..., /movies/...) and
are mapped into the media mounts like webhook paths. A source that keeps failing for
``ALERT_AFTER_SECONDS`` raises one ntfy alert, and another when it recovers.

Anime (two subtitle files) is decided from the apps themselves: Sonarr's series type "anime", or a
Radarr movie with genre Anime, or Animation with Japanese as the original language.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from .captions import media

log = logging.getLogger("speechdock.watch")
ALERT_AFTER_SECONDS = 2 * 3600
FIRST_LOOK = timedelta(days=2)  # how far back the very first look reaches
VIDEO_EXT = {".mkv", ".mp4", ".m4v", ".avi", ".mov", ".ts", ".m2ts", ".wmv", ".mpg", ".mpeg", ".webm"}


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    key: str


def sources() -> dict[str, Source]:
    data = Path(os.environ.get("SPDOCK_DATA_DIR", "/data"))
    found = {}
    for name in ("sonarr", "radarr", "bazarr"):
        url = os.environ.get(f"SPDOCK_{name.upper()}_URL", "").rstrip("/")
        key_file = Path(os.environ.get(f"SPDOCK_{name.upper()}_KEY_FILE", data / f"{name}-key"))
        if url and key_file.is_file():
            found[name] = Source(name, url, key_file.read_text().strip())
    return found


def _parse_time(value: str) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


class Library:
    """What the *arr apps and Bazarr know: recent changes and which titles are anime."""

    def __init__(self, found: dict[str, Source], client: httpx.AsyncClient):
        self.sources, self.client = found, client

    async def get(self, source: Source, path: str, params: dict | None = None):
        header = "X-API-KEY" if source.name == "bazarr" else "X-Api-Key"
        response = await self.client.get(source.url + path, params=params, headers={header: source.key})
        response.raise_for_status()
        return response.json()

    async def imports(self, name: str, since: datetime) -> list[str]:
        events = await self.get(self.sources[name], "/api/v3/history/since",
                                {"date": since.strftime("%Y-%m-%dT%H:%M:%SZ"), "eventType": "downloadFolderImported"})
        return [e["data"]["importedPath"] for e in events if (e.get("data") or {}).get("importedPath")]

    async def subtitle_downloads(self, since: datetime) -> list[str]:
        """Video paths Bazarr gave a subtitle (downloaded, upgraded, synced) since ``since``, newest first."""
        found = []
        for kind in ("episodes", "movies"):
            start = 0
            while True:
                page = await self.get(self.sources["bazarr"], f"/api/{kind}/history", {"start": start, "length": 200})
                items = page.get("data") or []
                for item in items:
                    stamp = _parse_time(item.get("parsed_timestamp") or item.get("timestamp") or "")
                    if stamp is not None and stamp < since:
                        items = []
                        break
                    if item.get("subtitles_path"):
                        found.append(item["subtitles_path"])
                if not items or start + 200 >= (page.get("total") or 0):
                    break
                start += 200
        return found

    async def anime_folders(self) -> set[str]:
        folders = set()
        if "sonarr" in self.sources:
            for series in await self.get(self.sources["sonarr"], "/api/v3/series"):
                if series.get("seriesType") == "anime" and series.get("path"):
                    folders.add(series["path"].rstrip("/"))
        if "radarr" in self.sources:
            for movie in await self.get(self.sources["radarr"], "/api/v3/movie"):
                genres = set(movie.get("genres") or [])
                japanese = (movie.get("originalLanguage") or {}).get("name") == "Japanese"
                if movie.get("path") and ("Anime" in genres or ("Animation" in genres and japanese)):
                    folders.add(movie["path"].rstrip("/"))
        return folders


def video_for(path: str, roots: dict[str, str]) -> Path | None:
    """The video a path names: the path itself, or for a subtitle the video whose name prefixes it."""
    try:
        resolved = media.resolve_media_path(path, roots)
        return resolved
    except media.MediaError:
        pass
    # A subtitle path: map its folder, then find the video it belongs to.
    for prefix, local in roots.items():
        if path.startswith(prefix.rstrip("/") + "/"):
            folder = Path(local + path[len(prefix.rstrip("/")):]).parent
            name = Path(path).name
            if folder.is_dir():
                videos = [p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXT and name.startswith(p.stem + ".")]
                if videos:
                    return max(videos, key=lambda p: len(p.stem))
    return None


class Watcher:
    """The scheduled look; ``submit(path, anime)`` queues one tidy-up job."""

    def __init__(self, store, notifier, submit, interval_minutes: float | None = None):
        self.store, self.notifier, self.submit = store, notifier, submit
        self.interval = 60 * float(interval_minutes or os.environ.get("SPDOCK_TIDY_INTERVAL_MINUTES", 30))
        self._task: asyncio.Task | None = None
        self._failing_since: dict[str, float] = {}
        self.last_result: dict | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _loop(self) -> None:
        await asyncio.sleep(60)  # let the engine and the queue come up first
        while True:
            if self.store.get_settings().get("auto_tidy"):
                try:
                    self.last_result = await self.look()
                except Exception:
                    log.exception("Scheduled subtitle tidy-up failed")
            await asyncio.sleep(self.interval)

    def _since(self, name: str) -> datetime:
        value = self.store.get_settings().get("tidy_checkpoints", {}).get(name)
        return _parse_time(value) or datetime.now(timezone.utc) - FIRST_LOOK

    def _checkpoint(self, name: str, when: datetime) -> None:
        points = dict(self.store.get_settings().get("tidy_checkpoints", {}))
        points[name] = when.isoformat()
        self.store.update_settings({"tidy_checkpoints": points})

    async def look(self) -> dict:
        """One scheduled look: queue a tidy-up for every video changed since the last look."""
        found = sources()
        started = datetime.now(timezone.utc)
        roots = media.media_roots()
        paths: list[str] = []
        failed: dict[str, str] = {}
        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            library = Library(found, client)
            # Which titles are anime decides the files a video gets, so without the list nothing is queued and
            # no checkpoint moves: the next look picks the same changes up.
            try:
                anime = await library.anime_folders()
                await self._recovered("anime list")
            except (httpx.HTTPError, ValueError) as exc:
                failed["anime list"] = type(exc).__name__
                await self._failing("anime list", failed["anime list"])
                return {"at": started.isoformat(), "changed": 0, "queued": 0, "not_found": 0, "failed": failed}
            for name in ("sonarr", "radarr", "bazarr"):
                if name not in found:
                    continue
                try:
                    since = self._since(name)
                    got = await (library.subtitle_downloads(since) if name == "bazarr" else library.imports(name, since))
                    paths += got
                    self._checkpoint(name, started)
                    await self._recovered(name)
                except (httpx.HTTPError, ValueError, KeyError) as exc:
                    failed[name] = type(exc).__name__
                    await self._failing(name, failed[name])
        queued, unknown = 0, 0
        for path in dict.fromkeys(paths):
            video = await asyncio.to_thread(video_for, path, roots)
            if video is None:
                unknown += 1
                continue
            if await self.submit(str(video), any(path.startswith(folder + "/") for folder in anime)):
                queued += 1
        result = {"at": started.isoformat(), "changed": len(dict.fromkeys(paths)), "queued": queued,
                  "not_found": unknown, "failed": failed}
        log.info("Scheduled subtitle tidy-up: %s", result)
        return result

    async def _failing(self, name: str, reason: str) -> None:
        first = self._failing_since.setdefault(name, time.monotonic())
        if time.monotonic() - first >= ALERT_AFTER_SECONDS:
            await self.notifier.problem(f"source:{name}", f"{name.capitalize()} unreachable",
                                        f"The scheduled subtitle tidy-up cannot read {name.capitalize()} ({reason}) "
                                        f"for over {ALERT_AFTER_SECONDS // 3600} hours. New downloads wait until it answers.")

    async def _recovered(self, name: str) -> None:
        if self._failing_since.pop(name, None) is not None:
            await self.notifier.resolved(f"source:{name}", f"{name.capitalize()} reachable again",
                                         f"The scheduled subtitle tidy-up reads {name.capitalize()} again.")
