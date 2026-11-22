"""Problem alerts to ntfy: one message per problem, repeated at most every ``REPEAT_SECONDS``.

Configured by environment: ``SPDOCK_NTFY_URL`` (e.g. https://ntfy.nateshome.net), ``SPDOCK_NTFY_TOPIC``
and a write-only token in ``SPDOCK_NTFY_TOKEN_FILE`` (default /data/ntfy-token). Without a URL and a
token nothing is sent. Sending never raises: an alert that cannot be delivered is logged and dropped.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import httpx

log = logging.getLogger("speechdock.notify")
REPEAT_SECONDS = 6 * 3600


class Notifier:
    def __init__(self, url: str | None = None, topic: str | None = None, token_file: Path | None = None):
        self.url = (url if url is not None else os.environ.get("SPDOCK_NTFY_URL", "")).rstrip("/")
        self.topic = topic if topic is not None else os.environ.get("SPDOCK_NTFY_TOPIC", "speechdock")
        self.token_file = token_file or Path(os.environ.get("SPDOCK_NTFY_TOKEN_FILE",
                                                            Path(os.environ.get("SPDOCK_DATA_DIR", "/data")) / "ntfy-token"))
        self._last: dict[str, float] = {}  # problem key -> when it was last sent
        self._suppressed: dict[str, int] = {}  # problem key -> repeats not sent since

    @property
    def configured(self) -> bool:
        return bool(self.url and self.topic and self.token_file.is_file())

    async def problem(self, key: str, title: str, message: str) -> bool:
        """Report a problem; the same ``key`` is sent again only after REPEAT_SECONDS, with a count."""
        now = time.monotonic()
        if key in self._last and now - self._last[key] < REPEAT_SECONDS:
            self._suppressed[key] = self._suppressed.get(key, 0) + 1
            return False
        repeats = self._suppressed.pop(key, 0)
        if repeats:
            message += f"\n({repeats} more like this since the last alert.)"
        self._last[key] = now
        return await self.send(title, message, priority="high", tags="warning")

    async def resolved(self, key: str, title: str, message: str) -> bool:
        """Report that a problem sent earlier is over; nothing is sent if it was never reported."""
        if self._last.pop(key, None) is None:
            return False
        self._suppressed.pop(key, None)
        return await self.send(title, message, priority="default", tags="white_check_mark")

    async def send(self, title: str, message: str, *, priority: str = "default", tags: str = "") -> bool:
        if not self.configured:
            log.info("ntfy not configured; alert not sent: %s", title)
            return False
        try:
            token = self.token_file.read_text().strip()
            async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
                response = await client.post(f"{self.url}/{self.topic}", content=message.encode(),
                                             headers={"Authorization": f"Bearer {token}", "Title": f"SpeechDock: {title}",
                                                      "Priority": priority, "Tags": tags})
                response.raise_for_status()
            return True
        except (OSError, httpx.HTTPError) as exc:
            log.warning("ntfy alert not delivered (%s): %s", type(exc).__name__, title)
            return False
