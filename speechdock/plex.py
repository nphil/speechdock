"""Ask Plex to rescan one folder after its subtitle files changed (a partial scan, not the library).

Uses the Plex settings alignment already has: ``SPDOCK_ALIGNMENT_PLEX_URL``, the token file
``SPDOCK_ALIGNMENT_PLEX_TOKEN_FILE`` and ``SPDOCK_ALIGNMENT_ROOTS`` ([{"plex", "local"}] path pairs).
Without them nothing is asked; Plex then finds the files on its own next scan. Never raises.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from xml.etree import ElementTree

import httpx

log = logging.getLogger("speechdock.plex")


def _config() -> tuple[str, str, list[tuple[str, str]]] | None:
    try:
        url = os.environ["SPDOCK_ALIGNMENT_PLEX_URL"].rstrip("/")
        token = Path(os.environ["SPDOCK_ALIGNMENT_PLEX_TOKEN_FILE"]).read_text().strip()
        roots = [(entry["plex"].rstrip("/"), entry["local"].rstrip("/")) for entry in json.loads(os.environ["SPDOCK_ALIGNMENT_ROOTS"])]
    except (KeyError, OSError, ValueError, TypeError):
        return None
    return (url, token, roots) if url and token else None


def plex_path(local: str, roots: list[tuple[str, str]]) -> str | None:
    for plex, mount in roots:
        if local == mount or local.startswith(mount + "/"):
            return plex + local[len(mount):]
    return None


async def refresh_folder(folder: str) -> bool:
    config = _config()
    if config is None:
        return False
    url, token, roots = config
    target = plex_path(folder, roots)
    if target is None:
        return False
    try:
        async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
            sections = await client.get(f"{url}/library/sections", headers={"X-Plex-Token": token})
            sections.raise_for_status()
            for directory in ElementTree.fromstring(sections.text).iter("Directory"):
                for location in directory.iter("Location"):
                    root = (location.get("path") or "").rstrip("/")
                    if root and (target == root or target.startswith(root + "/")):
                        response = await client.get(f"{url}/library/sections/{directory.get('key')}/refresh",
                                                    params={"path": target}, headers={"X-Plex-Token": token})
                        response.raise_for_status()
                        return True
    except (httpx.HTTPError, ElementTree.ParseError) as exc:
        log.warning("Plex folder refresh failed (%s)", type(exc).__name__)
    return False
