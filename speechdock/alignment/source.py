"""Trusted configured PMS identity resolution and read-only, descriptor-pinned media access."""
from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import stat
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import httpx
import numpy as np

from .core import Rejected
from .protocol import Window, manifest_hash

MAX_SOURCE = 4 * 1024 * 1024
# Exclude playlists and demuxers capable of following arbitrary nested sources.
FORMATS = "matroska,webm,mov,mp4,m4a,3gp,3g2,mj2,mpegts,avi,wav,flac,ogg"


@dataclass(frozen=True)
class Configuration:
    url: str
    token_file: Path
    roots: tuple[tuple[str, str], ...]
    model_dir: Path

    @classmethod
    def environment(cls):
        try:
            url = os.environ["SPDOCK_ALIGNMENT_PLEX_URL"].rstrip("/")
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path:
                raise ValueError()
            mappings = json.loads(os.environ["SPDOCK_ALIGNMENT_ROOTS"])
            roots = tuple((entry["plex"], entry["local"]) for entry in mappings)
            if not roots or len(roots) > 16:
                raise ValueError()
            for pair in roots:
                for root in pair:
                    if not root.startswith("/") or ".." in root.split("/") or root == "/":
                        raise ValueError()
            return cls(url, Path(os.environ["SPDOCK_ALIGNMENT_PLEX_TOKEN_FILE"]), roots, Path(os.environ["SPDOCK_ALIGNMENT_MODEL_DIR"]))
        except (KeyError, TypeError, ValueError):
            raise Rejected("alignment_not_configured") from None


def revision(fd: int) -> tuple[int, ...]:
    value = os.fstat(fd)
    if not stat.S_ISREG(value.st_mode):
        raise Rejected("media_not_regular")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


@contextmanager
def open_media(source: str, roots: tuple[tuple[str, str], ...]):
    """Walk every directory with O_NOFOLLOW; no resolve/check/open race or symlinks."""
    if not source.startswith("/") or any(item in {"..", "."} for item in source.split("/")):
        raise Rejected("media_root_denied")
    matches = [(prefix, local) for prefix, local in roots if source.startswith(prefix.rstrip("/") + "/")]
    if len(matches) != 1:
        raise Rejected("media_root_denied")
    prefix, local = matches[0]
    relative = source[len(prefix.rstrip("/")) + 1:]
    parts = PurePosixPath(local).parts[1:] + PurePosixPath(relative).parts
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, component in enumerate(parts):
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if index < len(parts) - 1:
                flags |= os.O_DIRECTORY
            child = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        revision(fd)
        yield fd
    except OSError:
        raise Rejected("media_open_denied") from None
    finally:
        os.close(fd)


def command(args: list[str], fd: int, output_limit: int, timeout: float = 30) -> tuple[bytes, bytes]:
    """Bound stdout AND stderr while running, not after allocating unbounded output."""
    process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(fd,))
    output = [bytearray(), bytearray()]
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            for index, stream in enumerate((process.stdout, process.stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, index)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise Rejected("source_process_timeout")
                for key, _ in selector.select(min(remaining, 0.2)):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    index = key.data
                    if len(output[index]) + len(chunk) > (output_limit if index == 0 else 1024 * 1024):
                        raise Rejected("source_output_limit")
                    output[index].extend(chunk)
        if process.wait(timeout=max(0.01, deadline - time.monotonic())):
            raise Rejected("source_decode_failed")
        return bytes(output[0]), bytes(output[1])
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()


def probe(fd: int) -> dict:
    body, _ = command(["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
                       "-format_whitelist", FORMATS, "-show_streams", "-show_format",
                       "-of", "json", f"/proc/self/fd/{fd}"], fd, 1024 * 1024)
    try:
        return json.loads(body)
    except ValueError:
        raise Rejected("invalid_container_metadata") from None


def stream_index(stream: dict, inventory: dict, kind: str) -> int:
    try:
        index = int(stream["index"])
        actual = [item for item in inventory["streams"] if item["index"] == index]
        if len(actual) != 1 or actual[0]["codec_type"] != kind:
            raise ValueError()
        actual = actual[0]
        aliases = {"srt": "subrip", "mov_text": "mov_text"}
        if aliases.get(stream["codec"], stream["codec"]) != actual["codec_name"]:
            raise ValueError()
        if kind == "audio":
            if int(stream["channels"]) != actual["channels"]:
                raise ValueError()
            language = stream.get("languageCode", "")
            if language not in {"en", "eng"} or actual.get("tags", {}).get("language") not in {"en", "eng"}:
                raise Rejected("unsupported_audio_language")
        return index
    except (KeyError, ValueError, TypeError):
        raise Rejected("selected_stream_unverified") from None


def parse_subtitles(body: bytes) -> list[tuple[int, int, str]]:
    """Plain SRT/VTT only; preserve source order, duplicates and overlaps. Never strip markup."""
    try:
        text = body.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeError:
        raise Rejected("subtitle_encoding_unsupported") from None
    stamp = r"(?:(\d{2,}):)?(\d{2}):(\d{2})[,.](\d{3})"
    timing = re.compile(rf"^{stamp} --> {stamp}$")
    result = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()
        if not lines or lines[0].startswith("WEBVTT"):
            continue
        position = 0 if " --> " in lines[0] else 1
        if position >= len(lines) or not (match := timing.fullmatch(lines[position])):
            raise Rejected("subtitle_format_unsupported")
        values = match.groups()
        def milliseconds(parts):
            hour, minute, second, milli = parts
            if int(minute) > 59 or int(second) > 59:
                raise Rejected("subtitle_bounds_invalid")
            return ((int(hour or 0) * 60 + int(minute)) * 60 + int(second)) * 1000 + int(milli)
        start, end = milliseconds(values[:4]), milliseconds(values[4:])
        cue_text = "\n".join(lines[position + 1:])
        if not cue_text or end <= start or any(char in cue_text for char in "<>\\{}"):
            raise Rejected("subtitle_format_unsupported")
        result.append((start, end, cue_text))
    if not result or len(result) > 100000:
        raise Rejected("subtitle_source_empty_or_large")
    return result


class PlexSource:
    def __init__(self, config: Configuration):
        self.config = config

    def fetch(self, path: str) -> bytes:
        # All callers supply fixed endpoints or a validated fresh PMS Stream key.
        try:
            token = self.config.token_file.read_text().strip()
            if not token or len(token) > 4096 or "\n" in token or "\r" in token:
                raise Rejected("plex_token_unavailable")
            with httpx.Client(timeout=10, follow_redirects=False, trust_env=False) as client:
                with client.stream("GET", self.config.url + path, headers={"X-Plex-Token": token, "Accept": "application/xml"}) as response:
                    if response.status_code != 200:
                        raise Rejected("plex_source_unavailable")
                    body = bytearray()
                    deadline = time.monotonic() + 15
                    for chunk in response.iter_bytes(chunk_size=65536):
                        if time.monotonic() > deadline:
                            raise Rejected("plex_source_timeout")
                        if len(body) + len(chunk) > MAX_SOURCE:
                            raise Rejected("plex_source_too_large")
                        body.extend(chunk)
                    return bytes(body)
        except (OSError, httpx.HTTPError):
            raise Rejected("plex_source_unavailable") from None

    def xml(self, path: str):
        body = self.fetch(path)
        if b"<!DOCTYPE" in body.upper() or b"<!ENTITY" in body.upper():
            raise Rejected("plex_metadata_invalid")
        try:
            return ET.fromstring(body)
        except ET.ParseError:
            raise Rejected("plex_metadata_invalid") from None

    def resolve(self, request: Window) -> dict:
        identity = request.identity
        server = self.xml("/identity")
        if server.get("machineIdentifier") != identity.server_machine_id:
            raise Rejected("server_identity_changed")
        metadata = self.xml("/library/metadata/" + identity.rating_key)
        matches = []
        for video in metadata.findall("Video"):
            if video.get("ratingKey") != identity.rating_key:
                continue
            for media in video.findall("Media"):
                if media.get("id") != identity.media_id:
                    continue
                parts = media.findall("Part")
                if len(parts) != 1:
                    raise Rejected("multipart_unsupported")
                part = parts[0]
                if part.get("id") == identity.part_id and part.get("key") == identity.part_key:
                    matches.append(part)
        if len(matches) != 1:
            raise Rejected("media_identity_changed")
        part = matches[0]
        def selected(stream_id, kind):
            items = [dict(item.attrib) for item in part.findall("Stream") if item.get("id") == stream_id and item.get("streamType") == kind]
            if len(items) != 1:
                raise Rejected("selected_stream_missing")
            return items[0]
        audio = selected(identity.audio_stream_id, "2")
        subtitle = selected(identity.subtitle_stream_id, "3")
        if subtitle.get("codec") != identity.subtitle_codec or subtitle.get("codec") not in {"srt", "subrip", "webvtt", "mov_text", "text"}:
            raise Rejected("subtitle_codec_unsupported")
        if subtitle.get("languageCode") not in {"en", "eng"}:
            raise Rejected("unsupported_subtitle_language")
        if not part.get("file"):
            raise Rejected("media_file_unavailable")
        return {"part": dict(part.attrib), "audio": audio, "subtitle": subtitle}

    def subtitle_bytes(self, resolved: dict, request: Window, fd: int, inventory: dict) -> bytes:
        subtitle = resolved["subtitle"]
        key = subtitle.get("key")
        if request.subtitle_source_kind == "sidecar":
            if key != "/library/streams/" + request.identity.subtitle_stream_id:
                raise Rejected("subtitle_source_kind_mismatch")
            for suffix in ("?encoding=utf-8&format=srt", "?encoding=utf-8", ""):
                try:
                    return self.fetch(key + suffix)
                except Rejected as exc:
                    if str(exc) != "plex_source_unavailable":
                        raise
            raise Rejected("subtitle_source_unavailable")
        if key:
            raise Rejected("subtitle_source_kind_mismatch")
        index = stream_index(subtitle, inventory, "subtitle")
        body, _ = command(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-copyts",
                           "-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS,
                           "-i", f"/proc/self/fd/{fd}", "-map", f"0:{index}",
                           "-c:s", "srt", "-f", "srt", "pipe:1"], fd, MAX_SOURCE)
        return body

    def verify_cues(self, request: Window, body: bytes) -> dict[str, str]:
        expected = hashlib.sha256(body).hexdigest() if request.subtitle_source_kind == "sidecar" else manifest_hash(request.cues)
        if expected != request.subtitle_source_sha256:
            raise Rejected("subtitle_hash_mismatch")
        source = parse_subtitles(body)
        available = Counter(source)
        requested = Counter((cue.authored_start_ms, cue.authored_end_ms, cue.text) for cue in request.cues)
        if any(count > available[key] for key, count in requested.items()):
            raise Rejected("subtitle_manifest_mismatch")
        if request.subtitle_source_kind == "sidecar":
            for cue in request.cues:
                if cue.ordinal >= len(source) or source[cue.ordinal] != (cue.authored_start_ms, cue.authored_end_ms, cue.text):
                    raise Rejected("subtitle_manifest_mismatch")
        rejected = {}
        for cue in request.cues:
            key = (cue.authored_start_ms, cue.authored_end_ms, cue.text)
            if available[key] > 1:
                rejected[cue.cue_id] = "ambiguous_duplicate_cue"
            elif any(other != key and max(other[0], key[0]) < min(other[1], key[1]) for other in source):
                rejected[cue.cue_id] = "overlapping_source_cues"
        return rejected


def decode_audio(fd: int, index: int, start_ms: int, end_ms: int, origin_ms: int, inventory: dict) -> tuple[np.ndarray, float]:
    """Seek in absolute container PTS, then measure actual PCM PTS after trim/resample."""
    start = (start_ms + origin_ms) / 1000
    end = (end_ms + origin_ms) / 1000
    try:
        container_start = float(inventory["format"]["start_time"])
    except (ValueError, KeyError):
        raise Rejected("container_timeline_unknown") from None
    seek = max(container_start, start - 1)
    # Disable FFmpeg accurate-seek trimming: on nonzero-start containers it may
    # add format.start_time again. We trim raw PTS ourselves, measure them, then
    # reset ONLY the output clock so -t bounds decoding irrespective of origin.
    args = ["ffmpeg", "-nostdin", "-v", "info", "-threads", "1", "-copyts", "-noaccurate_seek",
            "-seek_timestamp", "1", "-ss", str(seek), "-protocol_whitelist", "file,pipe",
            "-format_whitelist", FORMATS, "-i", f"/proc/self/fd/{fd}", "-map", f"0:{index}",
            "-vn", "-sn", "-t", str(end - start), "-af",
            f"atrim=start={start},aresample=16000,ashowinfo,asetpts=PTS-STARTPTS",
            "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"]
    pcm, diagnostic = command(args, fd, 16 * 16000 * 4)
    frames = re.findall(rb"n:\d+ pts:(-?\d+) pts_time:[^ ]+.*?rate:16000 nb_samples:(\d+)", diagnostic)
    if not frames or not pcm or len(pcm) % 4:
        raise Rejected("audio_timeline_unverified")
    cursor = int(frames[0][0])
    first = cursor
    for pts, count in frames:
        if int(pts) != cursor:
            raise Rejected("audio_timeline_discontinuous")
        cursor += int(count)
    if cursor - first < len(pcm) // 4:
        raise Rejected("audio_timeline_unverified")
    actual_origin = first / 16 - origin_ms
    if actual_origin < start_ms - 0.1 or (first + len(pcm) // 4) / 16 - origin_ms > end_ms + 0.1:
        raise Rejected("audio_outside_bounds")
    return np.frombuffer(pcm, dtype="<f4"), actual_origin
