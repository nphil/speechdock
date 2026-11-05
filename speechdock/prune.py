"""Audio pruning: keep only the wanted audio languages of a Matroska file, by stream copy.

A caller names the languages to keep; every other tagged audio track is removed with mkvmerge
(no re-encoding), the result is checked against the original (stream counts, duration, chapters,
codecs, tags, default flag, a decode check of the head and tail) and only then renamed over the
original. Untagged tracks are kept. Nothing here decides what to keep; that is the caller's policy.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

from .audio import tool_failure
from .captions import media
from .captions.media import MediaError
from .inspect import language_code
from .languages import LANGUAGES, normalize_language

MIN_FREE_BYTES = 1 << 30
DECODE_SECONDS = 10
# ISO 639-2 bibliographic/terminologic pairs: a remux may write the other spelling of the same language.
_SAME = {}
for pair in ("fre fra", "ger deu", "chi zho", "cze ces", "dut nld", "per fas", "rum ron", "slo slk", "gre ell", "arm hye",
             "baq eus", "bur mya", "geo kat", "ice isl", "mac mkd", "may msa", "mao mri", "alb sqi", "tib bod", "wel cym"):
    a, b = pair.split()
    _SAME[a] = _SAME[b] = a


class PruneError(ValueError):
    """The file was left untouched; the message says why."""


def prune_roots() -> list[str]:
    """Mount paths prune may write to: ``SPDOCK_PRUNE_ROOTS`` (JSON list), else the media mounts only."""
    raw = os.environ.get("SPDOCK_PRUNE_ROOTS")
    roots = json.loads(raw) if raw else list(media.media_roots().values())
    if not isinstance(roots, list) or not roots or any(not isinstance(r, str) or not r.startswith("/") for r in roots):
        raise ValueError("SPDOCK_PRUNE_ROOTS must be a JSON list of absolute paths")
    return [os.path.normpath(r) for r in roots]


def check_writable_root(path: Path) -> None:
    real = os.path.realpath(path)
    if not any(real.startswith(os.path.realpath(root) + "/") for root in prune_roots()):
        raise PruneError("Audio pruning is not allowed in this folder")


def parse_keep(value: object) -> list[str]:
    """Language codes or names to keep, in order of preference (the first is made default if needed)."""
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise PruneError("keep must be a non-empty list of language codes or names")
    codes = []
    for item in value:
        code = language_code(item)
        if code not in LANGUAGES:
            try:
                code = normalize_language(item)
            except ValueError:
                raise PruneError(f"Unknown language in keep: {item}") from None
        if code not in codes:
            codes.append(code)
    return codes


def _canon_tag(tag: str) -> str:
    tag = (tag or "").strip().lower()
    return _SAME.get(tag, tag)


def _lang_title(stream: dict) -> tuple[str, str]:
    return _canon_tag(media.tag(stream, "language")), media.tag(stream, "title")


async def _run(command: list[str], timeout: float) -> tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(*command, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    return process.returncode, out, err


async def probe_full(path: Path) -> tuple[dict, str]:
    """Every stream, chapter and the format; plus ffprobe's warnings (a damaged file prints some)."""
    code, out, err = await _run(["ffprobe", "-v", "warning", "-show_streams", "-show_chapters", "-show_format", "-of", "json", str(path)], 600)
    if code:
        raise MediaError(tool_failure("Could not read the video", "ffprobe", code, err))
    try:
        return json.loads(out), err.decode(errors="replace").strip()
    except ValueError as exc:
        raise MediaError("Could not read the video's stream list") from exc


async def decode_check(path: Path) -> tuple[bool, str]:
    """Decode the first and last seconds of the video and every audio track; (ok, error text)."""
    for prefix in (["-t", str(DECODE_SECONDS)], ["-sseof", f"-{DECODE_SECONDS}"]):
        code, _, err = await _run(["ffmpeg", "-nostdin", "-v", "error", *prefix, "-i", str(path), "-map", "0:v:0?", "-map", "0:a", "-f", "null", "-"], 1800)
        text = "\n".join(line for line in err.decode(errors="replace").splitlines() if "non monotonically increasing dts" not in line).strip()
        if code or text:
            return False, text[-300:] or f"ffmpeg exited with code {code}"
    return True, ""


def plan(streams: list[dict], keep: list[str]) -> tuple[list[dict], list[dict]]:
    """(audio tracks to remove, audio tracks kept). A track is removed only when its tag names a known
    language that is not kept; untagged tracks and tags nobody can name are kept."""
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    remove, kept = [], []
    for stream in audio:
        code = language_code(media.tag(stream, "language"))
        (remove if code in LANGUAGES and code not in keep else kept).append(stream)
    return remove, kept


def _stream_info(stream: dict) -> dict:
    tag = media.tag(stream, "language")
    code = language_code(tag)
    return {"index": stream.get("index"), "codec": stream.get("codec_name"), "channels": stream.get("channels"), "tag": tag,
            "language": code, "name": LANGUAGES[code].title() if code in LANGUAGES else None, "title": media.tag(stream, "title") or None,
            "default": bool((stream.get("disposition") or {}).get("default"))}


async def run_prune(*, job: dict, progress) -> dict:
    """The job: decide, remux to a temporary file beside the original, verify, replace."""
    request = job["request"]
    path = Path(request["path"])
    keep = parse_keep(request["keep"])
    check_writable_root(path)
    if path.suffix.lower() != ".mkv":
        raise PruneError("Only Matroska (.mkv) files are pruned")
    started = time.monotonic()
    original, original_warnings = await probe_full(path)
    streams = original.get("streams", [])
    remove, kept = plan(streams, keep)
    report = {"path": str(path), "keep": keep, "removed": [_stream_info(s) for s in remove], "kept": [_stream_info(s) for s in kept],
              "bytes_before": path.stat().st_size, "outcome": "unchanged"}
    await progress(0.1, None)
    if not remove:
        return {"result_json": report}
    if not any(language_code(media.tag(s, "language")) in keep for s in kept):
        raise PruneError("No audio track in a kept language would remain; the file was left untouched")
    stat = path.stat()
    if shutil.disk_usage(path.parent).free < stat.st_size + MIN_FREE_BYTES:
        raise PruneError("Not enough free space beside the file for a safe rewrite")
    if request.get("dry_run"):
        report["outcome"] = "dry_run"
        return {"result_json": report}

    temp = path.with_name(f".{path.stem}.prune{path.suffix}")
    temp.unlink(missing_ok=True)
    try:
        command = ["mkvmerge", "-o", str(temp), "--audio-tracks", "!" + ",".join(str(s["index"]) for s in remove)]
        default_fix = None
        if any((s.get("disposition") or {}).get("default") for s in remove) and not any((s.get("disposition") or {}).get("default") for s in kept):
            preferred = next((s for code in keep for s in kept if language_code(media.tag(s, "language")) == code), kept[0])
            command += ["--default-track-flag", f"{preferred['index']}:1"]
            default_fix = preferred["index"]
        command.append(str(path))
        code, out, err = await _run(command, 7200)
        await progress(0.6, None)
        if code >= 2 or not temp.is_file():
            raise PruneError(tool_failure("mkvmerge could not rewrite the file", "mkvmerge", code, out + err))
        warnings = [line for line in out.decode(errors="replace").splitlines() if line.startswith("Warning")] if code == 1 else []

        new, new_warnings = await probe_full(temp)
        if new_warnings and not original_warnings:
            raise PruneError("ffprobe reports a problem with the rewritten file: " + new_warnings[-200:])
        try:
            drift = abs(float(original["format"]["duration"]) - float(new["format"]["duration"]))
        except (KeyError, TypeError, ValueError):
            raise PruneError("The rewritten file has no readable duration") from None
        if drift > 1:
            raise PruneError(f"The rewritten file's duration differs by {drift:.2f} s")
        removed_indexes = {s["index"] for s in remove}
        expected = Counter(s["codec_type"] for s in streams if s["index"] not in removed_indexes)
        actual = Counter(s["codec_type"] for s in new.get("streams", []))
        if actual != expected:
            raise PruneError(f"Stream counts differ: {dict(actual)} instead of {dict(expected)}")
        if len(original.get("chapters", [])) != len(new.get("chapters", [])):
            raise PruneError("The chapter count differs")
        kept_original = [s for s in streams if s["index"] not in removed_indexes]
        if [s.get("codec_name") for s in kept_original] != [s.get("codec_name") for s in new["streams"]]:
            raise PruneError("The stream codecs differ")
        if [_lang_title(s) for s in kept_original if s["codec_type"] != "attachment"] != [_lang_title(s) for s in new["streams"] if s["codec_type"] != "attachment"]:
            raise PruneError("The stream language or title tags differ")
        if default_fix is not None and not any((s.get("disposition") or {}).get("default") for s in new["streams"] if s["codec_type"] == "audio"):
            raise PruneError("The rewritten file has no default audio track")
        ok, why = await decode_check(temp)
        if not ok:
            original_ok, _ = await decode_check(path)
            if original_ok:
                raise PruneError("The rewritten file does not decode cleanly: " + why)
            warnings.append("decode warnings are present in the original too; accepted: " + why[:120])
        await progress(0.9, None)

        mode = stat.st_mode & 0o7777
        if stat.st_uid != os.getuid():
            mode |= 0o666  # the original owner (download client, media manager) must still be able to replace or hardlink it
        os.chmod(temp, mode)
        os.utime(temp, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    report.update({"outcome": "pruned", "bytes_after": path.stat().st_size, "seconds": round(time.monotonic() - started, 1),
                   "default_set": default_fix, "warnings": warnings})
    return {"result_json": report}
