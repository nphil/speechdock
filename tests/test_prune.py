import json
import os
import shutil
import subprocess

import httpx
import pytest

from speechdock import prune
from speechdock.app import app
from speechdock.captions import Ledger
from speechdock.jobs import JobManager
from speechdock.prune import PruneError, parse_keep, plan, run_prune
from speechdock.store import Store

TOOLS = all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe", "mkvmerge"))


def test_keep_accepts_codes_and_names_in_order():
    assert parse_keep(["eng", "Japanese", "en", "ja"]) == ["en", "ja"]
    assert parse_keep(["japanese", "ENG"]) == ["ja", "en"]
    for bad in ([], ["tlh"], "en", [1]):
        with pytest.raises(PruneError):
            parse_keep(bad)


def stream(index, lang, kind="audio", default=0, title=None):
    tags = {"language": lang} if lang is not None else {}
    if title:
        tags["title"] = title
    return {"index": index, "codec_type": kind, "codec_name": "aac", "tags": tags, "disposition": {"default": default}}


def test_only_tagged_other_languages_are_removed():
    streams = [stream(0, None, "video"), stream(1, "jpn"), stream(2, "eng"), stream(3, "por"), stream(4, "und"), stream(5, None), stream(6, "tlh"), stream(7, "eng", "subtitle")]
    remove, kept = plan(streams, ["en", "ja"])
    assert [s["index"] for s in remove] == [3]
    assert [s["index"] for s in kept] == [1, 2, 4, 5, 6]  # untagged and unknown tags stay


async def async_run(job, progress=None):
    async def record(value, duration=None):
        pass
    return (await run_prune(job=job, progress=progress or record))["result_json"]


def make_mkv(path, tracks, duration=4):
    """A small MKV with a colour-bar video and one tone per audio track, tagged as given ((lang, title, default))."""
    command = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=10:duration={duration}"]
    for n in range(len(tracks)):
        command += ["-f", "lavfi", "-i", f"sine=frequency={300 + 100 * n}:duration={duration}"]
    command += ["-map", "0:v"]
    for n in range(len(tracks)):
        command += ["-map", f"{n + 1}:a"]
    command += ["-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac"]
    for n, (lang, title, default) in enumerate(tracks):
        if lang:
            command += [f"-metadata:s:a:{n}", f"language={lang}"]
        if title:
            command += [f"-metadata:s:a:{n}", f"title={title}"]
        command += [f"-disposition:a:{n}", "default" if default else "0"]
    command.append(str(path))
    subprocess.run(command, check=True)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)], check=True, capture_output=True).stdout
    return [(s["codec_type"], (s.get("tags") or {}).get("language"), (s.get("disposition") or {}).get("default")) for s in json.loads(out)["streams"]]


@pytest.mark.skipif(not TOOLS, reason="ffmpeg, ffprobe and mkvmerge are needed")
async def test_a_real_file_loses_only_the_other_language_and_keeps_its_default(tmp_path, monkeypatch):
    monkeypatch.setenv("SPDOCK_PRUNE_ROOTS", json.dumps([str(tmp_path)]))
    video = tmp_path / "show.mkv"
    make_mkv(video, [("por", "Portuguese dub", 1), ("jpn", None, 0), ("eng", None, 0), (None, "untagged", 0)])
    before = video.stat().st_size
    os.utime(video, (1_700_000_000, 1_700_000_000))

    report = await async_run({"request": {"path": str(video), "keep": ["en", "ja"]}})
    assert report["outcome"] == "pruned"
    assert [t["tag"] for t in report["removed"]] == ["por"] and [t["tag"] for t in report["kept"]] == ["jpn", "eng", ""]
    assert report["default_set"] == 3  # the removed track was default; English (keep[0]) is now
    assert report["bytes_after"] < before
    assert probe(video) == [("video", None, 0), ("audio", "jpn", 0), ("audio", "eng", 1), ("audio", None, 0)]
    assert video.stat().st_mtime == 1_700_000_000
    assert not list(tmp_path.glob(".*prune*"))

    again = await async_run({"request": {"path": str(video), "keep": ["en", "ja"]}})
    assert again["outcome"] == "unchanged" and again["removed"] == []


@pytest.mark.skipif(not TOOLS, reason="ffmpeg, ffprobe and mkvmerge are needed")
async def test_dry_run_and_refusals_leave_the_file_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("SPDOCK_PRUNE_ROOTS", json.dumps([str(tmp_path / "allowed")]))
    (tmp_path / "allowed").mkdir()
    video = tmp_path / "allowed" / "show.mkv"
    make_mkv(video, [("por", None, 1), ("eng", None, 0)])
    before = video.stat().st_size

    report = await async_run({"request": {"path": str(video), "keep": ["en"], "dry_run": True}})
    assert report["outcome"] == "dry_run" and [t["tag"] for t in report["removed"]] == ["por"]
    assert video.stat().st_size == before

    with pytest.raises(PruneError, match="No audio track in a kept language"):
        await async_run({"request": {"path": str(video), "keep": ["ja"]}})
    assert video.stat().st_size == before

    outside = tmp_path / "show.mkv"
    shutil.copy(video, outside)
    with pytest.raises(PruneError, match="not allowed"):
        await async_run({"request": {"path": str(outside), "keep": ["en"]}})

    mp4 = tmp_path / "allowed" / "show.mp4"
    mp4.write_bytes(b"x")
    with pytest.raises(PruneError, match="Matroska"):
        await async_run({"request": {"path": str(mp4), "keep": ["en"]}})


@pytest.mark.skipif(not TOOLS, reason="ffmpeg, ffprobe and mkvmerge are needed")
async def test_a_rewrite_that_fails_verification_is_discarded(tmp_path, monkeypatch):
    monkeypatch.setenv("SPDOCK_PRUNE_ROOTS", json.dumps([str(tmp_path)]))
    video = tmp_path / "show.mkv"
    make_mkv(video, [("por", None, 1), ("eng", None, 0)])
    before = probe(video)

    async def short_decode(path):
        return (False, "frame decode error") if path.name.startswith(".") else (True, "")

    monkeypatch.setattr(prune, "decode_check", short_decode)
    with pytest.raises(PruneError, match="does not decode cleanly"):
        await async_run({"request": {"path": str(video), "keep": ["en"]}})
    assert probe(video) == before and not list(tmp_path.glob(".*prune*"))


class Models:
    async def list_models(self):
        return []


@pytest.fixture
async def service(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    downloads = tmp_path / "downloads"
    (downloads / "usenet").mkdir(parents=True)
    (downloads / "usenet" / "a.mkv").write_bytes(b"x")
    (downloads / "torrent.mkv").write_bytes(b"x")
    seen = []

    async def processor(job, path, progress):
        seen.append(job["request"])
        return {"result_json": {"path": job["request"]["path"], "keep": job["request"]["keep"], "outcome": "unchanged", "removed": [], "kept": [], "bytes_before": 1}}

    jobs = JobManager(store, processor)
    monkeypatch.setenv("SPDOCK_INSPECT_ROOTS", '{"/downloads": "%s"}' % downloads)
    monkeypatch.setenv("SPDOCK_CAPTION_ROOTS", '{"/tv": "%s"}' % (tmp_path / "tv"))
    monkeypatch.setenv("SPDOCK_PRUNE_ROOTS", json.dumps([str(downloads / "usenet")]))
    for name, value in (("store", store), ("models", Models()), ("jobs", jobs), ("ledger", Ledger(store.db))):
        monkeypatch.setattr(app.state, name, value, raising=False)
    await jobs.start()
    yield type("Service", (), {"store": store, "jobs": jobs, "seen": seen, "downloads": downloads})()
    await jobs.close()
    store.close()


def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_prune_endpoint_waits_for_the_report_and_only_writes_inside_allowed_folders(service):
    async with client() as http:
        response = await http.post("/api/prune", json={"path": "/downloads/usenet/a.mkv", "keep": ["eng", "Japanese"], "source": "nzbget", "note": "after the gate"})
        assert response.status_code == 200, response.text
        assert response.json()["keep"] == ["en", "ja"] and service.seen[0]["keep"] == ["en", "ja"] and service.seen[0]["dry_run"] is False
        listing = (await http.get("/api/jobs")).json()["items"][0]
        assert listing["task"] == "prune" and listing["source"] == "nzbget" and listing["note"] == "after the gate" and listing["prune"] == {"outcome": "unchanged"}
        detail = (await http.get(f"/api/jobs/{response.json()['job_id']}")).json()
        assert detail["prune"]["removed"] == [] and detail["result_available"] is False
        assert (await http.patch(f"/api/inspect/{response.json()['job_id']}", json={"note": "kept"})).json()["note"] == "kept"

        forbidden = await http.post("/api/prune", json={"path": "/downloads/torrent.mkv", "keep": ["en"]})
        assert forbidden.status_code == 422 and "not allowed" in forbidden.json()["detail"]
        assert (await http.post("/api/prune", json={"path": "/downloads/usenet/a.mkv", "keep": []})).status_code == 422
        assert (await http.post("/api/prune", json={"path": "/downloads/usenet/a.mkv", "keep": ["tlh"]})).status_code == 422
        assert (await http.post("/api/prune", json={"path": "/downloads/usenet/missing.mkv", "keep": ["en"]})).status_code == 422
    assert len(service.seen) == 1
