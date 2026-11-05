import io

import httpx
import pytest

from speechdock.app import app
from speechdock.captions import Ledger, media
from speechdock.captions.routes import ENGLISH_ORIGINAL, UNKNOWN_ORIGINAL, arr_paths, original_language
from speechdock.jobs import JobManager, QueueFull
from speechdock.store import Store


class Upload:
    def __init__(self, data=b"audio"):
        self.file = io.BytesIO(data)

    async def read(self, size):
        return self.file.read(size)


class Models:
    async def list_models(self):
        return [{"model_id": "parakeet", "registered": True, "translation_capable": False, "runtime": "crispasr", "architecture": "parakeet"}]


@pytest.fixture
async def service(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    media = tmp_path / "tv"
    (media / "Show").mkdir(parents=True)
    for name in ("Ep 1.mkv", "Ep 2.mkv", "Ep 3.mkv"):
        (media / "Show" / name).touch()
    store.update_settings({"transcription_model": "parakeet"})
    ran = []

    async def processor(job, path, progress):
        ran.append((job["task"], job["filename"]))
        return {"result_text": None, "duration_seconds": 1, "result_json": {"outcome": "no_speech"}} if job["task"] == "captions" \
            else {"result_text": "", "duration_seconds": 1}

    jobs = JobManager(store, processor)
    monkeypatch.setenv("SPDOCK_CAPTION_ROOTS", '{"/tv": "%s"}' % media)
    for name, value in (("store", store), ("models", Models()), ("jobs", jobs), ("ledger", Ledger(store.db))):
        monkeypatch.setattr(app.state, name, value, raising=False)
    yield type("Service", (), {"store": store, "jobs": jobs, "ran": ran, "media": media})()
    await jobs.close()
    store.close()


def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


JAPANESE = {"id": 8, "name": "Japanese"}


def sonarr(path, event="Download", language=JAPANESE, **extra):
    series = {"path": "/tv/Show", "originalLanguage": language}
    return {"eventType": event, "series": series, "episodes": [{"id": 1}], "episodeFile": {"id": 7, "relativePath": path.rsplit("/", 1)[-1], "path": path}, **extra}


def test_webhook_paths_come_from_sonarr_and_radarr_payloads():
    assert arr_paths(sonarr("/tv/Show/Ep 1.mkv")) == ["/tv/Show/Ep 1.mkv"]
    assert arr_paths({"eventType": "Download", "movie": {"folderPath": "/movies/A (2020)"}, "movieFile": {"relativePath": "A.mkv"}}) == ["/movies/A (2020)/A.mkv"]
    assert arr_paths({"episodeFiles": [{"path": "/tv/a.mkv"}, {"path": "/tv/b.mkv"}, "junk"], "episodeFile": {"path": "/tv/c.mkv"}}) == ["/tv/c.mkv", "/tv/a.mkv", "/tv/b.mkv"]
    assert arr_paths({"eventType": "Download"}) == []


async def test_captions_run_behind_bazarr_and_never_use_up_its_queue(service):
    jobs = service.jobs
    created = [await jobs.submit_captions(source="captions", path=str(service.media / "Show" / f"Ep {n}.mkv"), force=False, model="parakeet") for n in (1, 2, 3)]
    assert all(flag for _, flag in created)
    for _ in range(8):  # eight waiting Bazarr requests still fit while captions are queued
        await jobs.submit(Upload(), source="bazarr", filename="x.mkv", task="transcribe", language="en", model="parakeet")
    with pytest.raises(QueueFull):
        await jobs.submit(Upload(), source="bazarr", filename="x.mkv", task="transcribe", language="en", model="parakeet")
    await jobs.start()
    for job, _ in created:
        await jobs.wait(job["id"])
    assert [task for task, _ in service.ran] == ["transcribe"] * 8 + ["captions"] * 3


async def test_a_video_is_not_queued_twice_and_a_forced_request_upgrades_the_waiting_job(service):
    jobs, path = service.jobs, str(service.media / "Show" / "Ep 1.mkv")
    first, created = await jobs.submit_captions(source="captions", path=path, force=False, model="parakeet")
    again, created_again = await jobs.submit_captions(source="webhook", path=path, force=True, model="parakeet")
    assert created and not created_again and again["id"] == first["id"]
    assert jobs.store.get_job(first["id"], internal=True)["request"] == {"path": path, "force": True}
    assert jobs.store.list_jobs()["total"] == 1


async def test_caption_jobs_show_their_report_in_the_job_views(service):
    jobs = service.jobs
    await jobs.start()
    job, _ = await jobs.submit_captions(source="captions", path=str(service.media / "Show" / "Ep 1.mkv"), force=False, model="parakeet")
    done = await jobs.wait(job["id"])
    listed = service.store.list_jobs()["items"][0]
    assert listed["task"] == "captions" and listed["filename"] == "Ep 1.mkv" and "result_text" not in listed
    assert listed["captions"]["outcome"] == "no_speech" and listed["captions"]["path"].endswith("Ep 1.mkv")
    assert done["captions"]["outcome"] == "no_speech" and done["result_available"] is False


async def test_single_and_batch_endpoints_validate_paths_and_report_each_one(service):
    async with client() as http:
        one = await http.post("/api/captions", json={"path": "/tv/Show/Ep 1.mkv"})
        assert one.status_code == 202 and one.json()["task"] == "captions" and one.json()["captions"]["force"] is False
        assert (await http.post("/api/captions", json={"path": "/etc/passwd"})).status_code == 422
        assert (await http.post("/api/captions", json={"path": "/tv/Show/missing.mkv"})).status_code == 422
        assert (await http.post("/api/captions", json={"path": "/tv/Show/Ep 1.mkv", "extra": 1})).status_code == 422
        batch = (await http.post("/api/captions/batch", json={"paths": ["/tv/Show/Ep 1.mkv", "/tv/Show/Ep 2.mkv", "/tv/Show/Ep 2.mkv", "/nope.mkv"], "force": True})).json()
        assert [item["path"].rsplit("/", 1)[-1] for item in batch["queued"]] == ["Ep 2.mkv"]
        assert [item["path"].rsplit("/", 1)[-1] for item in batch["already_queued"]] == ["Ep 1.mkv"]
        assert [item["path"] for item in batch["rejected"]] == ["/nope.mkv"]
        assert (await http.post("/api/captions/batch", json={"paths": []})).status_code == 422


def dub_streams(*languages, **extra):
    return [{"index": i, "codec_type": "audio", "codec_name": "aac", "tags": {"language": lang}, "disposition": {}, **extra}
            for i, lang in enumerate(languages, 1)]


@pytest.fixture
def probed(monkeypatch):
    """Replace ffprobe: ``probed.streams[file name]`` is its stream list or an exception to raise; ``probed.calls`` logs each probe."""
    state = type("Probed", (), {"streams": {}, "calls": []})()

    async def fake_probe(path, timeout=180):
        state.calls.append((path.name, timeout))
        result = state.streams[path.name]
        if isinstance(result, Exception):
            raise result
        return {"streams": result, "duration": 60.0}

    monkeypatch.setattr(media, "probe", fake_probe)
    return state


async def test_webhook_answers_tests_ignores_other_events_and_obeys_the_toggle(service, probed):
    probed.streams["Ep 1.mkv"] = dub_streams("jpn", "eng")
    async with client() as http:
        assert (await http.post("/api/captions/webhook", json={"eventType": "Test"})).json()["status"] == "ok"
        assert (await http.post("/api/captions/webhook", json={"eventType": "Grab"})).json()["status"] == "ignored"
        assert (await http.post("/api/captions/webhook", content=b"not json")).status_code == 400
        off = await http.post("/api/captions/webhook", json=sonarr("/tv/Show/Ep 1.mkv"))
        assert off.status_code == 200 and off.json()["status"] == "disabled" and service.store.list_jobs()["total"] == 0
        assert probed.calls == []  # nothing is probed while the toggle is off
        assert (await http.patch("/api/settings", json={"auto_captions": True})).json()["auto_captions"] is True
        for event in ("Download", "Upgrade"):
            reply = await http.post("/api/captions/webhook", json=sonarr("/tv/Show/Ep 1.mkv", event=event, isUpgrade=event == "Upgrade"))
            assert reply.status_code == 200 and reply.json()["status"] == "queued"
        listed = service.store.list_jobs()
        assert listed["total"] == 1  # the second event found the video already queued
        job = service.store.get_job(listed["items"][0]["id"], internal=True)
        assert job["source"] == "webhook" and job["request"]["force"] is True
        outside = await http.post("/api/captions/webhook", json=sonarr("/etc/passwd"))
        assert outside.json()["rejected"][0]["path"] == "/etc/passwd"
        assert [name for name, _ in probed.calls] == ["Ep 1.mkv", "Ep 1.mkv"]  # the outside path was never probed


async def test_webhook_queues_only_dubs_and_reports_the_rest(service, probed):
    probed.streams.update({
        "Ep 1.mkv": dub_streams("jpn", "eng"),
        "Ep 2.mkv": dub_streams("eng"),
        "Ep 3.mkv": media.MediaError("Could not read the video: ffprobe exited with code 1 and no message"),
    })
    service.store.update_settings({"auto_captions": True})
    async with client() as http:
        payload = {"eventType": "Download", "series": {"path": "/tv/Show", "originalLanguage": JAPANESE},
                   "episodeFiles": [{"relativePath": name} for name in ("Ep 1.mkv", "Ep 2.mkv", "Ep 3.mkv")]}
        reply = await http.post("/api/captions/webhook", json=payload)
    body = reply.json()
    assert reply.status_code == 200 and body["status"] == "queued"
    assert [item["path"].rsplit("/", 1)[-1] for item in body["queued"]] == ["Ep 1.mkv"] and body["already_queued"] == []
    assert [item["path"].rsplit("/", 1)[-1] for item in body["skipped"]] == ["Ep 2.mkv"]
    assert body["skipped"][0]["reason"] == "No English audio together with a non-English audio track (not a dub)"
    assert [(item["path"].rsplit("/", 1)[-1], item["error"]) for item in body["rejected"]] == \
        [("Ep 3.mkv", "Could not read the video: ffprobe exited with code 1 and no message")]
    assert all(timeout == 30 for _, timeout in probed.calls)
    assert service.store.list_jobs()["total"] == 1  # skipped and unreadable files were not queued
    assert app.state.ledger.listing(50, 0, None)["total"] == 0


def test_original_language_is_read_from_series_or_movie_and_placeholders_are_unknown():
    assert original_language({"series": {"originalLanguage": JAPANESE}}) == "Japanese"
    assert original_language({"movie": {"originalLanguage": {"id": 1, "name": "English"}}}) == "English"
    for payload in ({}, {"series": {}}, {"series": {"originalLanguage": None}}, {"series": {"originalLanguage": "Japanese"}},
                    {"series": {"originalLanguage": {"id": 0, "name": "Unknown"}}}, {"movie": {"originalLanguage": {"id": -1, "name": "Any"}}},
                    {"movie": {"originalLanguage": {"id": -2, "name": "Original"}}}, {"series": None}, {"series": {"originalLanguage": {"id": 8}}}):
        assert original_language(payload) is None


async def test_webhook_queues_a_foreign_original_and_skips_english_and_unknown_originals_without_probing(service, probed):
    probed.streams["Ep 1.mkv"] = dub_streams("jpn", "eng")
    probed.streams["Ep 2.mkv"] = dub_streams("eng", "por")
    service.store.update_settings({"auto_captions": True})
    async with client() as http:
        japanese = (await http.post("/api/captions/webhook", json=sonarr("/tv/Show/Ep 1.mkv"))).json()
        assert [item["path"].rsplit("/", 1)[-1] for item in japanese["queued"]] == ["Ep 1.mkv"] and japanese["skipped"] == []
        probed.calls.clear()
        english = (await http.post("/api/captions/webhook", json=sonarr("/tv/Show/Ep 2.mkv", language={"id": 1, "name": "English"}))).json()
        assert english["queued"] == [] and [item["reason"] for item in english["skipped"]] == [ENGLISH_ORIGINAL]
        for language in (None, {"id": 0, "name": "Unknown"}):
            unknown = (await http.post("/api/captions/webhook", json=sonarr("/tv/Show/Ep 2.mkv", language=language))).json()
            assert unknown["queued"] == [] and [item["reason"] for item in unknown["skipped"]] == [UNKNOWN_ORIGINAL]
        radarr = {"eventType": "Download", "movie": {"folderPath": "/tv/Show", "originalLanguage": {"id": 1, "name": "English"}}, "movieFile": {"relativePath": "Ep 2.mkv"}}
        assert (await http.post("/api/captions/webhook", json=radarr)).json()["skipped"][0]["reason"] == ENGLISH_ORIGINAL
    assert probed.calls == []  # a title-level skip needs no probe
    assert service.store.list_jobs()["total"] == 1
    assert app.state.ledger.listing(50, 0, None)["total"] == 0


async def test_manual_requests_ignore_the_original_language_filter(service, probed):
    service.store.update_settings({"auto_captions": True})
    async with client() as http:
        skipped = await http.post("/api/captions/webhook", json=sonarr("/tv/Show/Ep 2.mkv", language={"id": 1, "name": "English"}))
        assert skipped.json()["queued"] == [] and skipped.json()["skipped"]
        assert (await http.post("/api/captions", json={"path": "/tv/Show/Ep 2.mkv"})).status_code == 202
        batch = await http.post("/api/captions/batch", json={"paths": ["/tv/Show/Ep 3.mkv"]})
        assert [item["path"].rsplit("/", 1)[-1] for item in batch.json()["queued"]] == ["Ep 3.mkv"] and "skipped" not in batch.json()
    assert probed.calls == []


async def test_a_skipped_import_can_still_be_captioned_by_hand_and_manual_requests_are_not_filtered(service, probed):
    probed.streams["Ep 2.mkv"] = dub_streams("eng")
    service.store.update_settings({"auto_captions": True})
    async with client() as http:
        skipped = await http.post("/api/captions/webhook", json=sonarr("/tv/Show/Ep 2.mkv"))
        assert skipped.json()["skipped"] and skipped.json()["queued"] == []
        manual = await http.post("/api/captions", json={"path": "/tv/Show/Ep 2.mkv"})
        assert manual.status_code == 202 and manual.json()["captions"]["force"] is False
        batch = await http.post("/api/captions/batch", json={"paths": ["/tv/Show/Ep 1.mkv"]})
        assert [item["path"].rsplit("/", 1)[-1] for item in batch.json()["queued"]] == ["Ep 1.mkv"]
        assert "skipped" not in batch.json()
    assert [name for name, _ in probed.calls] == ["Ep 2.mkv"]  # only the webhook probes


async def test_auto_captions_is_off_until_enabled_and_must_be_a_boolean(service):
    async with client() as http:
        assert (await http.get("/api/settings")).json()["auto_captions"] is False
        assert (await http.patch("/api/settings", json={"auto_captions": "yes"})).status_code == 422
        assert (await http.patch("/api/settings", json={"auto_captions": None})).status_code == 422


async def test_the_decision_ledger_is_listed_and_filterable(service):
    ledger = app.state.ledger
    video = service.media / "Show" / "Ep 1.mkv"
    ledger.record(video, "matched_existing", {"outcome": "matched_existing", "matched": {"label": "embedded track 4"}})
    async with client() as http:
        everything = (await http.get("/api/captions")).json()
        assert everything["total"] == 1 and everything["items"][0]["details"]["matched"]["label"] == "embedded track 4"
        assert (await http.get("/api/captions", params={"outcome": "written"})).json()["total"] == 0
        assert (await http.get("/api/captions", params={"path": str(video)})).json()["items"][0]["outcome"] == "matched_existing"
        assert (await http.get("/api/captions", params={"path": "/none"})).json() == {"items": [], "total": 0}
