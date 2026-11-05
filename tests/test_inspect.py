import httpx
import pytest

from speechdock import inspect
from speechdock.app import app
from speechdock.audio import RATE, _pcm_wav
from speechdock.captions import Ledger, media
from speechdock.captions.media import MediaError
from speechdock.engine import EngineError, EngineUnavailable
from speechdock.inspect import InspectError, detect_track, language_code, parse_detect, run_inspection, wants_detection
from speechdock.jobs import JobManager
from speechdock.store import Store


def test_container_tags_map_to_whisper_codes():
    assert language_code("jpn") == "ja"
    assert language_code("eng") == "en"
    assert language_code("en-US") == "en"
    assert language_code("Japanese") == "ja"
    assert language_code("ger") == language_code("deu") == "de"
    assert language_code("und") == language_code("") == language_code("zxx") == ""
    assert language_code("tlh") == "tlh"  # unknown tags are passed through, never guessed


def test_detect_modes_and_language_lists_are_validated():
    assert parse_detect("untagged") == "untagged"
    assert parse_detect(["jpn", "en", "ja"]) == ["en", "ja"]
    for bad in ("maybe", [], ["tlh"], 3, ["en", 1]):
        with pytest.raises(InspectError):
            parse_detect(bad)


def test_which_tracks_are_detected():
    japanese = {"language": "ja", "commentary": False}
    untagged = {"language": "", "commentary": False}
    commentary = {"language": "", "commentary": True}
    assert [wants_detection(t, "untagged") for t in (japanese, untagged, commentary)] == [False, True, False]
    assert [wants_detection(t, "all") for t in (japanese, untagged, commentary)] == [True, True, False]
    assert [wants_detection(t, "none") for t in (japanese, untagged, commentary)] == [False, False, False]
    assert [wants_detection(t, ["ja"]) for t in (japanese, untagged, {"language": "pt", "commentary": False})] == [True, True, False]


def wav(seconds: float, speech: bool) -> bytes:
    """A mono 16 kHz WAV: a loud square wave the detector takes for speech, or digital silence."""
    frames = int(seconds * RATE)
    pcm = b"".join((b"\x00\x40" if (i // 160) % 2 else b"\x00\xc0") for i in range(frames)) if speech else bytes(frames * 2)
    return _pcm_wav(pcm)


class Detector:
    """Stands in for Silero: one 12-second speech span at 30 % of the region, or none at all."""

    def __init__(self, speech: bool):
        self.speech = speech

    def probabilities(self, pcm):
        frames = len(pcm) // 2
        start, end = frames * 3 // 10, frames * 3 // 10 + 12 * RATE
        for position in range(0, frames, 512):
            yield 0.9 if (self.speech and start <= position < end) else 0.0


class FakeEngine:
    def __init__(self, language="ja", probability=0.93, error=None):
        self.language, self.probability, self.error, self.calls = language, probability, error, []

    async def detect(self, wav_bytes, model):
        self.calls.append(len(wav_bytes))
        if self.error:
            raise self.error
        return {"language_code": self.language, "detected_language": "japanese", "probability": self.probability}


async def test_a_track_is_sampled_from_the_middle_and_the_longest_speech_window_is_detected(monkeypatch):
    regions = []

    async def read_region(path, index, start, seconds=inspect.REGION_SECONDS):
        regions.append(start)
        return wav(seconds, speech=start == 1380.0)  # speech only in the region centred on the middle

    monkeypatch.setattr(inspect, "read_region", read_region)
    engine = FakeEngine()
    result = await detect_track("/x.mkv", {"index": 1}, 2880.0, engine, "whisper", Detector(speech=True))
    assert regions == [1380.0]  # 50 % of 2880 s minus half the 120 s region
    assert result["language"] == "ja" and result["probability"] == 0.93 and result["name"] == "japanese"
    assert 1380 <= result["sample_start"] <= 1500 and 10 <= result["sample_seconds"] <= 28
    assert len(engine.calls) == 1


async def test_a_silent_middle_tries_30_and_70_percent_then_gives_up(monkeypatch):
    regions = []

    async def read_region(path, index, start, seconds=inspect.REGION_SECONDS):
        regions.append(start)
        return wav(seconds, speech=False)

    monkeypatch.setattr(inspect, "read_region", read_region)
    with pytest.raises(MediaError, match="No speech found at 50 %, 30 %, 70 %"):
        await detect_track("/x.mkv", {"index": 1}, 1000.0, FakeEngine(), "whisper", Detector(speech=False))


async def test_a_short_track_is_read_once_from_the_start(monkeypatch):
    regions = []

    async def read_region(path, index, start, seconds=inspect.REGION_SECONDS):
        regions.append(start)
        return wav(90, speech=True)

    monkeypatch.setattr(inspect, "read_region", read_region)
    result = await detect_track("/x.mkv", {"index": 2}, 90.0, FakeEngine("en"), "whisper", Detector(speech=True))
    assert regions == [0.0] and result["language"] == "en"


STREAMS = [
    {"index": 0, "codec_type": "video", "codec_name": "hevc", "width": 1920, "height": 1080, "disposition": {"default": 1}},
    {"index": 1, "codec_type": "audio", "codec_name": "aac", "channels": 2, "tags": {"language": "jpn", "title": "Japanese"}, "disposition": {"default": 1}},
    {"index": 2, "codec_type": "audio", "codec_name": "eac3", "channels": 6, "tags": {"language": "und"}, "disposition": {}},
    {"index": 3, "codec_type": "audio", "codec_name": "aac", "channels": 2, "tags": {"language": "eng", "title": "Director's commentary"}, "disposition": {}},
    {"index": 4, "codec_type": "subtitle", "codec_name": "ass", "tags": {"language": "eng"}, "disposition": {"forced": 1}},
]


async def inspection(monkeypatch, detect, engine, probe_error=None, detect_fn=None):
    async def probe(path, timeout=180):
        if probe_error:
            raise probe_error
        return {"streams": STREAMS, "duration": 1420.0}

    async def fake_detect(path, track, duration, eng, model, detector):
        if track["index"] == 2:  # the untagged track: Portuguese, found without the engine
            return {"language": "pt", "name": "portuguese", "probability": 0.88, "sample_start": 700.0, "sample_seconds": 20.0}
        await eng.detect(b"", model)
        return {"language": "ja", "name": "japanese", "probability": 0.95, "sample_start": 650.0, "sample_seconds": 25.0}

    monkeypatch.setattr(media, "probe", probe)
    monkeypatch.setattr(inspect, "detect_track", detect_fn or fake_detect)
    monkeypatch.setattr(inspect, "SpeechDetector", lambda path: object())
    progress = []

    async def record(value, duration=None):
        progress.append(value)

    async def engine_for_detection():
        return engine

    job = {"request": {"path": "/downloads/x.mkv", "detect": detect}, "model": "whisper"}
    result = await run_inspection(job=job, progress=record, engine_for_detection=engine_for_detection, model="whisper", vad_model="vad.onnx")
    return result["result_json"], progress


async def test_the_report_lists_every_stream_and_detects_only_what_was_asked(monkeypatch):
    report, progress = await inspection(monkeypatch, "untagged", FakeEngine())
    assert [v["codec"] for v in report["video"]] == ["hevc"] and report["video"][0]["width"] == 1920
    assert [s["language"] for s in report["subtitles"]] == ["en"] and report["subtitles"][0]["forced"] is True
    tags = [(t["index"], t["tag"], t["language"], t["commentary"], t["default"]) for t in report["audio"]]
    assert tags == [(1, "jpn", "ja", False, True), (2, "und", "", False, False), (3, "eng", "en", True, False)]
    assert report["audio"][0]["detected"] is None and report["audio"][2]["detected"] is None
    assert report["audio"][1]["detected"]["language"] == "pt"
    assert report["duration"] == 1420.0 and progress == [0.0, 1.0]

    report, _ = await inspection(monkeypatch, "all", FakeEngine())
    assert [t["detected"]["language"] if t["detected"] else None for t in report["audio"]] == ["ja", "pt", None]  # commentary is never detected

    report, _ = await inspection(monkeypatch, "none", FakeEngine())
    assert all(t["detected"] is None and "detection_error" not in t for t in report["audio"])


async def test_detection_failures_are_reported_per_track_and_an_absent_engine_is_not_waited_for_twice(monkeypatch):
    engine = FakeEngine(error=EngineError("Speech engine returned HTTP 500"))
    report, _ = await inspection(monkeypatch, "all", engine)
    assert report["audio"][0]["detected"] is None and report["audio"][0]["detection_error"] == "Speech engine returned HTTP 500"
    assert report["audio"][1]["detected"]["language"] == "pt"  # the next track was still tried

    async def unavailable(path, track, duration, eng, model, detector):
        raise EngineUnavailable("The speech engine did not come back")

    report, _ = await inspection(monkeypatch, "all", FakeEngine(), detect_fn=unavailable)
    assert [t.get("detection_error") for t in report["audio"]] == ["The speech engine did not come back"] * 2 + [None]


async def test_an_unreadable_file_fails_the_job(monkeypatch):
    with pytest.raises(MediaError, match="Could not read"):
        await inspection(monkeypatch, "none", FakeEngine(), probe_error=MediaError("Could not read the video: ffprobe exited with code 1"))


class Models:
    async def list_models(self):
        return [{"model_id": "whisper", "registered": True, "translation_capable": True, "language_detection_capable": True, "runtime": "whisper.cpp", "architecture": "whisper"}]


@pytest.fixture
async def service(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    downloads = tmp_path / "downloads"
    (downloads / "Show.S01").mkdir(parents=True)
    (downloads / "Show.S01" / "Show.S01E01.mkv").write_bytes(b"x")
    (downloads / "notes.txt").write_bytes(b"x")
    store.update_settings({"detection_model": "whisper"})
    seen = []

    async def processor(job, path, progress):
        seen.append(job["request"])
        return {"result_json": {"path": job["request"]["path"], "duration": 1.0, "detect": job["request"]["detect"], "video": [], "audio": [], "subtitles": []}, "duration_seconds": 1.0}

    jobs = JobManager(store, processor)
    monkeypatch.setenv("SPDOCK_INSPECT_ROOTS", '{"/downloads": "%s"}' % downloads)
    monkeypatch.setenv("SPDOCK_CAPTION_ROOTS", '{"/tv": "%s"}' % (tmp_path / "tv"))
    for name, value in (("store", store), ("models", Models()), ("jobs", jobs), ("ledger", Ledger(store.db))):
        monkeypatch.setattr(app.state, name, value, raising=False)
    await jobs.start()
    yield type("Service", (), {"store": store, "jobs": jobs, "seen": seen, "downloads": downloads})()
    await jobs.close()
    store.close()


def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_inspect_waits_for_the_report_and_shows_the_job_in_activity(service):
    async with client() as http:
        response = await http.post("/api/inspect", json={"path": "/downloads/Show.S01/Show.S01E01.mkv", "detect": ["jpn"], "source": "nzbget"})
        assert response.status_code == 200, response.text
        report = response.json()
        assert report["path"] == str(service.downloads / "Show.S01" / "Show.S01E01.mkv") and report["detect"] == ["ja"]
        job = (await http.get("/api/jobs")).json()["items"][0]
        assert job["task"] == "inspect" and job["source"] == "nzbget" and job["status"] == "completed" and job["filename"] == "Show.S01E01.mkv"
        assert job["result_available"] is False and job["note"] is None
        detail = (await http.get(f"/api/jobs/{report['job_id']}")).json()
        assert detail["inspection"]["detect"] == ["ja"]

        noted = await http.patch(f"/api/inspect/{report['job_id']}", json={"note": "Rejected: no Japanese audio"})
        assert noted.status_code == 200 and noted.json()["note"] == "Rejected: no Japanese audio"
        assert (await http.get("/api/jobs")).json()["items"][0]["note"] == "Rejected: no Japanese audio"
        assert (await http.patch("/api/inspect/00000000-0000-0000-0000-000000000000", json={"note": "x"})).status_code == 404


async def test_inspect_rejects_paths_outside_the_mounts_and_bad_requests(service):
    async with client() as http:
        for path, expected in (("/downloads/../etc/passwd", "inspectable folders"), ("/downloads/notes.txt", "Not a supported video"),
                               ("/downloads/Show.S01/missing.mkv", "not found"), ("relative.mkv", "absolute")):
            response = await http.post("/api/inspect", json={"path": path})
            assert response.status_code == 422 and expected in response.json()["detail"], (path, response.text)
        assert (await http.post("/api/inspect", json={"path": "/downloads/Show.S01/Show.S01E01.mkv", "detect": "sometimes"})).status_code == 422
        assert (await http.post("/api/inspect", json={"path": "/downloads/Show.S01/Show.S01E01.mkv", "source": "bad source!"})).status_code == 422
        assert (await http.post("/api/inspect", json={"path": "/downloads/Show.S01/Show.S01E01.mkv", "extra": 1})).status_code == 422
    assert service.seen == []


async def test_library_paths_are_inspectable_too(service, tmp_path):
    (tmp_path / "tv" / "Show").mkdir(parents=True)
    (tmp_path / "tv" / "Show" / "Ep.mkv").write_bytes(b"x")
    async with client() as http:
        response = await http.post("/api/inspect", json={"path": "/tv/Show/Ep.mkv", "detect": "none"})
        assert response.status_code == 200 and response.json()["path"] == str(tmp_path / "tv" / "Show" / "Ep.mkv")


async def test_a_failed_inspection_answers_502_with_the_reason(service, monkeypatch):
    async def failing(job, path, progress):
        raise MediaError("Could not read the video: ffprobe exited with code 1")

    service.jobs.processor = failing
    async with client() as http:
        response = await http.post("/api/inspect", json={"path": "/downloads/Show.S01/Show.S01E01.mkv"})
        assert response.status_code == 502 and "Could not read the video" in response.json()["detail"]
