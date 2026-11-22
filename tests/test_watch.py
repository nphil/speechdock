"""The scheduled look (what it queues, how it decides anime, when it alerts) and ntfy alert limits."""
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from speechdock import notify, watch

REAL_CLIENT = httpx.AsyncClient  # the fakes patch httpx itself, so keep the real class


class FakeStore:
    def __init__(self):
        self.settings = {"auto_tidy": True}

    def get_settings(self):
        return dict(self.settings)

    def update_settings(self, values):
        self.settings.update(values)
        return self.get_settings()


class FakeNotifier:
    def __init__(self):
        self.sent = []

    async def problem(self, key, title, message):
        self.sent.append(("problem", key))
        return True

    async def resolved(self, key, title, message):
        self.sent.append(("resolved", key))
        return True


@pytest.fixture
def library(tmp_path, monkeypatch):
    """Media mounts with one anime episode, one show episode and one film, plus *arr/Bazarr keys."""
    tv, movies = tmp_path / "tv", tmp_path / "movies"
    for path in (tv / "Anime" / "Season 1" / "Anime - S01E01.mkv", tv / "Show" / "Season 1" / "Show - S01E01.mkv",
                 movies / "Film (2001)" / "Film (2001).mkv"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    data = tmp_path / "data"
    data.mkdir()
    for name in ("sonarr", "radarr", "bazarr"):
        (data / f"{name}-key").write_text("key")
        monkeypatch.setenv(f"SPDOCK_{name.upper()}_URL", f"http://{name}")
    monkeypatch.setenv("SPDOCK_DATA_DIR", str(data))
    monkeypatch.setenv("SPDOCK_CAPTION_ROOTS", f'{{"/tv": "{tv}", "/movies": "{movies}"}}')
    return {"tv": tv, "movies": movies}


def serve(monkeypatch, routes, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append((request.url.host, request.url.path, dict(request.url.params)))
        result = routes.get((request.url.host, request.url.path))
        if isinstance(result, Exception):
            raise result
        if result is None:
            return httpx.Response(404)
        return httpx.Response(200, json=result)
    monkeypatch.setattr(watch.httpx, "AsyncClient", lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler)))


ROUTES = {
    ("sonarr", "/api/v3/series"): [{"path": "/tv/Anime", "seriesType": "anime"}, {"path": "/tv/Show", "seriesType": "standard"}],
    ("radarr", "/api/v3/movie"): [{"path": "/movies/Film (2001)", "genres": ["Drama"], "originalLanguage": {"name": "Korean"}}],
    ("sonarr", "/api/v3/history/since"): [{"data": {"importedPath": "/tv/Anime/Season 1/Anime - S01E01.mkv"}}],
    ("radarr", "/api/v3/history/since"): [{"data": {"importedPath": "/movies/Film (2001)/Film (2001).mkv"}}],
    ("bazarr", "/api/episodes/history"): {"data": [{"subtitles_path": "/tv/Show/Season 1/Show - S01E01.en.srt",
                                                    "timestamp": datetime.now(timezone.utc).isoformat()}], "total": 1},
    ("bazarr", "/api/movies/history"): {"data": [], "total": 0},
}


async def test_new_imports_and_new_subtitles_are_queued_with_the_right_anime_decision(library, monkeypatch):
    serve(monkeypatch, ROUTES)
    queued = []

    async def submit(path, anime):
        queued.append((path, anime))
        return True

    store = FakeStore()
    result = await watch.Watcher(store, FakeNotifier(), submit).look()
    assert sorted(queued) == sorted([
        (str(library["tv"] / "Anime" / "Season 1" / "Anime - S01E01.mkv"), True),
        (str(library["movies"] / "Film (2001)" / "Film (2001).mkv"), False),
        (str(library["tv"] / "Show" / "Season 1" / "Show - S01E01.mkv"), False),  # found from its subtitle's name
    ])
    assert result["queued"] == 3 and result["failed"] == {}
    assert set(store.settings["tidy_checkpoints"]) == {"sonarr", "radarr", "bazarr"}


async def test_the_next_look_asks_only_for_changes_since_the_last_one(library, monkeypatch):
    seen = []
    serve(monkeypatch, ROUTES, seen)

    async def submit(path, anime):
        return True

    store = FakeStore()
    earlier = datetime.now(timezone.utc) - timedelta(minutes=30)
    store.settings["tidy_checkpoints"] = {"sonarr": earlier.isoformat()}
    await watch.Watcher(store, FakeNotifier(), submit).look()
    asked = next(params for host, path, params in seen if host == "sonarr" and path.endswith("/since"))
    assert asked["date"] == earlier.strftime("%Y-%m-%dT%H:%M:%SZ")


async def test_without_the_anime_list_nothing_is_queued_and_no_checkpoint_moves(library, monkeypatch):
    serve(monkeypatch, {**ROUTES, ("sonarr", "/api/v3/series"): httpx.ConnectError("down")})
    queued = []

    async def submit(path, anime):
        queued.append(path)
        return True

    store = FakeStore()
    result = await watch.Watcher(store, FakeNotifier(), submit).look()
    assert queued == [] and "tidy_checkpoints" not in store.settings and "anime list" in result["failed"]


async def test_a_source_that_keeps_failing_alerts_once_and_again_when_it_recovers(library, monkeypatch):
    monkeypatch.setattr(watch, "ALERT_AFTER_SECONDS", 0)
    notifier = FakeNotifier()

    async def submit(path, anime):
        return True

    watcher = watch.Watcher(FakeStore(), notifier, submit)
    serve(monkeypatch, {**ROUTES, ("bazarr", "/api/episodes/history"): httpx.ConnectError("down")})
    await watcher.look()
    assert notifier.sent == [("problem", "source:bazarr")]
    serve(monkeypatch, ROUTES)
    await watcher.look()
    assert notifier.sent[-1] == ("resolved", "source:bazarr")


async def test_alerts_repeat_at_most_every_few_hours_and_count_what_was_held_back(tmp_path, monkeypatch):
    token = tmp_path / "token"
    token.write_text("tk_test")
    sent = []

    async def fake_send(self, title, message, **kw):
        sent.append((title, message))
        return True

    monkeypatch.setattr(notify.Notifier, "send", fake_send)
    n = notify.Notifier("https://ntfy.example", "homelabber", token)
    assert await n.problem("job:tidy", "Subtitle tidy-up failed", "a")
    assert not await n.problem("job:tidy", "Subtitle tidy-up failed", "b")
    assert not await n.resolved("job:other", "x", "y")  # never reported, so nothing to resolve
    monkeypatch.setattr(notify, "REPEAT_SECONDS", 0)
    await n.problem("job:tidy", "Subtitle tidy-up failed", "c")
    assert sent[-1][1].endswith("(1 more like this since the last alert.)") and len(sent) == 2


async def test_alerts_go_to_the_topic_with_the_token(tmp_path, monkeypatch):
    token = tmp_path / "token"
    token.write_text("tk_test\n")
    requests = []
    real = REAL_CLIENT

    def handler(request):
        requests.append(request)
        return httpx.Response(200)
    monkeypatch.setattr(notify.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler)))
    assert await notify.Notifier("https://ntfy.example/", "homelabber", token).send("Test", "body")
    assert str(requests[0].url) == "https://ntfy.example/homelabber"
    assert requests[0].headers["Authorization"] == "Bearer tk_test" and requests[0].content == b"body"
    assert not await notify.Notifier("", "homelabber", token).send("Test", "body")  # not configured: nothing sent
