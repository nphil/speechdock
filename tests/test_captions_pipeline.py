import random
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from speechdock.audio import parse_srt
from speechdock.captions import CaptionError, Ledger, run_caption_job
from speechdock.captions.ledger import digest, publish, sidecar_for
from speechdock.engine import Word
from speechdock.store import Store

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg is required")

VOCABULARY = ("river mountain castle shadow winter silver thunder garden window letter bridge candle forest "
              "harbor engine hammer ladder pocket marble velvet copper meadow lantern compass feather orchard").split()
GLUE = "the a to of and in is was we they you he she it that this for with".split()


def make_sentences(seed=5, count=14):
    rng = random.Random(seed)
    return [[rng.choice(GLUE if rng.random() < 0.4 else VOCABULARY) for _ in range(rng.randint(6, 9))] for _ in range(count)]


SENTENCES = make_sentences()


def timeline(sentences):
    clock, result = 0.5, []
    for words in sentences:
        end = clock + 0.22 * len(words)
        result.append((clock, end, words))
        clock = end + 0.3
    return result


def spoken(sentences=SENTENCES, confidence=1.0):
    return [Word(start + 0.22 * i, start + 0.22 * i + 0.2, token, confidence)
            for start, _, words in timeline(sentences) for i, token in enumerate(words)]


def srt(sentences):
    def stamp(seconds):
        ms = round(seconds * 1000)
        return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"
    return "\n".join(f"{n}\n{stamp(s)} --> {stamp(e)}\n{' '.join(w)}\n" for n, (s, e, w) in enumerate(timeline(sentences), 1))


def translation(sentences):
    rng = random.Random(11)
    return [[rng.choice(VOCABULARY) if rng.random() < 0.55 else token for token in words] for words in sentences]


class StubEngine:
    """Returns a fixed transcript for the first window and nothing after, like a short file."""

    def __init__(self, words=None):
        self.words, self.calls = words if words is not None else spoken(), 0

    async def transcribe_words(self, wav, model, language):
        self.calls += 1
        return (self.words if self.calls == 1 else []), 0


def make_video(path: Path, audio=(("eng", 1, {}),), subs=()):
    """Matroska with sine audio tracks (language, default, extra tags) and embedded SRT tracks (title, text)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    for _ in audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=20"]
    for number, (_, text) in enumerate(subs):
        (path.parent / f"_sub{number}.srt").write_text(text)
        command += ["-i", str(path.parent / f"_sub{number}.srt")]
    command += [arg for number in range(len(audio) + len(subs)) for arg in ("-map", str(number))]
    command += ["-c:a", "aac", "-c:s", "srt"]
    for number, (language, default, extra) in enumerate(audio):
        command += [f"-metadata:s:a:{number}", f"language={language}", f"-disposition:a:{number}", "default" if default else "0"]
        if "title" in extra:
            command += [f"-metadata:s:a:{number}", f"title={extra['title']}"]
        if extra.get("comment"):
            command += [f"-disposition:a:{number}", "comment"]
    for number, (title, _) in enumerate(subs):
        command += [f"-metadata:s:s:{number}", "language=eng", f"-metadata:s:s:{number}", f"title={title}"]
    subprocess.run([*command, str(path)], check=True)
    for number in range(len(subs)):
        (path.parent / f"_sub{number}.srt").unlink()
    return path


@pytest.fixture
def library(tmp_path):
    store = Store(tmp_path / "data")
    yield type("Library", (), {"root": tmp_path / "tv", "work": tmp_path / "work", "ledger": Ledger(store.db), "store": store})()
    store.close()


async def noop(progress, duration=None):
    return None


async def run(library, name, engine=None, force=False, path=None):
    engine = engine or StubEngine()
    job = {"id": str(uuid.uuid4()), "model": "parakeet", "request": {"path": path or f"/tv/Show/{name}", "force": force}}
    result = await run_caption_job(job=job, progress=noop, engine=engine, ledger=library.ledger, workdir=library.work,
                                   roots={"/tv": str(library.root)})
    return result["result_json"], result, engine


async def test_a_caption_file_is_written_beside_the_video_and_owned(library):
    video = make_video(library.root / "Show" / "Ep 1.mkv")
    report, result, engine = await run(library, "Ep 1.mkv")
    sidecar = sidecar_for(video)
    assert report["outcome"] == "written" and sidecar.name == "Ep 1.en.cc.srt" and sidecar.is_file()
    cues = parse_srt(sidecar.read_text())
    assert " ".join(c.text.replace("\n", " ") for c in cues) == " ".join(w.text for w in spoken())
    assert result["result_text"] == sidecar.read_text() and report["output"]["sha256"] == digest(sidecar)
    assert library.ledger.is_owned(sidecar) and engine.calls == 1
    assert [p.name for p in sidecar.parent.iterdir() if p.name.startswith(".")] == []  # no temp files left behind
    assert not list(library.work.glob("*/*"))  # scratch audio is removed


async def test_the_audio_stream_is_the_english_one_not_the_default_or_commentary(library):
    make_video(library.root / "Show" / "Ep 2.mkv", audio=(("jpn", 1, {}), ("eng", 0, {"title": "Director's Commentary", "comment": True}), ("eng", 0, {})))
    report, *_ = await run(library, "Ep 2.mkv")
    assert report["audio"]["index"] == 2 and report["audio"]["language"] == "eng"


async def test_a_matching_embedded_dub_caption_means_nothing_is_written(library):
    video = make_video(library.root / "Show" / "Ep 3.mkv", subs=(("Signs", srt(SENTENCES[:1])), ("Dub (SDH)", srt(SENTENCES)), ("English", srt(translation(SENTENCES)))))
    report, result, _ = await run(library, "Ep 3.mkv")
    assert report["outcome"] == "matched_existing" and not sidecar_for(video).exists()
    assert report["matched"]["title"] == "Dub (SDH)" and report["matched"]["agreement"] > 0.95
    scores = {c["title"]: c["agreement"] for c in report["candidates"]}
    assert scores["English"] < 0.7 and scores["Signs"] < 0.2
    assert library.ledger.get(video)["outcome"] == "matched_existing"
    assert result["result_text"]  # the transcript is kept for review, but not written to the library


async def test_a_matching_sidecar_counts_even_when_someone_else_named_it_cc(library):
    video = make_video(library.root / "Show" / "Ep 4.mkv")
    other = sidecar_for(video)
    other.write_text(srt(SENTENCES))
    report, *_ = await run(library, "Ep 4.mkv")
    assert report["outcome"] == "matched_existing" and report["matched"]["file"] == other.name
    assert other.read_text() == srt(SENTENCES)


async def test_a_translation_does_not_stop_the_caption_and_is_not_overwritten(library):
    video = make_video(library.root / "Show" / "Ep 5.mkv", subs=(("English", srt(translation(SENTENCES))),))
    (video.parent / "Ep 5.en.srt").write_text(srt(translation(SENTENCES)))
    report, *_ = await run(library, "Ep 5.mkv")
    assert report["outcome"] == "written" and report["matched"] is None
    assert {c["source"] for c in report["candidates"]} == {"embedded", "sidecar"}
    assert (video.parent / "Ep 5.en.srt").read_text() == srt(translation(SENTENCES))


async def test_a_file_without_english_audio_is_skipped_without_transcribing(library):
    video = make_video(library.root / "Show" / "Ep 6.mkv", audio=(("jpn", 1, {}),))
    report, _, engine = await run(library, "Ep 6.mkv")
    assert report["outcome"] == "no_english_audio" and engine.calls == 0 and not sidecar_for(video).exists()


async def test_music_only_audio_is_a_skip_not_a_caption(library):
    video = make_video(library.root / "Show" / "Ep 7.mkv")
    report, *_ = await run(library, "Ep 7.mkv", engine=StubEngine(spoken(SENTENCES[:1])))
    assert report["outcome"] == "no_speech" and not sidecar_for(video).exists()


async def test_a_file_that_is_not_ours_is_never_overwritten_even_when_forced(library):
    video = make_video(library.root / "Show" / "Ep 8.mkv")
    foreign = sidecar_for(video)
    foreign.write_text("1\n00:00:01,000 --> 00:00:02,000\nsomeone else's captions\n")
    before = foreign.read_bytes()
    for force in (False, True):
        with pytest.raises(CaptionError, match="not created by SpeechDock"):
            await run(library, "Ep 8.mkv", force=force)
        assert foreign.read_bytes() == before
    assert not library.ledger.is_owned(foreign)


async def test_a_rerun_keeps_our_file_unless_forced_and_then_replaces_only_it(library):
    video = make_video(library.root / "Show" / "Ep 9.mkv")
    first, *_ = await run(library, "Ep 9.mkv")
    sidecar = sidecar_for(video)
    original = sidecar.read_bytes()
    report, result, engine = await run(library, "Ep 9.mkv")
    assert report["outcome"] == "already_generated" and report["cached"] and engine.calls == 0 and sidecar.read_bytes() == original
    changed = spoken()[:-1]
    forced, _, _ = await run(library, "Ep 9.mkv", engine=StubEngine(changed), force=True)
    assert forced["outcome"] == "written" and forced["output"]["replaced"] is True
    assert sidecar.read_bytes() != original and library.ledger.is_owned(sidecar)


async def test_a_file_we_wrote_but_a_person_replaced_is_theirs_and_never_overwritten(library):
    video = make_video(library.root / "Show" / "Ep 10.mkv")
    await run(library, "Ep 10.mkv")
    sidecar = sidecar_for(video)
    sidecar.write_text("1\n00:00:01,000 --> 00:00:02,000\nhand written captions\n")
    assert not library.ledger.is_owned(sidecar)
    replaced = sidecar.read_bytes()
    with pytest.raises(CaptionError, match="not created by SpeechDock"):
        await run(library, "Ep 10.mkv", force=True)
    assert sidecar.read_bytes() == replaced


async def test_a_lightly_edited_copy_of_our_file_is_a_real_caption_and_is_left_alone(library):
    video = make_video(library.root / "Show" / "Ep 10b.mkv")
    await run(library, "Ep 10b.mkv")
    sidecar = sidecar_for(video)
    sidecar.write_text(sidecar.read_text() + "\n1000\n00:19:00,000 --> 00:19:01,000\nhand edit\n")
    edited = sidecar.read_bytes()
    report, *_ = await run(library, "Ep 10b.mkv", force=True)
    assert report["outcome"] == "matched_existing" and report["matched"]["file"] == sidecar.name
    assert sidecar.read_bytes() == edited


async def test_a_skip_decision_is_reused_until_the_video_changes(library):
    video = make_video(library.root / "Show" / "Ep 11.mkv", audio=(("jpn", 1, {}),))
    await run(library, "Ep 11.mkv")
    again, _, engine = await run(library, "Ep 11.mkv")
    assert again["outcome"] == "no_english_audio" and again["cached"] is True
    make_video(video, audio=(("eng", 1, {}),))  # the video is replaced: the old decision no longer applies
    replaced, *_ = await run(library, "Ep 11.mkv")
    assert replaced["outcome"] == "written" and not replaced.get("cached")


async def test_arr_paths_map_into_the_mounts_and_nothing_else_is_accepted(library, tmp_path):
    make_video(library.root / "Show" / "Ep 12.mkv")
    outside = tmp_path / "elsewhere.mkv"
    outside.touch()
    (library.root / "link.mkv").symlink_to(outside)
    (library.root / "notes.txt").write_text("x")
    for bad in (str(outside), "/tv/../elsewhere.mkv", "/tv/link.mkv", "/tv/notes.txt", "/tv/Show/missing.mkv", "relative.mkv", ""):
        with pytest.raises(CaptionError):
            await run(library, "x", path=bad)
    report, *_ = await run(library, "Ep 12.mkv", path=str(library.root / "Show" / "Ep 12.mkv"))  # already a local path
    assert report["outcome"] == "written"


def test_publish_never_replaces_without_permission_and_leaves_no_temp_files(tmp_path):
    target = tmp_path / "a.en.cc.srt"
    publish(target, b"first", replace=False)
    with pytest.raises(FileExistsError):
        publish(target, b"second", replace=False)
    assert target.read_bytes() == b"first"
    publish(target, b"third", replace=True)
    assert target.read_bytes() == b"third"
    assert [p.name for p in tmp_path.iterdir()] == ["a.en.cc.srt"]


def test_ledger_ownership_needs_the_exact_bytes_we_wrote(tmp_path, library):
    video = tmp_path / "v.mkv"
    video.write_text("video")
    sidecar = sidecar_for(video)
    sidecar.write_text("ours")
    assert not library.ledger.is_owned(sidecar)  # never recorded
    library.ledger.record(video, "written", {}, sidecar=sidecar, sha256=digest(sidecar))
    assert library.ledger.is_owned(sidecar)
    library.ledger.record(video, "matched_existing", {})  # a later decision keeps the ownership of the file we wrote
    assert library.ledger.is_owned(sidecar)
    sidecar.write_text("edited")
    assert not library.ledger.is_owned(sidecar)
    sidecar.unlink()
    assert not library.ledger.is_owned(sidecar)
