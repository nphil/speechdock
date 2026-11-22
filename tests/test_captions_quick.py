import random
import shutil
import struct
import subprocess
import uuid
from pathlib import Path

import pytest

from speechdock.audio import RATE
from speechdock.captions import Ledger, quick, run_caption_job
from speechdock.captions.ledger import sidecar_for
from speechdock.captions.media import extract_clip
from speechdock.captions.subtitles import SubCue, parse_subtitle
from speechdock.engine import Word
from speechdock.store import Store

VOCABULARY = ("river mountain castle shadow winter silver thunder garden window letter bridge candle forest "
              "harbor engine hammer ladder pocket marble velvet copper meadow lantern compass feather orchard "
              "tunnel anchor saddle pepper violin cobble ribbon cinder walnut thimble kettle trellis").split()
GLUE = "the a to of and in is was we they you he she it that this for with on at".split()
LENGTH = 1200.0  # a 20-minute episode
CLIPS = [(quick.clip_start(LENGTH, point), quick.clip_start(LENGTH, point) + quick.CLIP_SECONDS) for point in quick.POINTS]


def timeline(length=LENGTH, seed=3):
    """Dialogue over the whole file: (start, end, words), 0.35 s a word, 0.8 s between sentences."""
    rng, clock, result = random.Random(seed), 1.0, []
    while True:
        words = [rng.choice(GLUE if rng.random() < 0.45 else VOCABULARY) for _ in range(rng.randint(5, 10))]
        end = clock + 0.35 * len(words)
        if end > length - 1:
            return result
        result.append((clock, end, words))
        clock = end + 0.8


LINES = timeline()


def heard(lines=LINES, start=0.0, end=LENGTH):
    """The words a recognizer hears between ``start`` and ``end``, absolute times."""
    words = []
    for begin, finish, text in lines:
        step = (finish - begin) / len(text)
        words += [Word(begin + i * step, begin + (i + 1) * step - 0.02, token) for i, token in enumerate(text)
                  if start <= begin + (i + 0.5) * step < end]
    return words


def cues(lines=LINES, shift=0.0):
    return [SubCue(round((s + shift) * 1000), round((e + shift) * 1000), " ".join(w)) for s, e, w in lines]


def rewrite(lines, share, seed=9, region=None):
    """Replace this share of the words, as a translation would; ``region`` limits it to a time span."""
    rng = random.Random(seed)
    return [(s, e, [rng.choice(VOCABULARY) if rng.random() < share and (region is None or region[0] <= s < region[1]) else w for w in words])
            for s, e, words in lines]


def clips(lines=LINES):
    return [(start, end, heard(lines, start, end)) for start, end in CLIPS]


def judge(sources, lines=LINES):
    """The job's decision over the clips, without media: clips without speech are skipped, NEEDED must agree."""
    live, judged = list(range(len(sources))), 0
    for start, end, words in clips(lines)[:quick.MAX_CLIPS]:
        if judged == quick.NEEDED:
            break
        if quick.heard_count(words, start, end) < quick.MIN_HEARD:
            continue
        judged += 1
        live = [i for i in live if quick.passes(quick.clip_score(sources[i], words, start, end))]
        if not live:
            return None
    return live if judged == quick.NEEDED else None


def test_a_real_dub_caption_passes_every_clip():
    for start, end, words in clips():
        result = quick.clip_score(cues(), words, start, end)
        assert quick.passes(result) and result.agreement > 0.97
    assert judge([cues()]) == [0]


def test_a_dub_caption_with_a_small_sync_error_and_a_few_edits_still_passes():
    assert judge([cues(rewrite(LINES, 0.05), shift=0.8)]) == [0]


def test_cues_cut_by_the_clip_edges_do_not_count_against_a_real_caption():
    # A long cue straddling each clip boundary: the words inside the clip are heard, the cue is not scored.
    lines = [(s, e, w) for s, e, w in LINES if not any(s < edge + 3 < e or s < edge - 3 < e for clip in CLIPS for edge in clip)]
    for start, end in CLIPS:
        lines += [(start - 2.0, start + 3.0, ["copper", "lantern", "harbor", "violin", "the", "anchor"]),
                  (end - 3.0, end + 2.0, ["walnut", "ribbon", "the", "cinder", "kettle", "meadow"])]
    lines.sort()
    assert judge([cues(lines)], lines) == [0]


def test_a_translation_fails_even_where_it_matches_one_clip_closely():
    middle = CLIPS[0]
    close_in_the_middle = rewrite(rewrite(LINES, 0.5, region=(0, middle[0])), 0.5, seed=4, region=(middle[1], LENGTH))
    first = quick.clip_score(cues(close_in_the_middle), heard(LINES, *middle), *middle)
    assert quick.passes(first)  # the middle clip alone would be fooled
    assert judge([cues(close_in_the_middle)]) is None


def test_a_typical_translation_is_ruled_out_by_the_first_clip():
    start, end = CLIPS[0]
    result = quick.clip_score(cues(rewrite(LINES, 0.5)), heard(LINES, start, end), start, end)
    assert result.agreement < 0.7 and not quick.passes(result)


def test_a_truncated_track_fails_the_clips_it_does_not_reach():
    truncated = [line for line in LINES if line[0] < LENGTH * 0.4]
    assert judge([cues(truncated)]) is None


def test_a_signs_track_or_a_single_matching_line_is_not_enough():
    start, end = CLIPS[0]
    inside = [line for line in LINES if start + 5 < line[0] and line[1] < end - 5]
    sparse = [inside[0], inside[len(inside) // 2], inside[-1]]
    assert not quick.passes(quick.clip_score(cues(sparse), heard(LINES, start, end), start, end))
    assert not quick.passes(quick.clip_score(cues([inside[3]]), heard(LINES, start, end), start, end))


def test_a_clip_without_speech_is_replaced_by_the_next_position():
    song = CLIPS[1]  # the 30 % clip falls in an opening song: no dialogue there
    quiet = [line for line in LINES if not (song[0] - 5 < line[0] < song[1] + 5)]
    assert judge([cues(quiet)], quiet) == [0]


def test_too_few_clips_with_speech_make_the_check_inconclusive():
    silent = CLIPS[1:4]  # 30, 70 and 40 %: only two of the five clips read can be judged
    quiet = [line for line in LINES if not any(start - 5 < line[0] < end + 5 for start, end in silent)]
    assert judge([cues(quiet)], quiet) is None


def test_the_best_of_two_passing_tracks_is_the_one_matched():
    assert judge([cues(rewrite(LINES, 0.5)), cues()]) == [1]


# ---- with real media: the clip is read by seeking and the job uses it

pytestmark_media = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg is required")
SCALE = 25.0  # the test audio is a ramp: sample value = SCALE * seconds, so any window knows where it is


def srt(lines):
    def stamp(seconds):
        ms = round(seconds * 1000)
        return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"
    return "\n".join(f"{n}\n{stamp(s)} --> {stamp(e)}\n{' '.join(w)}\n" for n, (s, e, w) in enumerate(lines, 1))


def make_episode(path: Path, subs=()):
    """A 20-minute Matroska with ramp audio (exact PCM, so the stub engine can tell the time) and SRT tracks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
               "-i", f"aevalsrc=exprs={SCALE}*t/32768:s={RATE}:d={LENGTH}"]
    for number, (_, lines) in enumerate(subs):
        (path.parent / f"_sub{number}.srt").write_text(srt(lines))
        command += ["-i", str(path.parent / f"_sub{number}.srt")]
    command += [arg for number in range(1 + len(subs)) for arg in ("-map", str(number))]
    command += ["-c:a", "pcm_s16le", "-c:s", "srt", "-metadata:s:a:0", "language=eng", "-disposition:a:0", "default"]
    for number, (title, _) in enumerate(subs):
        command += [f"-metadata:s:s:{number}", "language=eng", f"-metadata:s:s:{number}", f"title={title}"]
    subprocess.run([*command, str(path)], check=True)
    for number in range(len(subs)):
        (path.parent / f"_sub{number}.srt").unlink()
    return path


class RampEngine:
    """Hears the script's words in whatever span of the ramp audio it is sent, and counts the audio."""

    def __init__(self):
        self.seconds = 0.0

    async def transcribe_words(self, wav, model, language):
        samples = len(wav[44:]) // 2
        first = struct.unpack_from("<h", wav, 44)[0] if samples else 0
        start, length = first / SCALE, samples / RATE
        self.seconds += length
        return [Word(w.start - start, w.end - start, w.text) for w in heard(LINES, start, start + length)], 0


@pytest.fixture
def library(tmp_path):
    store = Store(tmp_path / "data")
    yield type("Library", (), {"root": tmp_path / "tv", "work": tmp_path / "work", "ledger": Ledger(store.db), "store": store})()
    store.close()


async def noop(progress, duration=None):
    return None


async def run(library, name, engine):
    job = {"id": str(uuid.uuid4()), "model": "parakeet", "request": {"path": f"/tv/Show/{name}", "force": False}}
    result = await run_caption_job(job=job, progress=noop, engine=engine, ledger=library.ledger, workdir=library.work,
                                   roots={"/tv": str(library.root)})
    return result["result_json"]


@pytestmark_media
async def test_a_clip_is_read_by_seeking_with_the_same_times_as_a_whole_file_read(tmp_path):
    video = make_episode(tmp_path / "ep.mkv", subs=(("Dub", LINES),))
    pcm, files = await extract_clip(video, {"index": 0}, [{"index": 1, "codec_name": "subrip"}], tmp_path, 300.0, 90.0)
    samples = pcm.read_bytes()
    assert len(samples) / (2 * RATE) == pytest.approx(90.0, abs=0.05)
    assert struct.unpack_from("<h", samples, 0)[0] / SCALE == pytest.approx(300.0, abs=0.05)
    got = parse_subtitle(files[1].read_bytes(), "subrip")
    expected = [cue for cue in cues() if cue.end_ms > 300_000 and cue.start_ms < 390_000]
    assert {(c.start_ms, c.text) for c in got} >= {(c.start_ms, c.text) for c in expected if c.start_ms >= 300_000}
    assert max(c.start_ms for c in got) < 390_000  # reading stopped at the end of the clip


@pytestmark_media
async def test_an_episode_with_a_dub_caption_is_decided_from_the_clips_alone(library):
    video = make_episode(library.root / "Show" / "Ep 1.mkv", subs=(("English", rewrite(LINES, 0.5)), ("Dub (SDH)", LINES)))
    engine = RampEngine()
    report = await run(library, "Ep 1.mkv", engine)
    assert report["outcome"] == "matched_existing" and not sidecar_for(video).exists()
    assert report["matched"]["title"] == "Dub (SDH)" and report["quick"]["matched"]
    assert engine.seconds <= quick.NEEDED * quick.CLIP_SECONDS + 1  # only the clips were transcribed
    assert "extract" not in report["timings"] and "quick" in report["timings"]
    assert not list(library.work.glob("*/*"))


@pytestmark_media
async def test_an_episode_with_only_a_translation_gets_a_full_job_after_one_clip(library):
    video = make_episode(library.root / "Show" / "Ep 2.mkv", subs=(("English", rewrite(LINES, 0.5)),))
    engine = RampEngine()
    report = await run(library, "Ep 2.mkv", engine)
    assert report["outcome"] == "written" and sidecar_for(video).is_file()
    assert not report["quick"]["matched"] and len(report["quick"]["clips"]) == 1
    assert engine.seconds == pytest.approx(LENGTH + quick.CLIP_SECONDS, abs=2)
    assert not list(library.work.glob("*/*"))
