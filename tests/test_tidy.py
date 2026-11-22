"""Subtitle tidy-up rules, on real Matroska files made with ffmpeg."""
import shutil
import subprocess
from pathlib import Path

import pytest

from speechdock import tidy as sn

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg is required")

DUR = 120


def stamp(ms, ass=False):
    if ass:
        return f"{ms // 3600000}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02}.{ms % 1000 // 10:02}"
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def srt(lines):
    return "".join(f"{n}\n{stamp(s)} --> {stamp(e)}\n{t}\n\n" for n, (s, e, t) in enumerate(lines, 1))


def ass(events):
    head = ("[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n"
            "Style: Sign,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,8,10,10,10,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    return head + "".join(f"Dialogue: 0,{stamp(s, True)},{stamp(e, True)},{st},,0,0,0,,{t}\n" for s, e, st, t in events)


DIALOGUE = [(1000 + i * 2500, 3000 + i * 2500, f"line number {i} we said") for i in range(45)]
TRANSLATION = [(s, e, f"translated {i} words here") for i, (s, e, _) in enumerate(DIALOGUE)]
# A sign animated frame by frame (40 ms each), one in a drawing, one sign placed with \pos.
SIGN_FRAMES = [(20000 + k * 40, 20040 + k * 40, "Sign", "{\\fad(0,0)}Tokyo Station") for k in range(50)]
SIGNS = SIGN_FRAMES + [(30000, 33000, "Default", "{\\pos(100,50)}Episode Title"), (40000, 42000, "Sign", "{\\p1}m 0 0 l 10 10")]


def mkv(path: Path, audio=("jpn", "eng"), subs=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    for _ in audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=300:sample_rate=16000:duration={DUR}"]
    for n, (_, ext, text, _f) in enumerate(subs):
        f = path.parent / f"_in{n}.{ext}"
        f.write_text(text)
        cmd += ["-i", str(f)]
    cmd += [a for n in range(len(audio) + len(subs)) for a in ("-map", str(n))]
    cmd += ["-c:a", "aac"]
    for n, lang in enumerate(audio):
        cmd += [f"-metadata:s:a:{n}", f"language={lang}"]
    for n, (title, ext, _t, forced) in enumerate(subs):
        cmd += [f"-c:s:{n}", "ass" if ext == "ass" else "srt", f"-metadata:s:s:{n}", "language=eng", f"-metadata:s:s:{n}", f"title={title}"]
        if forced:
            cmd += [f"-disposition:s:{n}", "forced"]
    subprocess.run([*cmd, str(path)], check=True)
    for n in range(len(subs)):
        (path.parent / f"_in{n}.{subs[n][1]}").unlink()
    return path


@pytest.fixture
def lib(tmp_path, monkeypatch):
    tv = tmp_path / "tv"
    tv.mkdir()
    monkeypatch.setattr(sn, "LIBRARIES", (str(tv),))
    return tv


def run(video, tmp_path, *, anime=False, record=None):
    return sn.tidy(video, anime=anime, record=record, quarantine=tmp_path / "quarantine")


def record(video, candidates):
    st = video.stat()
    return {"video_size": st.st_size, "video_mtime_ns": st.st_mtime_ns, "details": {"candidates": candidates}}


def test_an_animated_sign_becomes_one_cue_at_the_top_and_drawings_are_dropped():
    cues = sn.collapse([c for c in sn.parse_ass(ass(SIGNS)) if c.sign])
    assert [(c.start, c.end, c.text) for c in cues] == [(20000, 22000, "{\\an8}Tokyo Station"), (30000, 33000, "{\\an8}Episode Title")]


def test_merging_signs_twice_changes_nothing():
    base = sn.parse_srt(srt(DIALOGUE))
    signs = sn.collapse([c for c in sn.parse_ass(ass(SIGNS)) if c.sign])
    once = sn.render(sn.merge_signs(base, signs))
    again = sn.render(sn.merge_signs(sn.parse_srt(once), signs))
    assert once == again and once.count("Tokyo Station") == 1


def test_short_top_dialogue_in_an_existing_file_is_kept():
    text = srt([(1000, 2000, "{\\an8}Hi"), *DIALOGUE[1:]])
    assert "{\\an8}Hi" in sn.render(sn.merge_signs(sn.collapse(sn.parse_srt(text)), []))


def test_a_show_gets_its_embedded_english_as_en_srt_and_junk_is_quarantined(lib, tmp_path):
    video = mkv(lib / "Show" / "Season 1" / "Show - S01E01.mkv", audio=("eng",), subs=(("English", "srt", srt(DIALOGUE), False),))
    (video.parent / "Show - S01E01.srt").write_text(srt([(s, e, f"esto es la frase {i} que dijimos") for i, (s, e, _) in enumerate(DIALOGUE)]))
    (video.parent / "Show - S01E01.es.srt").write_text(srt(DIALOGUE))
    report = run(video, tmp_path)
    assert report["errors"] == [] and "Line number 0" not in (video.parent / "Show - S01E01.en.srt").read_text()
    assert "line number 44 we said" in (video.parent / "Show - S01E01.en.srt").read_text()
    left = sorted(p.name for p in video.parent.iterdir())
    assert left == ["Show - S01E01.en.srt", "Show - S01E01.mkv"]
    assert len(report["quarantined"]) == 2 and report["journal"]
    assert run(video, tmp_path)["journal"] == []  # a second run changes nothing


def test_an_untagged_english_sidecar_is_renamed_and_an_existing_en_srt_is_never_rewritten(lib, tmp_path):
    video = mkv(lib / "Show" / "Show - S01E02.mkv", audio=("eng",))
    (video.parent / "Show - S01E02.srt").write_text(srt([(s, e, "you know what i want to do now right") for s, e, _ in DIALOGUE]))
    run(video, tmp_path)
    assert (video.parent / "Show - S01E02.en.srt").exists() and not (video.parent / "Show - S01E02.srt").exists()
    original = (video.parent / "Show - S01E02.en.srt").read_bytes()
    run(video, tmp_path)
    assert (video.parent / "Show - S01E02.en.srt").read_bytes() == original


def test_anime_gets_two_complete_files_matched_by_measurement_not_title(lib, tmp_path):
    full = ass([(s, e, "Default", t) for s, e, t in TRANSLATION] + SIGNS)
    video = mkv(lib / "Anime" / "Anime - S01E01.mkv",
                subs=(("English", "ass", full, False), ("Signs", "ass", ass(SIGNS), True), ("Full Subtitles", "srt", srt(DIALOGUE), False)))
    # The track titled "Full Subtitles" is the one that matches the dub: SpeechDock measured it.
    measured = record(video, [{"source": "embedded", "stream": 2, "agreement": 0.42}, {"source": "embedded", "stream": 4, "agreement": 0.93}])
    report = run(video, tmp_path, anime=True, record=measured)
    assert report["errors"] == []
    orig = (video.parent / "Anime - S01E01.en.srt").read_text()
    dub = (video.parent / "Anime - S01E01.en.cc.srt").read_text()
    assert "translated 3 words here" in orig and "we said" not in orig
    assert "line number 3 we said" in dub and "translated" not in dub
    for text in (orig, dub):
        assert text.count("{\\an8}Tokyo Station") == 1 and "{\\an8}Episode Title" in text and "m 0 0" not in text
    assert run(video, tmp_path, anime=True, record=measured)["journal"] == []


def test_anime_with_english_audio_and_no_measurement_asks_for_one(lib, tmp_path):
    video = mkv(lib / "Anime" / "Anime - S01E02.mkv", subs=(("English", "srt", srt(DIALOGUE), False),))
    report = run(video, tmp_path, anime=True, record=None)
    assert "speechdock" in report["pending"] and report["dub"] is None


def test_undo_restores_every_file(lib, tmp_path):
    video = mkv(lib / "Show" / "Show - S01E03.mkv", audio=("eng",), subs=(("English", "srt", srt(DIALOGUE), False),))
    (video.parent / "Show - S01E03.es.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhola\n")
    report = run(video, tmp_path)
    assert sn.undo(report["journal"]) == len(report["journal"])
    assert sorted(p.name for p in video.parent.iterdir()) == ["Show - S01E03.es.srt", "Show - S01E03.mkv"]


def test_a_film_with_only_a_dub_caption_keeps_it_and_waits_for_bazarr(lib, tmp_path):
    video = mkv(lib / "Film (2001)" / "Film (2001).mkv", audio=("kor", "eng"), subs=(("English [CC]", "srt", srt(DIALOGUE), False),))
    (video.parent / "Film (2001).en.cc.srt").write_text(srt(DIALOGUE))
    report = run(video, tmp_path, record=record(video, [{"source": "embedded", "stream": 2, "agreement": 0.95}]))
    assert report["orig"] is None and "bazarr" in report["pending"]
    assert sorted(p.name for p in video.parent.iterdir()) == ["Film (2001).en.cc.srt", "Film (2001).mkv"]


PICTURE_ONLY = {"format": {"duration": DUR}, "streams": [
    {"index": 1, "codec_type": "audio", "tags": {"language": "jpn"}},
    {"index": 2, "codec_type": "audio", "tags": {"language": "eng"}},
    {"index": 3, "codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle", "tags": {"language": "eng"}}]}


def test_picture_only_dialogue_is_read_measured_and_filed_as_translation_or_dub(lib, tmp_path, monkeypatch):
    monkeypatch.setattr(sn, "OCR_DIR", tmp_path / "ocr")
    video = lib / "Anime" / "Anime - S01E09.mkv"  # ffmpeg cannot write picture subtitles; the probe report is given
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    first = sn.decide(video, PICTURE_ONLY, [], anime=True, record=None)
    assert first.ocr == [3] and first.orig is None and "ocr-dialogue" in first.pending
    cache = sn.ocr_cache(video, 3)
    cache.parent.mkdir(parents=True)
    cache.write_text(srt(TRANSLATION))
    unmeasured = sn.decide(video, PICTURE_ONLY, [], anime=True, record=record(video, []))
    assert unmeasured.orig is None and "speechdock-ocr" in unmeasured.pending  # never filed before the dub is compared
    measured = sn.decide(video, PICTURE_ONLY, [], anime=True, record=record(video, [{"source": "ocr", "stream": 3, "agreement": 0.41}]))
    report = sn.apply_plan(measured, sn.Journal(tmp_path / "quarantine"))
    target = video.with_name("Anime - S01E09.en.srt")
    assert report["written"] == [str(target)] and "translated 7 words here" in target.read_text()
    dub = sn.decide(video, PICTURE_ONLY, [], anime=True, record=record(video, [{"source": "ocr", "stream": 3, "agreement": 0.93}]))
    assert dub.orig is None and dub.dub == {"ocr": str(cache), "ocr_stream": 3} and "bazarr" in dub.pending
