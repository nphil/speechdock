import os

import pytest

from speechdock.captions import media
from speechdock.captions.media import english_text_tracks, language_of, pick_english_audio, sidecar_subtitles


def audio(index, language, title="", default=0, tag_key="language", **disposition):
    tags = {tag_key: language, **({"title": title} if title else {})}
    return {"index": index, "codec_type": "audio", "codec_name": "aac", "channels": 2, "tags": tags,
            "disposition": {"default": default, "comment": 0, **disposition}}


def subtitle(index, language="eng", codec="subrip", forced=0, title=""):
    tags = {**({"language": language} if language else {}), **({"title": title} if title else {})}
    return {"index": index, "codec_type": "subtitle", "codec_name": codec, "tags": tags, "disposition": {"forced": forced}}


def test_english_is_chosen_even_when_japanese_is_the_default():
    streams = [audio(1, "jpn", default=1), audio(2, "eng")]
    assert pick_english_audio(streams)["index"] == 2


def test_default_english_stream_beats_an_earlier_non_default_one():
    streams = [audio(1, "eng"), audio(2, "jpn"), audio(3, "eng", default=1)]
    assert pick_english_audio(streams)["index"] == 3


def test_first_english_stream_when_none_is_marked_default():
    assert pick_english_audio([audio(4, "jpn"), audio(5, "eng"), audio(6, "eng")])["index"] == 5


def test_two_default_streams_still_give_the_english_one():
    # Some releases flag both the Japanese and the English track as default.
    assert pick_english_audio([audio(1, "jpn", default=1), audio(2, "eng", default=1)])["index"] == 2


def test_commentary_is_never_chosen_by_flag_or_title():
    streams = [audio(1, "eng", default=1, comment=1), audio(2, "eng", title="Director's Commentary", default=1), audio(3, "eng")]
    assert pick_english_audio(streams)["index"] == 3
    assert pick_english_audio(streams[:2]) is None


def test_audio_description_is_skipped():
    streams = [audio(1, "eng", default=1, visual_impaired=1), audio(2, "eng", title="English Audio Description"), audio(3, "eng", title="Stereo")]
    assert pick_english_audio(streams)["index"] == 3


def test_no_english_or_undetermined_language_yields_none():
    assert pick_english_audio([audio(1, "jpn", default=1), audio(2, "und"), audio(3, "")]) is None
    assert pick_english_audio([]) is None


def test_language_spellings_and_tag_case():
    for spelling in ("eng", "ENG", "en", "en-US", "en_GB", "English"):
        assert language_of(audio(1, spelling)) in {"en", "eng", "english"}, spelling
    assert pick_english_audio([audio(1, "eng", tag_key="LANGUAGE")])["index"] == 1
    assert language_of(audio(1, "jpn")) == "jpn"


def test_only_audio_streams_are_considered():
    stream = subtitle(1)
    assert pick_english_audio([stream]) is None


def test_embedded_text_tracks_exclude_bitmap_forced_and_foreign():
    streams = [
        subtitle(3, "eng", "ass", title="Dialogue"),
        subtitle(4, "eng", "hdmv_pgs_subtitle"),
        subtitle(5, "eng", "subrip", forced=1),
        subtitle(6, "spa", "subrip"),
        subtitle(7, "", "webvtt"),
        subtitle(8, "und", "mov_text"),
        subtitle(9, "zxx", "ass", title="Signs"),
    ]
    assert [s["index"] for s in english_text_tracks(streams)] == [3, 7, 8]


def test_sidecars_are_english_unforced_and_not_temporary(tmp_path):
    video = tmp_path / "Show - S01E01.mkv"
    video.touch()
    wanted = ["Show - S01E01.en.srt", "Show - S01E01.en.hi.srt", "Show - S01E01.srt", "Show - S01E01.en.cc.srt", "Show - S01E01.eng.ass"]
    ignored = ["Show - S01E01.en.forced.srt", "Show - S01E01.es.srt", "Show - S01E01.pt-BR.srt", "Show - S01E02.en.srt",
               ".Show - S01E01.en.cc.srt.ab12cd34.tmp", "Show - S01E01.nfo", "Show - S01E01-thumb.jpg"]
    for name in wanted + ignored:
        (tmp_path / name).write_text("x")
    assert sorted(p.name for p in sidecar_subtitles(video)) == sorted(wanted)


def test_oversized_sidecar_is_ignored(tmp_path, monkeypatch):
    video = tmp_path / "A.mkv"
    video.touch()
    (tmp_path / "A.en.srt").write_text("12345")
    monkeypatch.setattr(media, "MAX_SIDECAR_BYTES", 3)
    assert sidecar_subtitles(video) == []


def test_dub_situation_needs_english_and_a_tagged_non_english_audio_track():
    assert media.is_dub_situation([audio(1, "jpn"), audio(2, "eng")])
    assert media.is_dub_situation([audio(1, "ja"), audio(2, "en-US")])
    assert media.is_dub_situation([audio(1, "eng"), audio(2, "fre"), audio(3, "ger")])  # more than two languages
    assert media.is_dub_situation([audio(1, "eng"), audio(2, "eng", title="Commentary"), audio(3, "spa")])


def test_english_only_or_foreign_only_audio_is_not_a_dub():
    assert not media.is_dub_situation([audio(1, "eng"), audio(2, "eng")])
    assert not media.is_dub_situation([audio(1, "jpn"), audio(2, "kor")])
    assert not media.is_dub_situation([])
    assert not media.is_dub_situation([subtitle(1, "eng"), subtitle(2, "jpn")])  # only audio streams count


def test_commentary_and_description_tracks_count_for_neither_side():
    assert not media.is_dub_situation([audio(1, "jpn"), audio(2, "eng", title="Director's Commentary")])
    assert not media.is_dub_situation([audio(1, "jpn"), audio(2, "eng", descriptions=1)])
    assert not media.is_dub_situation([audio(1, "eng"), audio(2, "jpn", comment=1)])
    assert not media.is_dub_situation([audio(1, "eng"), audio(2, "fre", title="Audio Description")])


def test_untagged_and_undetermined_audio_counts_as_neither_language():
    assert not media.is_dub_situation([audio(1, "eng"), {"index": 2, "codec_type": "audio", "tags": {}, "disposition": {}}])
    assert not media.is_dub_situation([audio(1, "eng"), {"index": 2, "codec_type": "audio"}])
    for code in ("und", "zxx", "mis", "mul", ""):
        assert not media.is_dub_situation([audio(1, "eng"), audio(2, code)]), code
    assert not media.is_dub_situation([audio(1, "und"), audio(2, "jpn")])  # no English side
    assert media.is_dub_situation([audio(1, "eng"), audio(2, "und"), audio(3, "jpn")])


def fake_tool(tmp_path, monkeypatch, name, script):
    """Put an executable ``name`` running the shell ``script`` first on PATH."""
    path = tmp_path / "bin"
    path.mkdir(exist_ok=True)
    tool = path / name
    tool.write_text("#!/bin/sh\n" + script + "\n")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{path}:{os.environ['PATH']}")


async def test_probe_reports_the_exit_code_when_ffprobe_prints_nothing(tmp_path, monkeypatch):
    fake_tool(tmp_path, monkeypatch, "ffprobe", "exit 1")
    with pytest.raises(media.MediaError) as caught:
        await media.probe(tmp_path / "x.mkv")
    assert str(caught.value) == "Could not read the video: ffprobe exited with code 1 and no message"


async def test_probe_error_carries_the_stderr_without_blank_lines_and_runs_at_warning_level(tmp_path, monkeypatch):
    fake_tool(tmp_path, monkeypatch, "ffprobe",
              'case "$*" in *"-v warning"*) ;; *) exit 9;; esac\n'
              'printf "\\n[aac] Audio object type 42 is not implemented.\\n\\n  \\nInvalid data found when processing input\\n" >&2\nexit 1')
    with pytest.raises(media.MediaError) as caught:
        await media.probe(tmp_path / "x.mkv")
    assert str(caught.value) == ("Could not read the video: ffprobe exited with code 1: "
                                 "[aac] Audio object type 42 is not implemented.\nInvalid data found when processing input")


async def test_probe_error_keeps_only_the_last_300_characters(tmp_path, monkeypatch):
    fake_tool(tmp_path, monkeypatch, "ffprobe", 'i=0; while [ $i -lt 100 ]; do echo "noise line $i" >&2; i=$((i+1)); done; echo THE END >&2; exit 2')
    with pytest.raises(media.MediaError) as caught:
        await media.probe(tmp_path / "x.mkv")
    message = str(caught.value)
    assert message.endswith("noise line 99\nTHE END") and "exited with code 2: " in message
    assert len(message.split(": ", 2)[2]) <= 300


async def test_probe_returns_streams_and_duration_and_passes_the_timeout(tmp_path, monkeypatch):
    fake_tool(tmp_path, monkeypatch, "ffprobe",
              'echo \'{"streams": [{"index": 0, "codec_type": "audio"}], "format": {"duration": "12.5"}}\'\necho harmless warning >&2')
    result = await media.probe(tmp_path / "x.mkv", timeout=30)
    assert result == {"streams": [{"index": 0, "codec_type": "audio"}], "duration": 12.5}


async def test_extract_error_carries_the_exit_code_and_stderr(tmp_path, monkeypatch):
    fake_tool(tmp_path, monkeypatch, "ffmpeg", 'echo "Error while decoding stream #0:1: Invalid data found" >&2\nexit 183')

    async def progress(_):
        pass

    with pytest.raises(media.MediaError) as caught:
        await media.extract(tmp_path / "x.mkv", {"index": 1}, [], tmp_path, 10.0, progress)
    assert str(caught.value) == ("Could not decode the English audio: ffmpeg exited with code 183: "
                                 "Error while decoding stream #0:1: Invalid data found")
    fake_tool(tmp_path, monkeypatch, "ffmpeg", "exit 1")
    with pytest.raises(media.MediaError, match="ffmpeg exited with code 1 and no message"):
        await media.extract(tmp_path / "x.mkv", {"index": 1}, [], tmp_path, 10.0, progress)


async def test_extract_succeeds_when_ffmpeg_does_and_ignores_warnings(tmp_path, monkeypatch):
    fake_tool(tmp_path, monkeypatch, "ffmpeg", 'for last; do :; done\nprintf "\\0\\0" > "$last"\necho "some warning" >&2')

    async def progress(_):
        pass

    pcm, files, note = await media.extract(tmp_path / "x.mkv", {"index": 1}, [], tmp_path, 10.0, progress)
    assert pcm.read_bytes() == b"\0\0" and files == {} and note is None
