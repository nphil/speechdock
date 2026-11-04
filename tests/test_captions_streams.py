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
