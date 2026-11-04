import random

import pytest

from speechdock.captions.agreement import lcs_pairs, norm_words, score
from speechdock.captions.pipeline import AGREEMENT_THRESHOLD
from speechdock.captions.subtitles import SubCue, clean_text, parse_ass, parse_subtitle, parse_timed_text
from speechdock.engine import Word

VOCABULARY = ("river mountain castle shadow winter silver thunder garden window letter bridge candle forest "
              "harbor engine hammer ladder pocket marble velvet copper meadow lantern compass feather orchard "
              "tunnel anchor saddle pepper violin cobble ribbon cinder walnut thimble kettle trellis").split()
GLUE = "the a to of and in is was we they you he she it that this for with on at".split()


def script(seed=3, sentences=70):
    rng = random.Random(seed)
    return [[rng.choice(GLUE if rng.random() < 0.45 else VOCABULARY) for _ in range(rng.randint(6, 11))] for _ in range(sentences)]


def timeline(sentences, shift=0.0):
    """(start, end, words) per sentence: 0.4 s a word with a 0.7 s pause between sentences."""
    clock, result = 1.0, []
    for words in sentences:
        end = clock + 0.4 * len(words)
        result.append((clock + shift, end + shift, words))
        clock = end + 0.7
    return result


def spoken_words(sentences):
    words = []
    for start, end, text in timeline(sentences):
        step = (end - start) / len(text)
        words += [Word(start + i * step, start + (i + 1) * step - 0.02, token) for i, token in enumerate(text)]
    return words


def captions(sentences, shift=0.0):
    return [SubCue(round(start * 1000), round(end * 1000), " ".join(words)) for start, end, words in timeline(sentences, shift)]


def rewrite(sentences, share, seed=9):
    """Replace this share of the words, as a translation of the same scene would."""
    rng = random.Random(seed)
    return [[rng.choice(VOCABULARY) if rng.random() < share else word for word in words] for words in sentences]


SENTENCES = script()


def test_identical_text_and_timing_agree_fully():
    result = score(captions(SENTENCES), spoken_words(SENTENCES))
    assert result.agreement == pytest.approx(1.0) and result.precision == result.recall == pytest.approx(1.0)
    assert result.spoken_words == result.track_words


def test_a_real_dub_caption_with_a_few_edits_passes_the_threshold():
    result = score(captions(rewrite(SENTENCES, 0.06)), spoken_words(SENTENCES))
    assert result.agreement > AGREEMENT_THRESHOLD
    assert 0.85 < result.precision <= 1 and 0.85 < result.recall <= 1


def test_a_translation_of_the_same_scene_fails_the_threshold():
    # Half the words differ: same events and names, other wording.
    result = score(captions(rewrite(SENTENCES, 0.5)), spoken_words(SENTENCES))
    assert result.agreement < AGREEMENT_THRESHOLD - 0.15


def test_a_truncated_track_fails_even_though_everything_it_says_is_spoken():
    result = score(captions(SENTENCES[: len(SENTENCES) // 4]), spoken_words(SENTENCES))
    assert result.precision == pytest.approx(1.0)
    assert result.recall < 0.35 and result.agreement < AGREEMENT_THRESHOLD


def test_a_track_far_out_of_sync_fails():
    result = score(captions(SENTENCES, shift=45.0), spoken_words(SENTENCES))
    assert result.agreement < 0.2


def test_small_sync_error_is_tolerated():
    assert score(captions(SENTENCES, shift=1.2), spoken_words(SENTENCES)).agreement > 0.95


def test_a_spoken_word_is_never_credited_twice():
    sentences = SENTENCES[:20]
    doubled = [cue for cue in captions(sentences) for _ in range(2)]
    result = score(doubled, spoken_words(sentences))
    assert result.recall == pytest.approx(1.0) and result.precision == pytest.approx(0.5, abs=0.05)


def test_empty_inputs_score_zero():
    assert score([], spoken_words(SENTENCES[:3])).agreement == 0
    assert score(captions(SENTENCES[:3]), []).agreement == 0


def test_sound_descriptions_music_and_speaker_labels_do_not_count():
    assert clean_text("[door slams]\n♪ la la ♪ (laughs)\nNARRATOR: Hello there.") == "Hello there."
    cue = SubCue(1000, 3000, clean_text("- Hello there!\n<i>my friend</i> {\\an8}"))
    assert cue.text == "Hello there!\nmy friend"


def test_normalisation_ignores_case_punctuation_apostrophes_and_small_numbers():
    assert norm_words("It's 29 years, isn\u2019t it?!") == ["its", "twenty", "nine", "years", "isnt", "it"]
    assert norm_words("Top-tier 100%") == ["top", "tier", "100"]


def test_lcs_pairs_are_an_ordered_common_subsequence():
    pairs = lcs_pairs("a b c d e".split(), "b x c e".split())
    assert pairs == [(1, 0), (2, 2), (4, 3)]


def test_srt_with_overlaps_ads_and_bad_blocks_is_still_read():
    text = ("1\n00:00:01,000 --> 00:00:03,000\nHello there\n\n2\n00:00:02,500 --> 00:00:04,000\nStill here\n\n"
            "3\nnot a timing line\ntext\n\n4\n00:00:05,000 --> 00:00:06,000\nEnjoy Movies. Signup Here -> WWW.APP\n\n"
            "5\n00:01:00.5 --> 00:01:02.25\nVTT style stamp\n")
    cues = parse_timed_text(text)
    assert [(c.start_ms, c.end_ms, c.text) for c in cues] == [(1000, 3000, "Hello there"), (2500, 4000, "Still here"), (60500, 62250, "VTT style stamp")]


def test_ass_keeps_dialogue_and_drops_signs_songs_and_positioned_text():
    ass = "\n".join([
        "[Script Info]", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        r"Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\i1}Hello, there{\i0}\NFriend",
        r"Dialogue: 1,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\i1}Hello, there{\i0}\NFriend",
        "Dialogue: 0,0:00:04.00,0:00:06.00,Signs,,0,0,0,,A shop sign",
        "Dialogue: 0,0:00:07.00,0:00:09.00,OP Romaji,,0,0,0,,sing along",
        r"Dialogue: 0,0:00:10.00,0:00:12.00,Italics,,0,0,0,,{\pos(10,10)}Narration kept",
        r"Dialogue: 0,0:00:13.00,0:00:14.00,Misc,,0,0,0,,{\pos(10,10)}Floating text",
        "Comment: 0,0:00:15.00,0:00:16.00,Default,,0,0,0,,not an event",
    ])
    cues = parse_ass(ass)
    assert [(c.start_ms, c.text) for c in cues] == [(1000, "Hello, there\nFriend"), (10000, "Narration kept")]
    assert parse_subtitle(ass.encode(), "ass") == cues
