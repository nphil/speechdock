import numpy as np
import pytest

from speechdock.audio import RATE, render
from speechdock.captions import cues as cue_tools
from speechdock.captions import windows
from speechdock.captions.cues import CleanStats, build_cues, clean_words, wrap_lines
from speechdock.captions.names import Glossary, build_glossary, correct_names, distance, skeleton
from speechdock.captions.subtitles import SubCue
from speechdock.engine import Word


def words_at(text, start=0.0, pace=0.35, confidence=1.0):
    result, clock = [], start
    for token in text.split():
        result.append(Word(clock, clock + pace - 0.05, token, confidence))
        clock += pace
    return result


def test_loops_are_collapsed_but_short_repeats_survive():
    stats = CleanStats()
    chant = words_at("King! " * 9 + "We win.")
    kept = cue_tools.collapse_loops(chant, stats)
    assert [w.text for w in kept] == ["King!"] * 3 + ["We", "win."]
    assert stats.loops == 6
    phrase = words_at("Thank you. " * 5 + "Bye.")
    assert [w.text for w in cue_tools.collapse_loops(phrase, CleanStats())] == ["Thank", "you."] * 2 + ["Bye."]
    twice = words_at("No, no. Stop, stop.")
    assert cue_tools.collapse_loops(twice, CleanStats()) == twice


def test_words_squeezed_into_the_last_frames_of_a_window_are_dropped():
    good = words_at("Real speech here.")
    squeezed = [Word(19.84 + i * 0.04, 19.88 + i * 0.04, token) for i, token in enumerate("be a good thing.".split())]
    stats = CleanStats()
    assert cue_tools.drop_compressed(good + squeezed, stats) == good
    assert stats.compressed == 4
    pair = [Word(1, 1.04, "a"), Word(1.04, 1.08, "b")]
    assert cue_tools.drop_compressed(pair, CleanStats()) == pair


def test_unsure_runs_are_dropped_and_real_speech_is_kept():
    speech = words_at("I will find him and stop him.", confidence=0.97)
    singing = words_at("Send you can go out of that, country has your kai no car.", start=10, confidence=0.35)
    kept, stats = clean_words(speech + singing)
    assert kept == speech and stats.unsure == len(singing)
    short = words_at("Oh.", start=30, confidence=0.4)
    assert clean_words(short)[0] == short  # one or two words are never judged


def test_lines_are_balanced_and_never_wider_than_42():
    lines = wrap_lines("I truly want to keep secret everything after I wrote it down")
    assert len(lines) == 2 and max(map(len, lines)) <= 42 and abs(len(lines[0]) - len(lines[1])) < 15
    assert wrap_lines("Short line.") == ["Short line."]
    assert wrap_lines("x" * 50) is None
    assert wrap_lines("one two three four five six seven, eight nine ten eleven twelve") == [
        "one two three four five six seven,", "eight nine ten eleven twelve"]


def spoken_script():
    text = ("Master, where should we go next? Uh, let's see. We got the hot pot ingredients, so I guess that leaves the "
            "toilet paper and shampoo. Oh. Wow. I guess popularity gets you places. His own action figure. Must be awesome. "
            "I will contact the toy companies, but this is a don't call us, we'll call you situation. Let's roll. Right!")
    return words_at(text, start=2.0)


def test_cues_are_readable_ordered_and_never_overlap():
    cues = build_cues(spoken_script())
    assert render(cues, "srt")  # validates order, overlap and positive length
    for cue in cues:
        lines = cue.text.split("\n")
        assert 1 <= len(lines) <= 2 and all(len(line) <= 42 for line in lines)
        assert cue.end_ms - cue.start_ms <= 7000
    assert all(b.start_ms - a.end_ms >= 80 for a, b in zip(cues, cues[1:]))
    spoken = " ".join(w.text for w in spoken_script())
    assert " ".join(cue.text.replace("\n", " ") for cue in cues) == spoken


def test_cues_break_at_sentence_ends_before_mid_phrase():
    cues = build_cues(words_at("I will contact the toy companies right away. Nah, this is a don't call us situation.", pace=0.4))
    assert [cue.text.replace("\n", " ") for cue in cues] == [
        "I will contact the toy companies right away.", "Nah, this is a don't call us situation."]


def test_short_neighbours_share_a_cue_and_every_cue_stays_up_long_enough():
    cues = build_cues(words_at("Oh. Wow. Right!", pace=0.3))
    assert [cue.text for cue in cues] == ["Oh. Wow. Right!"]
    lone = build_cues(words_at("Yeah.", pace=0.3))[0]
    assert lone.end_ms - lone.start_ms >= 1000


def test_reading_speed_is_met_when_the_next_cue_leaves_room():
    cues = build_cues(words_at("Hey, if you snooze, you're gonna lose, dude.", pace=0.2) + words_at("Later.", start=8))
    first = cues[0]
    assert len(first.text.replace("\n", " ")) / ((first.end_ms - first.start_ms) / 1000) <= cue_tools.MAX_CPS + 0.5


def test_a_pause_always_starts_a_new_cue():
    cues = build_cues(words_at("First part", start=0) + words_at("second part", start=3.0))
    assert [cue.text for cue in cues] == ["First part", "second part"]


def test_distance_and_sound_skeleton():
    assert distance("genos", "jenos") == 1 and distance("saitama", "saitma") == 1 and distance("ab", "ba") == 1
    assert skeleton("Genos") == skeleton("Jenos") == "252"
    assert skeleton("Hold") == skeleton("Held")


def subtitle(*lines):
    return [SubCue(i * 2000, i * 2000 + 1500, line) for i, line in enumerate(lines)]


def test_names_are_repaired_from_the_subtitle_glossary():
    cues = subtitle("Call Genos now.", "The Hero Association sent Saitama.", "Tatsumaki is strong.")
    heard = words_at("Hey, Jenos, you need a hand. Saytama saved us. Tatsumaki is here.", confidence=0.8)
    glossary = build_glossary(cues, heard)
    assert {"genos", "saitama"} <= set(glossary.names)
    fixed, fixes = correct_names(heard, glossary)
    text = " ".join(w.text for w in fixed)
    assert "Genos," in text and "Saitama" in text and "Jenos" not in text and "Saytama" not in text
    assert {(f.heard, f.name) for f in fixes} == {("Jenos", "Genos"), ("Saytama", "Saitama")}
    assert [w.start for w in fixed] == [w.start for w in heard]  # timing is untouched


def test_a_name_written_only_at_a_sentence_start_needs_the_speaker_to_say_it_too():
    cues = subtitle("Genos, come here.")
    assert "genos" not in build_glossary(cues, words_at("Genos ran.")).names
    assert "genos" in build_glossary(cues, words_at("Genos ran. Genos jumped.")).names


def test_a_split_name_is_joined_and_keeps_both_words_timing():
    glossary = build_glossary(subtitle("Saitama is here.", "Where is Saitama?"), [])
    fixed, fixes = correct_names(words_at("Look, Sai Tama! He came."), glossary)
    assert [w.text for w in fixed][:2] == ["Look,", "Saitama!"]
    assert fixed[1].start == pytest.approx(0.35) and fixed[1].end == pytest.approx(0.35 * 2 + 0.3)
    assert fixes[0].heard == "Sai Tama"


def test_ordinary_words_and_confident_single_hearings_are_left_alone():
    glossary = Glossary({"held": "Held", "nezuko": "Nezuko"}, frozenset({"is", "a"}))
    confident = [Word(0, 0.5, "Hold", 0.96), Word(0.6, 1.0, "on."), Word(1.1, 1.5, "Nezuko"), Word(1.6, 1.7, "a")]
    fixed, fixes = correct_names(confident, glossary)
    assert fixed == confident and fixes == []  # "Hold" is not Held; Nezuko + "a" is not a split name
    unsure_twice = [Word(0, 0.5, "Hold", 0.96), Word(5, 5.5, "Hold", 0.96)]
    assert [w.text for w in correct_names(unsure_twice, glossary)[0]] == ["Held", "Held"]  # a repeated spelling is corrected


def test_words_said_in_lowercase_are_never_names():
    cues = subtitle("Hold on, Genos.", "We hold the line.")
    glossary = build_glossary(cues, words_at("hold the line"))
    assert "hold" not in glossary.names and "genos" in glossary.names


def test_quiet_points_are_chosen_for_window_cuts():
    seconds = 70
    audio = (np.random.default_rng(1).normal(0, 6000, seconds * RATE)).astype("<i2")
    for gap in ((24.0, 24.6), (51.0, 51.6)):
        audio[int(gap[0] * RATE):int(gap[1] * RATE)] = (np.random.default_rng(2).normal(0, 20, int((gap[1] - gap[0]) * RATE))).astype("<i2")
    cuts = windows.cut_points(audio)
    assert len(cuts) == 2
    assert 24.0 * RATE <= cuts[0] <= 24.6 * RATE and 51.0 * RATE <= cuts[1] <= 51.6 * RATE
    edges = [0, *cuts, len(audio)]
    assert all((b - a) / RATE <= windows.MAX_WINDOW for a, b in zip(edges, edges[1:]))


def test_silent_audio_has_no_windows(tmp_path):
    path = tmp_path / "silence.pcm"
    path.write_bytes(b"\x00\x00" * RATE * 5)
    assert windows.window_bounds(path) == []
    speech = tmp_path / "speech.pcm"
    speech.write_bytes((np.random.default_rng(3).normal(0, 3000, RATE * 5)).astype("<i2").tobytes())
    assert windows.window_bounds(speech) == [(0, RATE * 5)]
    assert windows.read_window(speech, 0, RATE).startswith(b"RIFF")
