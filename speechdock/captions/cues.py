"""Clean a timed transcript and turn it into readable caption cues.

Cues hold at most two lines of about 42 characters, break at sentence ends and pauses
before they break mid-phrase, stay on screen long enough to read, and never overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..audio import Cue
from ..engine import Word

MAX_LINE = 42
MAX_CHARS = 2 * MAX_LINE
MAX_CPS = 20.0  # reading speed the display time is stretched to, characters per second
MIN_DURATION = 1.0
MAX_DURATION = 6.5
PAUSE = 0.9  # silence that always ends a cue
GAP = 0.08  # minimum clear time between cues
LEAD, TAIL = 0.1, 0.3  # a cue appears slightly before its first word and lingers after its last
MIN_CONFIDENCE = 0.65  # a run of 3+ words the model was this unsure of on average is noise (songs, effects)

_END = re.compile(r"[.!?\u2026]+[\"')\]\u201d]*$")
_CLAUSE = re.compile(r"[,;:\u2014\u2013-][\"')\]\u201d]*$")
_ABBREVIATIONS = frozenset("mr. mrs. ms. dr. st. jr. sr. vs. etc. no. prof.".split())
_FUNCTION_WORDS = frozenset(
    "a an the of to in on at for and or but with that this his her my your our their its is are was were be".split())


@dataclass
class CleanStats:
    compressed: int = 0  # words squeezed into the last frames of a cut-off window
    unsure: int = 0  # words of low-confidence runs (singing, effects)
    loops: int = 0  # words of looped repetitions


def _text(words: list[Word]) -> str:
    return " ".join(word.text for word in words)


def _sentence_end(word: Word) -> bool:
    return bool(_END.search(word.text)) and word.text.lower() not in _ABBREVIATIONS


def sentences(words: list[Word]) -> list[list[Word]]:
    """Split at sentence-ending punctuation and at long pauses."""
    result, current = [], []
    for index, word in enumerate(words):
        current.append(word)
        following = words[index + 1] if index + 1 < len(words) else None
        if following is None or _sentence_end(word) or following.start - word.end > PAUSE:
            result.append(current)
            current = []
    return result


def drop_compressed(words: list[Word], stats: CleanStats) -> list[Word]:
    """Drop runs of 3+ words squeezed into a few frames: the model completing cut-off speech."""
    keep, run = [], []
    for word in [*words, None]:
        if word is not None and word.end - word.start <= 0.05:
            run.append(word)
            continue
        if len(run) >= 3:
            stats.compressed += len(run)
        else:
            keep.extend(run)
        run = []
        if word is not None:
            keep.append(word)
    return keep


def drop_unsure(words: list[Word], stats: CleanStats) -> list[Word]:
    """Drop stretches of 3+ words the model was mostly unsure of; real dialogue scores near 1."""
    keep = []
    for sentence in sentences(words):
        for start in range(0, len(sentence), 8):
            chunk = sentence[start:start + 8]
            if len(chunk) >= 3 and sum(word.confidence for word in chunk) / len(chunk) < MIN_CONFIDENCE:
                stats.unsure += len(chunk)
            else:
                keep.extend(chunk)
    return keep


def _token(word: Word) -> str:
    return re.sub(r"[^a-z0-9']", "", word.text.lower())


def collapse_loops(words: list[Word], stats: CleanStats) -> list[Word]:
    """Collapse phrases repeated in a loop: a single word more than 3 times, a phrase more than twice."""
    tokens = [_token(word) for word in words]
    keep, index = [], 0
    while index < len(words):
        for size in (1, 2, 3, 4):
            unit = tokens[index:index + size]
            if len(unit) < size or not all(unit):
                continue
            repeats = 1
            while tokens[index + repeats * size:index + (repeats + 1) * size] == unit:
                repeats += 1
            allowed = 3 if size == 1 else 2
            if repeats > allowed:
                keep.extend(words[index:index + allowed * size])
                stats.loops += (repeats - allowed) * size
                index += repeats * size
                break
        else:
            keep.append(words[index])
            index += 1
    return keep


def clean_words(words: list[Word]) -> tuple[list[Word], CleanStats]:
    stats = CleanStats()
    return collapse_loops(drop_unsure(drop_compressed(words, stats), stats), stats), stats


def wrap_lines(text: str) -> list[str] | None:
    """Split into at most two balanced lines of at most MAX_LINE characters; None if impossible."""
    if len(text) <= MAX_LINE:
        return [text]
    tokens, best = text.split(" "), None
    for index in range(1, len(tokens)):
        top, bottom = " ".join(tokens[:index]), " ".join(tokens[index:])
        if len(top) > MAX_LINE or len(bottom) > MAX_LINE:
            continue
        cost = abs(len(top) - len(bottom)) + (3 if len(top) > len(bottom) else 0)
        cost += 6 if tokens[index - 1].lower() in _FUNCTION_WORDS else 0
        cost -= 8 if _END.search(top) or _CLAUSE.search(top) else 0
        if best is None or cost < best[0]:
            best = (cost, [top, bottom])
    return best[1] if best else None


def _fits(words: list[Word]) -> bool:
    return (len(_text(words)) <= MAX_CHARS and words[-1].end - words[0].start <= MAX_DURATION
            and wrap_lines(_text(words)) is not None)


def _split(words: list[Word]) -> list[list[Word]]:
    """Break a sentence that is too long at its most natural point, recursively."""
    if _fits(words) or len(words) < 2:
        return [words]
    total, best = len(_text(words)), None
    for index in range(1, len(words)):
        left, right = words[:index], words[index:]
        cost = abs(len(_text(left)) - len(_text(right))) / total * 10
        cost -= 5 if _CLAUSE.search(left[-1].text) else 0
        cost -= min(right[0].start - left[-1].end, 1.0) * 6
        cost += 4 if left[-1].text.lower().strip(",;:") in _FUNCTION_WORDS else 0
        if best is None or cost < best[0]:
            best = (cost, index)
    return _split(words[:best[1]]) + _split(words[best[1]:])


def segments(words: list[Word]) -> list[list[Word]]:
    """Word groups that each become one cue."""
    pieces = [piece for sentence in sentences(words) for piece in _split(sentence)]
    merged: list[list[Word]] = []
    for piece in pieces:
        if merged:
            joined = merged[-1] + piece
            short = min(len(_text(merged[-1])), len(_text(piece))) < 20
            if short and piece[0].start - merged[-1][-1].end <= 0.6 and _fits(joined) and joined[-1].end - joined[0].start <= 4.5:
                merged[-1] = joined
                continue
        merged.append(piece)
    return merged


def build_cues(words: list[Word]) -> list[Cue]:
    """Readable, non-overlapping cues for timed words (seconds), in order."""
    groups = segments(words)
    cues, previous_end = [], 0.0
    for index, group in enumerate(groups):
        text = _text(group)
        lines = wrap_lines(text)
        if lines is None:  # one unbreakable token wider than a line
            lines = [text]
        start = max(group[0].start - LEAD, previous_end + (GAP if cues else 0.0), 0.0)
        end = max(group[-1].end + TAIL, start + MIN_DURATION, start + len(text) / MAX_CPS)
        if index + 1 < len(groups):
            end = min(end, max(groups[index + 1][0].start - GAP, group[-1].end))
        end = max(end, start + 0.3)
        cues.append(Cue(round(start * 1000), round(end * 1000), "\n".join(lines)))
        previous_end = end
    return cues
