"""How well an existing subtitle agrees with what is actually said in the audio.

A real dub caption repeats the spoken English almost verbatim; a translation of the
original-language script shares only names and filler words, and a truncated or partial
track covers only part of the dialogue. Both directions are measured so either fails.
"""
from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from ..engine import Word
from .subtitles import SubCue

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)*")


def _spell(number: int) -> list[str]:
    if number < 20:
        return [_ONES[number]]
    tens, ones = divmod(number, 10)
    return [_TENS[tens]] + ([_ONES[ones]] if ones else [])


def norm_words(text: str) -> list[str]:
    """Lowercase comparable tokens: no punctuation or apostrophes, small numbers spelled out."""
    result = []
    for token in _TOKEN.findall(text.lower().replace("\u2019", "'")):
        token = token.replace("'", "")
        if token.isdigit() and int(token) < 100:
            result.extend(_spell(int(token)))
        elif token:
            result.append(token)
    return result


def lcs_pairs(left: list[str], right: list[str]) -> list[tuple[int, int]]:
    """Index pairs of one longest common subsequence."""
    n, m = len(left), len(right)
    if not n or not m:
        return []
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row, below = table[i], table[i + 1]
        for j in range(m - 1, -1, -1):
            row[j] = below[j + 1] + 1 if left[i] == right[j] else max(below[j], row[j + 1])
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        if left[i] == right[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


@dataclass(frozen=True)
class Agreement:
    agreement: float  # F1 of precision and recall: 1.0 means the same words at the same times
    precision: float  # share of the track's words that are heard near their cue
    recall: float  # share of the spoken words that the track carries
    track_words: int
    spoken_words: int


def score(cues: list[SubCue], spoken: list[Word], tolerance: float = 2.5) -> Agreement:
    """Compare a subtitle with the timed transcript of the same audio.

    Each cue's words are matched in order against the not-yet-claimed spoken words within
    ``tolerance`` seconds of the cue, so no spoken word is credited twice.
    """
    heard = [(word.start + word.end) / 2 for word in spoken for _ in norm_words(word.text)]
    tokens = [token for word in spoken for token in norm_words(word.text)]
    claimed = [False] * len(tokens)
    matched = track_words = 0
    for cue in sorted(cues, key=lambda item: (item.start_ms, item.end_ms)):
        words = norm_words(cue.text)
        if not words:
            continue
        track_words += len(words)
        first = bisect_left(heard, cue.start_ms / 1000 - tolerance)
        last = bisect_right(heard, cue.end_ms / 1000 + tolerance)
        free = [index for index in range(first, last) if not claimed[index]]
        pairs = lcs_pairs(words, [tokens[index] for index in free])
        for _, position in pairs:
            claimed[free[position]] = True
        matched += len(pairs)
    precision = matched / track_words if track_words else 0.0
    recall = matched / len(tokens) if tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Agreement(f1, precision, recall, track_words, len(tokens))
