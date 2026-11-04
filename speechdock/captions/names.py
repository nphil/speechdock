"""Proper-noun repair: fix names the speech model misspells using the file's own subtitle.

Parakeet accepts no vocabulary prompt, so names it has never seen come back phonetically
spelled ("Jenos" for "Genos"). The same file's English subtitle spells every name correctly,
so names collected from it correct the transcript afterwards. A correction needs a
capitalized token the speaker never says in lowercase that sounds like a known name and is
one or two letters away from it, and that is either repeated (a consistent misspelling) or
one the model itself was unsure of; ordinary words are left alone.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace

from ..engine import Word
from .subtitles import SubCue

_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\u2019-]*[A-Za-z]|[A-Za-z]")
_EDGES = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)
_COMMON = frozenset(
    "yeah okay hmm huh whoa wow hello sorry thanks thank please sure alright anyway nope yep "
    "gonna wanna gotta well come listen look right fine stop wait what why when where which "
    "while this that these those there here then than they them their with would could should "
    "about after again also because before being between both each every just maybe never only "
    "other some such very still".split())
_SOUNDS = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"), **dict.fromkeys("dt", "3"),
           "l": "4", **dict.fromkeys("mn", "5"), "r": "6"}
UNSURE = 0.9  # a single occurrence is corrected only when the model's confidence is at most this


def skeleton(word: str) -> str:
    """Sound class of each consonant with vowels dropped: Genos and Jenos share one skeleton."""
    out = ""
    for char in word.lower():
        sound = _SOUNDS.get(char)
        if sound and not out.endswith(sound):
            out += sound
    return out


def _fold(word: str) -> str:
    """Spelling variants that sound alike: Philo/Filo, Cuseno/Kuseno, Kyrie/Kirie."""
    return word.lower().replace("ph", "f").replace("ck", "k").replace("c", "k").replace("q", "k").replace("y", "i")


def distance(a: str, b: str) -> int:
    """Edit distance counting an adjacent swap as one edit."""
    before, row = None, list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            best = min(row[j] + 1, current[j - 1] + 1, row[j - 1] + (left != right))
            if before is not None and j > 1 and left == b[j - 2] and a[i - 2] == right:
                best = min(best, before[j - 2] + 1)
            current.append(best)
        before, row = row, current
    return row[-1]


def _stem(token: str) -> str:
    token = token.replace("\u2019", "'")
    return token[:-2] if token.lower().endswith("'s") else token


def _is_title(token: str) -> bool:
    return len(token) >= 4 and token.isalpha() and token[0].isupper() and token[1:].islower()


def _parts(word: Word) -> tuple[str, str, str, str]:
    """(leading punctuation, bare token, trailing punctuation, possessive suffix)."""
    before, core, after = _EDGES.match(word.text).groups()
    bare = _stem(core)
    return before, bare, after, core[len(bare):]


@dataclass(frozen=True)
class Glossary:
    names: dict[str, str]  # lowercase -> spelling
    known: frozenset[str]  # lowercase words seen in lowercase: ordinary vocabulary


def build_glossary(cues: list[SubCue], words: list[Word]) -> Glossary:
    """Proper nouns of the subtitle, checked against what the speaker said.

    A name is a capitalized token of 4+ letters that is never written or said in lowercase
    and is not a common interjection. One written only at the start of a sentence needs
    more evidence: a second written occurrence, or the speaker saying it capitalized twice.
    """
    known, spoken_title = set(), Counter()
    for word in words:
        core = _parts(word)[1]
        if core.islower():
            known.add(core)
        elif _is_title(core):
            spoken_title[core.lower()] += 1
    inside, start = Counter(), Counter()
    for cue in cues:
        for line in cue.text.split("\n"):
            tokens = _TOKEN.findall(line)
            for position, token in enumerate(tokens):
                stem = _stem(token)
                if stem.islower():
                    known.add(stem)
                elif _is_title(stem):
                    (start if position == 0 else inside)[stem] += 1
    names = {}
    for name in sorted({*inside, *start}, key=lambda item: -(inside[item] + start[item])):
        key = name.lower()
        written = inside[name] + start[name]
        if key in known or key in _COMMON:
            continue
        if inside[name] or written >= 2 or spoken_title[key] >= 2:
            names[key] = name
    return Glossary(names, frozenset(known))


@dataclass(frozen=True)
class Correction:
    heard: str
    name: str
    count: int


def correct_names(words: list[Word], glossary: Glossary) -> tuple[list[Word], list[Correction]]:
    """Replace misspelled names, and names split in two words ("Sai Tama"), with glossary spellings."""
    names, known = glossary.names, glossary.known
    if not names:
        return words, []
    by_sound: dict[str, list[str]] = {}
    for name in names:
        by_sound.setdefault(skeleton(name), []).append(name)
    heard_count = Counter(core.lower() for core in (_parts(word)[1] for word in words) if _is_title(core))

    def lookup(key: str) -> str | None:
        """The one glossary name this token is a misspelling of (or exactly spells)."""
        if key in names:
            return names[key]
        folded = _fold(key)
        close = [name for name in by_sound.get(skeleton(key), ())
                 if distance(folded, _fold(name)) <= (1 if len(name) < 7 else 2)]
        return names[close[0]] if len(close) == 1 and len(key) >= 4 else None

    fixes: Counter = Counter()
    result: list[Word] = []
    index = 0
    while index < len(words):
        word = words[index]
        before, core, after, possessive = _parts(word)
        key = core.lower()
        if _is_title(core) and key not in known and key not in names and (heard_count[key] >= 2 or word.confidence <= UNSURE):
            name = lookup(key)
            if name:
                result.append(replace(word, text=f"{before}{name}{possessive}{after}"))
                fixes[(core, name)] += 1
                index += 1
                continue
        if index + 1 < len(words) and core.isalpha() and core[0].isupper() and not after:
            following = words[index + 1]
            _, core2, after2, possessive2 = _parts(following)
            second = core2.lower()
            if (core2.isalpha() and len(core) + len(core2) >= 6 and key not in names and second not in names
                    and (key not in known or second not in known)):
                name = lookup(key + second)
                if name:
                    result.append(Word(word.start, following.end, f"{before}{name}{possessive2}{after2}",
                                       min(word.confidence, following.confidence)))
                    fixes[(f"{core} {core2}", name)] += 1
                    index += 2
                    continue
        result.append(word)
        index += 1
    return result, [Correction(heard, name, count) for (heard, name), count in fixes.most_common()]
