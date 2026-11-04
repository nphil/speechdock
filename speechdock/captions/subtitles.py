"""Tolerant readers for existing English subtitles (SRT, WebVTT, ASS/SSA).

Real-world subtitle files overlap, carry styling, signs and advertisements, so unlike
``audio.parse_srt`` nothing here raises on odd content: a reader returns what it can use.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SubCue:
    start_ms: int
    end_ms: int
    text: str  # plain text: markup, sound descriptions and speaker labels removed


_STAMP = r"(?:\d+:)?\d{1,2}:\d{2}[,.]\d{1,3}"
_ARROW = re.compile(rf"^\s*({_STAMP})\s*-->\s*({_STAMP})")
_TAGS = re.compile(r"<[^>]*>|\{[^}]*\}")
_SOUND = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_NOTES = re.compile("[♪♫♬]")
_LABEL = re.compile(r"^[A-Z][A-Z0-9 .'-]{1,24}:\s+")
_SPAM = re.compile(r"www\.|https?://|\.(?:com|net|org|app)\b|opensubtitles|addic7ed|subscene|yts\b|yify", re.I)
# Style and effect heuristics proven on this library's ASS releases: keep dialogue, drop signs and songs.
_SIGN_STYLE = re.compile(r"sign|song|kara|^op|^ed\b|lyric|title|note|^ts|typeset|insert|screen|text", re.I)
_DIALOGUE_STYLE = re.compile(r"default|main|dialog|italic|flashback|top|alt|overlap|internal|narrat|thought", re.I)
_POSITIONED = re.compile(r"\\(?:pos|move|org|i?clip)\(|\\p[1-9]|\\[kK][fo]?\d")


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _milliseconds(stamp: str) -> int:
    parts = stamp.replace(",", ".").split(":")
    seconds, _, fraction = parts[-1].partition(".")
    hours = int(parts[-3]) if len(parts) > 2 else 0
    return ((hours * 60 + int(parts[-2])) * 60 + int(seconds)) * 1000 + int(fraction.ljust(3, "0")[:3])


def clean_text(raw: str) -> str:
    """Plain spoken text of one cue; empty when nothing speakable (or only spam) remains."""
    text = raw.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
    text = html.unescape(_TAGS.sub("", text))
    lines = []
    for line in text.split("\n"):
        if _NOTES.search(line):
            continue  # sung lyrics are not speech
        line = _SOUND.sub(" ", line)
        line = _LABEL.sub("", line.strip())
        line = re.sub(r"^\s*[-–—]\s*", "", line)
        line = " ".join(line.split())
        if line:
            lines.append(line)
    text = "\n".join(lines)
    return "" if _SPAM.search(text) else text


def parse_timed_text(text: str) -> list[SubCue]:
    """Read SRT or WebVTT blocks without validating order, overlap or indexes."""
    cues, lines, index = [], text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), 0
    while index < len(lines):
        match = _ARROW.match(lines[index])
        index += 1
        if not match:
            continue
        body = []
        while index < len(lines) and lines[index].strip():
            body.append(lines[index])
            index += 1
        start, end = _milliseconds(match.group(1)), _milliseconds(match.group(2))
        cleaned = clean_text("\n".join(body))
        if cleaned and end > start:
            cues.append(SubCue(start, end, cleaned))
    return cues


def _ass_milliseconds(stamp: str) -> int:
    hours, minutes, seconds = stamp.strip().split(":")
    whole, _, fraction = seconds.partition(".")
    return ((int(hours) * 60 + int(minutes)) * 60 + int(whole)) * 1000 + int(fraction.ljust(3, "0")[:3])


def parse_ass(text: str) -> list[SubCue]:
    """Dialogue events of an ASS/SSA script; signs, songs, karaoke and drawings are dropped."""
    fields, cues = None, []
    for line in text.splitlines():
        if line.startswith("Format:") and fields is None and "Start" in line and "Text" in line:
            fields = [name.strip() for name in line[7:].split(",")]
            continue
        if not line.startswith("Dialogue:") or not fields:
            continue
        values = dict(zip(fields, line[9:].split(",", len(fields) - 1)))
        style, body = values.get("Style", "").strip(), values.get("Text", "")
        dialogue_style = bool(_DIALOGUE_STYLE.search(style))
        if (_SIGN_STYLE.search(style) and not dialogue_style) or (_POSITIONED.search(body) and not dialogue_style):
            continue
        try:
            start, end = _ass_milliseconds(values["Start"]), _ass_milliseconds(values["End"])
        except (KeyError, ValueError):
            continue
        cleaned = clean_text(body)
        if cleaned and end > start:
            cues.append(SubCue(start, end, cleaned))
    cues.sort(key=lambda cue: (cue.start_ms, cue.end_ms))
    merged: list[SubCue] = []
    for cue in cues:
        # ASS stacks the same line on several layers (shadow, outline): keep one.
        if merged and merged[-1].text == cue.text and cue.start_ms <= merged[-1].end_ms + 50:
            merged[-1] = SubCue(merged[-1].start_ms, max(merged[-1].end_ms, cue.end_ms), cue.text)
        else:
            merged.append(cue)
    return merged


def parse_subtitle(data: bytes | str, codec: str | None = None) -> list[SubCue]:
    """Parse subtitle text by codec/extension hint ('ass', 'ssa' or SRT/WebVTT-like)."""
    text = data if isinstance(data, str) else decode_text(data)
    if (codec or "").lower() in {"ass", "ssa"} or text.lstrip().startswith("[Script Info]"):
        return parse_ass(text)
    return parse_timed_text(text)
