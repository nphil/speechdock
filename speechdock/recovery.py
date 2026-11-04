"""Recover a speech window whose timings the engine's checks rejected, without inventing any timing.

The checks stay strict. When one rejects a window, Parakeet answers the same way every time, so asking again
is pointless; instead the same audio is cut at quiet points near its middle and the pieces are sent again, each
through the very same checks. Only when no cut works does a second model (full Whisper) get the window. Every
word and time in the result comes from an engine reply that passed the checks.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import numpy as np

from .audio import RATE, Cue, _pcm_wav, _wav_pcm, parse_srt, render
from .engine import Engine, ResponseRejected

log = logging.getLogger("speechdock.recovery")

MIN_PIECE = 4.0  # seconds; a piece is cut only if both halves keep at least this much
SPLIT_FRACTION = 0.2  # cut within +-20 % of the middle of a piece
CANDIDATES = 5  # quietest distinct cut points tried per piece
MIN_GAP = 0.25  # seconds between candidate cut points
BUDGET = 40  # engine calls per failing window (the worst real window needed 20)
FALLBACK_BUDGET = 12


@dataclass(frozen=True)
class Fallback:
    """A second model for a window the first one's timings cannot be recovered for."""
    engine: Engine
    model: str


def cut_candidates(samples: np.ndarray) -> list[int]:
    """Sample indexes of the quietest places near the middle, quietest first.

    10 ms mean-square frames smoothed over 50 ms; the lowest level wins (ties: closest to the middle). The
    cut sits on the centre of a frame, a whole number of milliseconds, so offsets stay exact."""
    count, frame, smooth = len(samples), RATE // 100, 5
    low = max(int(MIN_PIECE * RATE), int(count * (0.5 - SPLIT_FRACTION)))
    high = min(count - int(MIN_PIECE * RATE), int(count * (0.5 + SPLIT_FRACTION)))
    if high < low:
        return []
    frames = count // frame
    energy = (samples[:frames * frame].astype(np.float64).reshape(frames, frame) ** 2).mean(axis=1)
    level = np.convolve(np.pad(energy, smooth // 2, mode="edge"), np.ones(smooth) / smooth, mode="valid")
    index = np.arange(low // frame, min(frames - 1, high // frame) + 1)
    order = np.lexsort((np.abs(index - count / 2 / frame), np.round(10 * np.log10(level[index] + 1), 1)))
    picked: list[int] = []
    for position in order:
        cut = min(max(int(index[position]) * frame + frame // 2, low), high)
        if all(abs(cut - other) >= MIN_GAP * RATE for other in picked):
            picked.append(cut)
        if len(picked) == CANDIDATES:
            break
    return picked


async def _recover(strict: Callable[[np.ndarray, float], Awaitable[list[Cue]]], samples: np.ndarray, offset: float,
                   budget: list[int], known_failed: bool = False) -> list[Cue] | None:
    """Cues for ``samples`` (times from the window start, ``offset`` seconds in) or None when neither the
    piece nor any cut of it passes the checks. ``known_failed`` skips the retry of audio that just failed."""
    if not known_failed:
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        try:
            return await strict(samples, offset)
        except ResponseRejected:
            pass
    for cut in cut_candidates(samples):
        left = await _recover(strict, samples[:cut], offset, budget)
        if left is None:
            continue
        right = await _recover(strict, samples[cut:], offset + cut / RATE, budget)
        if right is not None:
            return left + right
    return None


def clock(seconds: float) -> str:
    minutes, rest = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02}:{rest:02}" if hours else f"{minutes}:{rest:02}"


async def transcribe_window(engine: Engine, wav: bytes, model: str, language: str, task: str, *,
                            fallback: Callable[[], Awaitable[Fallback | None]] | None = None,
                            at: float = 0.0, recovered: list[dict] | None = None) -> str:
    """One window as SRT text (times from the window start), recovering it if the engine's reply is rejected.

    ``at`` is where the window starts in the recording (for messages and the ``recovered`` record). A window
    that cannot be recovered raises ``ResponseRejected`` naming its place in the recording."""
    try:
        return await engine.transcribe(wav, model, language, task)
    except ResponseRejected as rejected:
        first = rejected
    samples = np.frombuffer(_wav_pcm(wav), dtype="<i2")
    end = at + len(samples) / RATE

    async def attempt(by: Engine, by_model: str, budget: int, known_failed: bool) -> list[Cue] | None:
        async def strict(piece: np.ndarray, offset: float) -> list[Cue]:
            cues = parse_srt(await by.transcribe(_pcm_wav(piece.tobytes()), by_model, language, task), offset)
            if cues and cues[-1].end_ms > round((offset + len(piece) / RATE) * 1000):
                raise ResponseRejected("Subtitle extends beyond its audio piece")
            return cues
        return await _recover(strict, samples, 0.0, [budget], known_failed)

    cues, how = await attempt(engine, model, BUDGET, True), "cut"
    second = await fallback() if cues is None and fallback is not None else None
    if cues is None and second is not None:
        cues, how = await attempt(second.engine, second.model, FALLBACK_BUDGET, False), "fallback"
    if cues is None:
        raise ResponseRejected(f"{first} (audio {clock(at)}-{clock(end)}; cutting it at quiet points"
                               + (" and the fallback model" if second is not None else "") + " did not help)") from first
    log.warning("Speech engine reply rejected at %s-%s (%s); recovered by %s", clock(at), clock(end), first, how)
    if recovered is not None:
        recovered.append({"start": round(at, 2), "end": round(end, 2), "how": how, "reason": str(first)})
    return render(cues, "srt")
