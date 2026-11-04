"""Cut PCM audio into engine-sized windows at quiet points instead of mid-word."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audio import RATE, _pcm_wav

FRAME = RATE // 50  # 20 ms
MAX_WINDOW = 28.0  # seconds; the same bound as the standard transcription windows
SEARCH = 9.0  # seconds before the limit in which a quiet cut point is looked for
MIN_WINDOW = 0.2


def _energy(samples: np.ndarray) -> np.ndarray:
    """RMS per 20 ms frame, smoothed over 100 ms, read in blocks so long films stay small."""
    frames = len(samples) // FRAME
    rms = np.empty(frames, dtype=np.float32)
    if not frames:
        return rms
    block = 3000
    for first in range(0, frames, block):
        chunk = samples[first * FRAME:min(frames, first + block) * FRAME].astype(np.float32)
        rms[first:first + len(chunk) // FRAME] = np.sqrt((chunk.reshape(-1, FRAME) ** 2).mean(axis=1))
    return np.convolve(rms, np.ones(5, dtype=np.float32) / 5, mode="same")


def cut_points(samples: np.ndarray) -> list[int]:
    """Sample indexes ending each window: the latest point within SEARCH seconds of the limit
    whose energy is near the quietest, so windows stay long and cuts fall in pauses."""
    energy = _energy(samples)
    limit, search = int(MAX_WINDOW * 50), int(SEARCH * 50)
    cuts, start = [], 0
    while len(energy) - start > limit:
        low, high = start + limit - search, start + limit
        span = energy[low:high]
        floor = float(span.min())
        quiet = np.nonzero(span <= max(floor * 1.25, floor + 3.0))[0]
        start = low + int(quiet[-1])
        cuts.append(start * FRAME)
    return cuts


def window_bounds(pcm_path: Path) -> list[tuple[int, int]]:
    """(first, last+1) sample of each window to transcribe, skipping digital silence."""
    samples = np.memmap(pcm_path, dtype="<i2", mode="r")
    bounds = [0, *cut_points(samples), len(samples)]
    return [(begin, end) for begin, end in zip(bounds, bounds[1:])
            if end - begin >= MIN_WINDOW * RATE and np.asarray(samples[begin:end]).any()]


def read_window(pcm_path: Path, begin: int, end: int) -> bytes:
    """One window as a mono 16 kHz WAV."""
    samples = np.memmap(pcm_path, dtype="<i2", mode="r")
    return _pcm_wav(np.asarray(samples[begin:end]).tobytes())
