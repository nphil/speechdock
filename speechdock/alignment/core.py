"""CPU-only known-text CTC alignment. Scores are model scores, not confidence probabilities."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

REVISION = "a19f851b3d42865797e410752b4c570c871e4825"
MODEL_SHA256 = "cd5040c147381580ed73258143dd8e0c28e800a09e74ee42ee2b3e8cb4d760a3"
VOCAB_SHA256 = "4178db26b3c7570f6a47f14ac6a1c7b32950b8c2800fb097287e53776934f1c5"
ALGORITHM = "wav2vec2-base-960h-int8-ctc-viterbi-greedy-guard-v1"
PROVENANCE = "cpu_acoustic_ctc;independent_unprompted_greedy_exact_lexical_check;20ms_frames;uncalibrated_scores"
SAMPLE_RATE = 16000
STRIDE = 320


class Rejected(ValueError):
    """A safe, public reason code; never include paths, URLs or credentials."""


def digest_file(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def words(text: str) -> list[tuple[str, int, int]]:
    """Retain exact UTF-8 spans including attached punctuation, without rewriting text."""
    result = []
    punctuation = '.,!?;:"()“”‘’…—–'
    for match in re.finditer(r"\S+", text):
        raw = match.group()
        lexical = raw.strip(punctuation).replace("’", "'")
        if not re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)*", lexical):
            raise Rejected("unsupported_text")
        result.append((lexical.upper(), len(text[:match.start()].encode()), len(text[:match.end()].encode())))
    if not result or len(result) > 64:
        raise Rejected("unsupported_text")
    return result


def viterbi(logp: np.ndarray, target: list[int], blank: int) -> list[list[int]]:
    """Full CTC topology: blank, repeat/stay and legal skip; no wildcard or interpolation."""
    states = np.full(2 * len(target) + 1, blank, dtype=np.int32)
    states[1::2] = target
    count = len(states)
    if len(logp) < len(target):
        raise Rejected("incomplete_acoustic_path")
    prev = np.full(count, -np.inf, dtype=np.float32)
    prev[0] = logp[0, blank]
    prev[1] = logp[0, target[0]]
    back = np.zeros((len(logp), count), dtype=np.uint8)
    legal_skip = np.zeros(count, dtype=bool)
    legal_skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    for frame in range(1, len(logp)):
        one = np.concatenate(([-np.inf], prev[:-1]))
        two = np.concatenate(([-np.inf, -np.inf], prev[:-2]))
        two[~legal_skip] = -np.inf
        candidates = np.stack((prev, one, two))
        choice = np.argmax(candidates, axis=0)
        back[frame] = choice
        prev = candidates[choice, np.arange(count)] + logp[frame, states]
    state = count - 1 if prev[-1] > prev[-2] else count - 2
    if not np.isfinite(prev[state]):
        raise Rejected("incomplete_acoustic_path")
    frames: list[list[int]] = [[] for _ in target]
    for frame in range(len(logp) - 1, -1, -1):
        if state % 2:
            frames[state // 2].append(frame)
        if frame:
            state -= int(back[frame, state])
    if state not in (0, 1) or any(not item for item in frames):
        raise Rejected("incomplete_acoustic_path")
    return [list(reversed(item)) for item in frames]


class AcousticAligner:
    def __init__(self, directory: Path):
        import onnxruntime as ort

        model = directory / "model_quantized.onnx"
        vocab = directory / "vocab.json"
        if digest_file(model) != MODEL_SHA256 or digest_file(vocab) != VOCAB_SHA256:
            raise Rejected("model_checksum_mismatch")
        self.vocab = json.loads(vocab.read_text())
        self.inverse = {value: key for key, value in self.vocab.items()}
        self.blank = self.vocab["<pad>"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])

    def emissions(self, pcm: np.ndarray) -> np.ndarray:
        if pcm.ndim != 1 or not 1600 <= len(pcm) <= SAMPLE_RATE * 16 or not np.isfinite(pcm).all():
            raise Rejected("invalid_audio_bounds")
        if float(np.sqrt(np.mean(pcm * pcm))) < 0.0005:
            raise Rejected("silence")
        audio = (pcm - pcm.mean()) / np.sqrt(pcm.var() + 1e-7)
        logits = self.session.run(["logits"], {"input_values": audio.astype(np.float32)[None, :]})[0][0]
        maximum = logits.max(axis=1, keepdims=True)
        shifted = logits - maximum
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    def greedy(self, logp: np.ndarray) -> str:
        tokens = []
        previous = None
        for token in np.argmax(logp, axis=1):
            token = int(token)
            if token != previous and token != self.blank:
                tokens.append(self.inverse[token])
            previous = token
        return " ".join("".join(tokens).replace("|", " ").split())

    def align(self, pcm: np.ndarray, text: str, origin_ms: float, start_ms: int, end_ms: int) -> dict:
        lexical = words(text)
        normalized = " ".join(word[0] for word in lexical)
        # Repeated lexical units can have indistinguishable alternate assignments. Be conservative.
        if len({word[0] for word in lexical}) != len(lexical):
            raise Rejected("ambiguous_repeated_words")
        logp = self.emissions(pcm)
        transcript = self.greedy(logp)
        if transcript != normalized:
            raise Rejected("independent_lexical_mismatch")
        target = [self.vocab[char] for char in normalized.replace(" ", "|")]
        frames = viterbi(logp, target, self.blank)
        scores = [float(logp[item, token].mean()) for item, token in zip(frames, target)]
        if min(scores) < -3.0 or float(np.mean(scores)) < -1.0:
            raise Rejected("weak_acoustic_evidence")
        result = []
        cursor = 0
        previous_end = start_ms
        for word, first, last in lexical:
            selected = frames[cursor:cursor + len(word)]
            begin = round(origin_ms + selected[0][0] * 1000 * STRIDE / SAMPLE_RATE)
            end = round(origin_ms + (selected[-1][-1] + 1) * 1000 * STRIDE / SAMPLE_RATE)
            if not start_ms <= begin < end <= end_ms or begin < previous_end:
                raise Rejected("outside_visible_bounds")
            if end - begin < 20 or end - begin > max(1200, len(word) * 250):
                raise Rejected("implausible_word_duration")
            result.append({"start_byte": first, "end_byte": last, "start_ms": begin, "end_ms": end})
            previous_end = end
            cursor += len(word) + 1
        return {"words": result, "mean_log_score": float(np.mean(scores)), "minimum_character_log_score": min(scores)}
