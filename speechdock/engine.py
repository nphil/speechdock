"""Clients and timing adapters for shared Whisper and pinned CrispASR runtimes."""

import io
import math
import textwrap
import wave
from urllib.parse import urlsplit, urlunsplit

import httpx

from .audio import Cue, render
from .languages import LANGUAGES, normalize_language


class EngineError(ValueError):
    """An actionable engine connectivity or response error."""


def _aligned_cues(payload: dict) -> list[Cue]:
    """Use aligned cue endpoints; internal token intervals are not subtitles."""
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise ValueError("Invalid speech engine segments")
    cues = []
    for segment in payload["segments"]:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError("Invalid speech engine segment")
        words = segment.get("words")
        if not isinstance(words, list) or not words:
            raise ValueError("Speech engine did not return aligned word timings")
        if not isinstance(words[0], dict) or not isinstance(words[-1], dict):
            raise ValueError("Invalid speech engine cue endpoints")
        # whisper.cpp's internal subword alignment can overlap or reverse.
        # We expose segment cues, not individual word intervals: consume only
        # the first onset and final end, then validate the resulting cues.
        times = (words[0].get("start"), words[-1].get("end"))
        if any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times):
            raise ValueError("Invalid speech engine cue endpoints")
        start, end = (round(t * 1000) for t in times)
        cues.append(Cue(start, end, segment["text"].strip()))
    return cues


def _native_word_cues(payload: dict, duration: float) -> list[Cue]:
    """Split actual CrispASR words, never interpolate timings or discard text."""
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise ValueError("Invalid speech engine segments")
    cues = []
    previous_end = 0.0
    segment_texts = []
    for segment in payload["segments"]:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError("Invalid speech engine segment")
        text = segment["text"].strip()
        segment_texts.append(text)
        words = segment.get("words")
        if not text and not words:
            continue
        if not isinstance(words, list) or not words:
            raise ValueError("Speech engine did not return word timings")
        positions = [i for i, char in enumerate(text) if not char.isspace()]
        compact = "".join(text[i] for i in positions)
        consumed = 0
        first = last = None
        first_char = 0
        last_char = 0
        for word in words:
            if not isinstance(word, dict) or not isinstance(word.get("word"), str):
                raise ValueError("Invalid speech engine word")
            token = "".join(word["word"].split())
            if not token or not compact.startswith(token, consumed):
                raise ValueError("Speech engine word timings do not preserve the complete transcript")
            start, end = word.get("start"), word.get("end")
            if (any(type(t) not in (int, float) or not math.isfinite(t) for t in (start, end))
                    or start < previous_end or start < 0 or end <= start or end > duration
                    or round(end * 1000) > duration * 1000):
                raise ValueError("Speech engine returned invalid, overlapping, or out-of-bounds word timings")
            next_char = positions[consumed + len(token) - 1] + 1
            candidate = text[first_char:next_char].strip()
            lines = textwrap.wrap(candidate, width=42, break_long_words=False, break_on_hyphens=False)
            if first is not None and (end - first > 7 or len(candidate) > 84 or len(lines) > 2 or start - last > 0.8):
                cues.append(Cue(round(first * 1000), round(last * 1000), _wrap_cue(text[first_char:last_char])))
                first = None
                first_char = last_char
            if first is None:
                first = start
            if end - first > 7 or len(text[first_char:next_char].strip()) > 84:
                raise ValueError("Speech engine word cannot form a readable subtitle without invented timing")
            last, last_char, previous_end = end, next_char, end
            consumed += len(token)
        if consumed != len(compact):
            raise ValueError("Speech engine word timings do not preserve the complete transcript")
        cues.append(Cue(round(first * 1000), round(last * 1000), _wrap_cue(text[first_char:])))
    full_text = payload.get("text")
    if full_text is not None and (not isinstance(full_text, str)
            or "".join(full_text.split()) != "".join("".join(segment_texts).split())):
        raise ValueError("Speech engine segments do not preserve the complete transcript")
    return cues

def _canary_word_cues(payload: dict, duration: float) -> list[Cue]:
    """Group inseparable DTW cells; do not pretend these are CTC word onsets.

    Pinned canary.cpp:3162-3169 expands tied paths to one encoder frame.
    Its baked frontend uses floor(samples/160) mel frames, then three padded
    stride-2 convolutions: ceil(mel_frames/8) cells of 80 ms. Only the final
    padded cell is intersected with actual audio support, not arbitrary times.
    model_formats validates the corresponding 16 kHz/160-hop/8x profile.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise ValueError("Invalid speech engine segments")
    samples = round(duration * 16000)
    padded_end_ms = ((samples // 160 + 7) // 8) * 80
    physical_end = math.floor(duration * 1000) / 1000
    segments = []
    previous_start = previous_end = 0.0
    for segment in payload["segments"]:
        if not isinstance(segment, dict) or not isinstance(segment.get("words"), list):
            raise ValueError("Speech engine did not return word timings")
        clusters = []
        for word in segment["words"]:
            if not isinstance(word, dict) or not isinstance(word.get("word"), str):
                raise ValueError("Invalid speech engine word")
            start, end = word.get("start"), word.get("end")
            if (any(type(t) not in (int, float) or not math.isfinite(t) for t in (start, end))
                    or start < previous_start or end < previous_end or end <= start
                    or start < 0 or start >= physical_end
                    or any(abs(t * 1000 - round(t / 0.08) * 80) > 0.000001 for t in (start, end))
                    or previous_end - start > 0.080001):
                raise ValueError("Canary returned timings outside its monotone DTW frame contract")
            previous_start, previous_end = start, end
            if end > duration:
                if abs(end * 1000 - padded_end_ms) > 0.000001:
                    raise ValueError("Canary timestamp exceeds its final padded encoder cell")
                end = physical_end
            if clusters and start < clusters[-1]["end"]:
                clusters[-1]["end"] = end
                clusters[-1]["word"] += " " + word["word"]
            else:
                clusters.append({"word": word["word"], "start": start, "end": end})
        segments.append({**segment, "words": clusters})
    return _native_word_cues({**payload, "segments": segments}, duration)



def _wrap_cue(text: str) -> str:
    return "\n".join(textwrap.wrap(text.strip(), width=42, break_long_words=False, break_on_hyphens=False))



def _base_url(value: str) -> str:
    parts = urlsplit(value.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise EngineError("Engine URL must be an HTTP(S) URL without credentials, query, or fragment")
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


class Engine:
    def __init__(self, settings: dict, *, runtime="whisper.cpp", architecture="whisper"):
        if (runtime, architecture) not in {("whisper.cpp", "whisper"), ("crispasr", "parakeet"), ("crispasr", "canary")}:
            raise EngineError("Unsupported speech runtime or architecture")
        self.runtime, self.architecture = runtime, architecture
        self.url = _base_url(settings["engine_url"])
        self.timeout = httpx.Timeout(float(settings.get("inference_timeout_seconds", 3600)), connect=float(settings.get("connect_timeout_seconds", 10)))

    async def probe(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout.connect), trust_env=False) as client:
                response = await client.get(self.url + "/v1/models")
                response.raise_for_status()
                data = response.json()
                models = data.get("data")
                if not isinstance(models, list) or any(not isinstance(m, dict) or not isinstance(m.get("id"), str) for m in models):
                    raise ValueError("Invalid model list")
                return {"reachable": True, "models": [m["id"] for m in models], "error": None}
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            return {"reachable": False, "models": [], "error": f"Engine probe failed: {type(exc).__name__}"}

    async def _request(self, wav: bytes, model: str, fields: dict) -> httpx.Response:
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                response = await client.post(self.url + "/v1/audio/transcriptions", data={"model": model, **fields}, files={"file": ("audio.wav", wav, "audio/wav")})
                response.raise_for_status()
                return response
        except httpx.HTTPStatusError as exc:
            raise EngineError(f"Speech engine returned HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise EngineError(f"Speech engine request failed: {type(exc).__name__}") from exc

    async def transcribe(self, wav: bytes, model: str, language: str, task: str) -> str:
        if task not in {"transcribe", "translate"}:
            raise EngineError("Unsupported speech task")
        language = normalize_language(language)
        if self.runtime == "crispasr":
            if task == "translate":
                raise EngineError("Translated CrispASR subtitle timings have not been validated; use a translation-capable Whisper model")
            if self.architecture == "canary" and language == "auto":
                raise EngineError("Canary requires an explicit source language")
            fields = {"language": language, "response_format": "verbose_json",
                      "require_word_timestamps": "true", "no_auto_aligner": "true",
                      "lid_backend": "off", "translate": "false"}
            if self.architecture == "canary":
                fields.update(source_lang=language, target_lang=language)
        else:
            fields = {"language": language, "response_format": "verbose_json", "no_timestamps": "false", "token_timestamps": "true", "max_len": "84", "split_on_word": "true"}
            if task == "translate":
                fields["translate"] = "true"
        response = await self._request(wav, model, fields)
        try:
            with wave.open(io.BytesIO(wav), "rb") as source:
                frames, rate = source.getnframes(), source.getframerate()
            if self.runtime == "crispasr":
                adapter = _canary_word_cues if self.architecture == "canary" else _native_word_cues
                return render(adapter(response.json(), frames / rate), "srt")
            # Preserve Whisper segment text and aligned endpoints as before.
            cues = _aligned_cues(response.json())
            if cues and cues[-1].end_ms * rate > frames * 1000:
                # whisper.cpp may round the final end one 10 ms tick beyond
                # the supplied audio; never alter other invalid timings.
                duration_ms = frames * 1000 // rate
                final = cues[-1]
                if final.end_ms * rate - frames * 1000 > 10 * rate or final.start_ms >= duration_ms:
                    raise ValueError("Speech engine subtitle exceeds audio duration")
                cues[-1] = Cue(final.start_ms, duration_ms, final.text)
            return render(cues, "srt")
        except (ValueError, wave.Error, EOFError) as exc:
            raise EngineError(str(exc)) from exc

    async def detect(self, wav: bytes, model: str) -> dict:
        if self.runtime != "whisper.cpp":
            raise EngineError("This model cannot detect language; select a Whisper detection model")
        response = await self._request(wav, model, {"language": "auto", "detect_language": "true", "response_format": "verbose_json"})
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Invalid response")
            value = payload.get("detected_language") or payload.get("language")
            if not isinstance(value, str):
                raise ValueError("No detected language")
            code = normalize_language(value)
            if code == "auto":
                raise ValueError("No detected language")
        except (ValueError, TypeError) as exc:
            raise EngineError("Speech engine did not return a recognized detected language") from exc
        return {"language_code": code, "detected_language": LANGUAGES[code]}
