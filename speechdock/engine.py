"""Client for the shared llama-swap / whisper-server instance."""

import io
import math
import wave
from urllib.parse import urlsplit, urlunsplit

import httpx

from .audio import Cue, render
from .languages import LANGUAGES, normalize_language


class EngineError(ValueError):
    """An actionable engine connectivity or response error."""


def _aligned_cues(payload: dict) -> list[Cue]:
    """Use acoustic token endpoints, not predicted segment timestamp tokens."""
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise ValueError("Invalid speech engine segments")
    cues = []
    for segment in payload["segments"]:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError("Invalid speech engine segment")
        words = segment.get("words")
        if not isinstance(words, list) or not words:
            raise ValueError("Speech engine did not return aligned word timings")
        previous_end = 0
        for index, word in enumerate(words):
            if not isinstance(word, dict):
                raise ValueError("Invalid speech engine word timings")
            times = (word.get("start"), word.get("end"))
            if any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times):
                raise ValueError("Invalid speech engine word timings")
            start, end = (round(t * 1000) for t in times)
            # Punctuation and subword tokens can have zero duration. Keep their
            # text in the segment, but require ordered, nonnegative alignment.
            if start < previous_end or end < start:
                raise ValueError("Invalid or overlapping speech engine word timings")
            if index == 0:
                first_start = start
            previous_end = end
        cues.append(Cue(first_start, previous_end, segment["text"].strip()))
    return cues


def _base_url(value: str) -> str:
    parts = urlsplit(value.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise EngineError("Engine URL must be an HTTP(S) URL without credentials, query, or fragment")
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


class Engine:
    def __init__(self, settings: dict):
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
        fields = {"language": normalize_language(language), "response_format": "verbose_json", "no_timestamps": "false", "token_timestamps": "true", "max_len": "84", "split_on_word": "true"}
        if task == "translate":
            fields["translate"] = "true"
        response = await self._request(wav, model, fields)
        try:
            # Segment timestamps are model predictions and can extend past the
            # WAV even with token_timestamps enabled. The words carry the actual
            # alignment; retain every segment's text and use those endpoints.
            cues = _aligned_cues(response.json())
            with wave.open(io.BytesIO(wav), "rb") as source:
                frames, rate = source.getnframes(), source.getframerate()
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
