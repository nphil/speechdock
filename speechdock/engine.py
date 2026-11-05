"""Clients and timing adapters for shared Whisper and pinned CrispASR runtimes."""

import asyncio
import io
import logging
import math
import textwrap
import time
import unicodedata
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from .audio import Cue, parse_srt, render
from .languages import LANGUAGES, normalize_language


class EngineError(ValueError):
    """An actionable engine connectivity or response error."""


class EngineUnavailable(EngineError):
    """The engine is down, reloading or not ready yet; asking again later is expected to work."""


class ResponseRejected(EngineError):
    """The engine answered, but its reply failed SpeechDock's checks (timings, completeness, format).
    Asking again about the very same audio fails the same way; cutting the audio differently may not."""


log = logging.getLogger("speechdock.engine")

ENGINE_WAIT_SECONDS = 900.0  # how long one request waits for a restarting or reloading engine
RETRY_FIRST_SECONDS = 1.0
RETRY_CAP_SECONDS = 15.0
# 502/503/504 is what the reverse proxy or llama-swap answers while the engine restarts, reloads or loads a model.
TRANSIENT_STATUS = frozenset({502, 503, 504})
TRANSIENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError, httpx.WriteError,
                    httpx.RemoteProtocolError, httpx.PoolTimeout)


class EngineWait:
    """Bounded, growing pauses while the engine is not there; one instance per outage.

    ``pause`` sleeps and returns, or raises ``EngineUnavailable`` once the limit is spent, so a job
    waits for a restart instead of failing, and a dead engine still ends the job in the end. The
    optional ``on_wait(reason, waited_seconds)`` hook is awaited before every sleep (jobs use it
    to notice a cancellation).
    """

    def __init__(self, limit: float, first: float, cap: float,
                 on_wait: Callable[[str, float], Awaitable[None]] | None = None):
        self.limit, self.first, self.cap, self.on_wait = limit, first, cap, on_wait
        self.started: float | None = None
        self.attempt = 0

    async def pause(self, reason: str) -> None:
        now = time.monotonic()
        if self.started is None:
            self.started = now
            log.warning("Speech engine unavailable (%s); waiting up to %g s for it to come back", reason, self.limit)
        waited = now - self.started
        delay = min(self.cap, self.first * 2 ** self.attempt)
        if waited + delay > self.limit:
            span = f"{self.limit / 60:g} min" if self.limit >= 60 else f"{self.limit:g} s"
            raise EngineUnavailable(f"The speech engine was not available for {span} ({reason})")
        self.attempt += 1
        if self.on_wait is not None:
            await self.on_wait(reason, waited)
        await asyncio.sleep(delay)

    def recovered(self) -> None:
        if self.started is not None:
            log.warning("Speech engine is back after %.0f s", time.monotonic() - self.started)
            self.started, self.attempt = None, 0


def make_wait(settings: dict, on_wait: Callable[[str, float], Awaitable[None]] | None = None) -> EngineWait:
    """A bounded wait for one outage. ``engine_wait_seconds`` / ``engine_retry_seconds`` are not saved
    settings; they exist so tests can shorten the waits."""
    first = float(settings.get("engine_retry_seconds", RETRY_FIRST_SECONDS))
    return EngineWait(float(settings.get("engine_wait_seconds", ENGINE_WAIT_SECONDS)), first, max(first, RETRY_CAP_SECONDS), on_wait)


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

def _parakeet_token_cues(payload: dict, duration: float) -> list[Cue]:
    """Use native lexical-token bounds while retaining every transcript character."""
    if (not isinstance(payload, dict) or payload.get("backend") != "parakeet"
            or not isinstance(payload.get("text"), str) or not isinstance(payload.get("segments"), list)):
        raise ValueError("Invalid native Parakeet response")
    segments = []
    for segment in payload["segments"]:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError("Invalid native Parakeet segment")
        text = segment["text"]
        tokens = segment.get("tokens")
        if tokens is None and not text.strip():
            tokens = []
        if not isinstance(tokens, list):
            raise ValueError("Native Parakeet segment is missing tokens")
        words = []
        current = ""
        leading = ""
        first = last = None
        boundary = False
        token_text = []
        for token in tokens:
            if not isinstance(token, dict) or not isinstance(token.get("text"), str):
                raise ValueError("Invalid native Parakeet token")
            raw = token["text"]
            if not raw:
                continue
            if raw.isspace():
                boundary = True
                continue
            if raw.startswith(" "):
                boundary = True
                raw = raw[1:]
            if not raw or any(char.isspace() for char in raw):
                raise ValueError("Invalid native Parakeet token text")
            token_text.append(raw)
            start, end = token.get("t0"), token.get("t1")
            if (any(type(t) not in (int, float) or not math.isfinite(t) for t in (start, end))
                    or start < 0 or end < start or end > duration * 1000):
                raise ValueError("Invalid native Parakeet token timings")
            punct = all(unicodedata.category(char).startswith("P") for char in raw)
            if punct:
                opening = all(unicodedata.category(char) in {"Ps", "Pi"} or char in "\"'" for char in raw)
                if first is None or leading or (boundary and opening):
                    leading += raw
                else:
                    current += raw
                continue
            if boundary and first is not None:
                words.append({"word": current, "start": first / 1000, "end": last / 1000})
                current = ""
                first = last = None
            current += leading + raw
            leading = ""
            if first is None:
                first = start
            elif start < last:
                raise ValueError("Invalid native Parakeet token timings")
            last = end
            boundary = False
        if leading:
            if first is None:
                raise ValueError("Native Parakeet punctuation has no lexical timing")
            current += leading
        if token_text and first is None:
            raise ValueError("Native Parakeet punctuation has no lexical timing")
        if first is not None:
            words.append({"word": current, "start": first / 1000, "end": last / 1000})
        if "".join(token_text) != "".join(text.split()):
            raise ValueError("Native Parakeet tokens do not preserve the complete transcript")
        segments.append({"text": text, "words": words})
    return _native_word_cues({"text": payload["text"], "segments": segments}, duration)


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



@dataclass(frozen=True)
class Word:
    """One recognized word: engine-reported onset/end in seconds, and its confidence."""
    start: float
    end: float
    text: str
    confidence: float = 1.0


def _parakeet_words(payload: dict, duration: float) -> tuple[list[Word], int]:
    """Read native Parakeet tokens as timed words for caption building.

    ``_parakeet_token_cues`` rejects a whole window when one token interval is empty or
    overlapping, which suits Bazarr's strict contract but fails real episodes. Captions
    need the words, so text is never altered and unusable intervals are repaired to a
    monotone sequence inside the audio; each repair is counted. Punctuation stays in
    the text but never sets a word's timing.
    """
    if (not isinstance(payload, dict) or payload.get("backend") != "parakeet"
            or not isinstance(payload.get("segments"), list)):
        raise ValueError("Invalid native Parakeet response")
    found = []
    for segment in payload["segments"]:
        tokens = segment.get("tokens") if isinstance(segment, dict) else None
        if tokens is None and isinstance(segment, dict) and not str(segment.get("text", "")).strip():
            continue
        if not isinstance(tokens, list):
            raise ValueError("Native Parakeet segment is missing tokens")
        word, leading, boundary = None, "", True
        for token in tokens:
            if not isinstance(token, dict) or not isinstance(token.get("text"), str):
                raise ValueError("Invalid native Parakeet token")
            raw = token["text"]
            if raw.isspace():
                boundary = True
                continue
            if raw.startswith(" "):
                boundary = True
            raw = "".join(raw.split())
            if not raw:
                continue
            if all(unicodedata.category(char).startswith("P") for char in raw):
                opening = all(unicodedata.category(char) in {"Ps", "Pi"} or char in "\"'" for char in raw)
                if word is None or leading or (boundary and opening):
                    leading += raw
                else:
                    word["text"] += raw
                continue
            if boundary and word is not None:
                found.append(word)
                word = None
            start, end = token.get("t0"), token.get("t1")
            if any(type(t) not in (int, float) or not math.isfinite(t) for t in (start, end)):
                start = end = word["end"] if word else (found[-1]["end"] if found else 0)
            score = token.get("confidence")
            score = float(score) if type(score) in (int, float) and math.isfinite(score) else 1.0
            if word is None:
                word = {"text": "", "start": start, "end": end, "scores": []}
            word["text"] += leading + raw
            word["start"], word["end"] = min(word["start"], start), max(word["end"], end)
            word["scores"].append(score)
            leading, boundary = "", False
        if word is not None:
            found.append(word)
    words, repairs, previous_end = [], 0, 0.0
    for item in found:
        start = min(max(item["start"] / 1000, previous_end), duration)
        end = min(max(item["end"] / 1000, start + 0.04), duration)
        repairs += abs(start - item["start"] / 1000) > 0.001 or abs(end - item["end"] / 1000) > 0.001
        words.append(Word(start, end, item["text"], sum(item["scores"]) / len(item["scores"])))
        previous_end = end
    return words, repairs


def words_from_cues(cues: list[Cue]) -> list[Word]:
    """Spread each cue's words across its interval, for models without word timing."""
    words = []
    for cue in cues:
        parts = cue.text.split()
        total = sum(len(part) + 1 for part in parts)
        position = cue.start_ms
        for part in parts:
            end = position + (cue.end_ms - cue.start_ms) * (len(part) + 1) / total
            words.append(Word(position / 1000, end / 1000, part))
            position = end
    return words


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
    def __init__(self, settings: dict, *, runtime="whisper.cpp", architecture="whisper",
                 on_wait: Callable[[str, float], Awaitable[None]] | None = None):
        if (runtime, architecture) not in {("whisper.cpp", "whisper"), ("crispasr", "parakeet"), ("crispasr", "canary")}:
            raise EngineError("Unsupported speech runtime or architecture")
        self.runtime, self.architecture = runtime, architecture
        self.url = _base_url(settings["engine_url"])
        self.timeout = httpx.Timeout(float(settings.get("inference_timeout_seconds", 3600)), connect=float(settings.get("connect_timeout_seconds", 10)))
        self.wait_settings = {key: settings[key] for key in ("engine_wait_seconds", "engine_retry_seconds") if key in settings}
        self.on_wait = on_wait

    def waiter(self) -> EngineWait:
        """A fresh bounded wait for one outage, with this engine's limits and hook."""
        return make_wait(self.wait_settings, self.on_wait)

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
        """POST one window. A restarting or reloading engine (connection refused or dropped,
        HTTP 502/503/504) is waited for with growing pauses up to the wait limit, then
        ``EngineUnavailable``; any other failure is an immediate ``EngineError``."""
        path = ("/upstream/" + quote(model, safe="") + "/inference"
                if (self.runtime, self.architecture) == ("crispasr", "parakeet")
                else "/v1/audio/transcriptions")
        wait = self.waiter()
        while True:
            try:
                async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                    response = await client.post(self.url + path, data={"model": model, **fields}, files={"file": ("audio.wav", wav, "audio/wav")})
                    response.raise_for_status()
                wait.recovered()
                return response
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in TRANSIENT_STATUS:
                    raise EngineError(f"Speech engine returned HTTP {status}") from exc
                reason = f"HTTP {status}"
            except TRANSIENT_ERRORS as exc:
                reason = type(exc).__name__
            except httpx.HTTPError as exc:
                raise EngineError(f"Speech engine request failed: {type(exc).__name__}") from exc
            await wait.pause(reason)

    def _crisp_fields(self, language: str) -> dict:
        fields = {"language": language, "response_format": "verbose_json",
                  "require_word_timestamps": "true", "no_auto_aligner": "true",
                  "lid_backend": "off", "translate": "false"}
        if self.architecture == "canary":
            fields.update(source_lang=language, target_lang=language)
        return fields

    async def transcribe_words(self, wav: bytes, model: str, language: str) -> tuple[list[Word], int]:
        """Recognize one bounded window as timed words, in seconds from the window start.

        Parakeet gives real word timing (repaired where unusable, see ``_parakeet_words``).
        Other models fall back to their cue-level subtitles with word times spread over each
        cue. The second value counts repaired intervals.
        """
        if (self.runtime, self.architecture) != ("crispasr", "parakeet"):
            return words_from_cues(parse_srt(await self.transcribe(wav, model, language, "transcribe"))), 0
        response = await self._request(wav, model, self._crisp_fields(normalize_language(language)))
        try:
            with wave.open(io.BytesIO(wav), "rb") as source:
                duration = source.getnframes() / source.getframerate()
            return _parakeet_words(response.json(), duration)
        except (ValueError, wave.Error, EOFError) as exc:
            raise EngineError(str(exc)) from exc

    async def transcribe(self, wav: bytes, model: str, language: str, task: str) -> str:
        if task not in {"transcribe", "translate"}:
            raise EngineError("Unsupported speech task")
        language = normalize_language(language)
        if self.runtime == "crispasr":
            if task == "translate":
                raise EngineError("Translated CrispASR subtitle timings have not been validated; use a translation-capable Whisper model")
            if self.architecture == "canary" and language == "auto":
                raise EngineError("Canary requires an explicit source language")
            fields = self._crisp_fields(language)
        else:
            fields = {"language": language, "response_format": "verbose_json", "no_timestamps": "false", "token_timestamps": "true", "max_len": "84", "split_on_word": "true"}
            if task == "translate":
                fields["translate"] = "true"
        response = await self._request(wav, model, fields)
        try:
            with wave.open(io.BytesIO(wav), "rb") as source:
                frames, rate = source.getnframes(), source.getframerate()
            if self.runtime == "crispasr":
                adapter = _canary_word_cues if self.architecture == "canary" else _parakeet_token_cues
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
            raise ResponseRejected(str(exc)) from exc

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
            probability = payload.get("detected_language_probability")
            probability = round(float(probability), 4) if isinstance(probability, (int, float)) and 0 <= probability <= 1 else None
        except (ValueError, TypeError) as exc:
            raise EngineError("Speech engine did not return a recognized detected language") from exc
        result = {"language_code": code, "detected_language": LANGUAGES[code]}
        if probability is not None:
            result["probability"] = probability  # whisper.cpp reports it; other servers may not
        return result
