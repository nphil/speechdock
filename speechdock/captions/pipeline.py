"""Dub caption job: transcribe the English dub and write ``<video>.en.cc.srt`` beside the video.

Steps: pick the English audio stream; read the file once for that audio plus every embedded
English text track; transcribe in quiet-point windows; clean the transcript; score every
English subtitle (embedded or beside the video) against it and stop, writing nothing, if one
is already a real dub caption; otherwise repair names from those subtitles, build cues and
publish the file atomically - never over a file SpeechDock did not write.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

from ..audio import RATE, render
from ..engine import Word
from . import agreement, media, names, windows
from . import cues as cue_tools
from .ledger import Ledger, digest, publish, sidecar_for
from .subtitles import parse_subtitle

AGREEMENT_THRESHOLD = 0.75  # measured on this library: dub captions 0.91-0.95, translations 0.31-0.62
MIN_WORDS = 25  # fewer recognized words than this is not a speech track


class CaptionError(ValueError):
    """The job cannot continue; the message is shown in Activity."""


def _track_label(track: dict) -> str:
    title = media.tag(track, "title")
    return f"embedded track {track['index']}" + (f" \u201c{title}\u201d" if title else "")


def analyse(words: list[Word], sources: list[dict]) -> dict:
    """Clean the transcript, score each existing subtitle against it and prepare the cues.

    ``sources`` are ``{"meta": {...}, "cues": [SubCue]}``; the result holds the scores, the
    matched source (or None), glossary corrections and the rendered SRT.
    """
    cleaned, stats = cue_tools.clean_words(words)
    result = {"cleaned": len(cleaned), "stats": asdict(stats), "candidates": [], "matched": None,
              "corrections": [], "terms": [], "srt": "", "cues": 0}
    if len(cleaned) < MIN_WORDS:
        return result
    for source in sources:
        score = agreement.score(source["cues"], cleaned)
        result["candidates"].append({**source["meta"], "cues": len(source["cues"]), "agreement": round(score.agreement, 3),
                                     "precision": round(score.precision, 3), "recall": round(score.recall, 3)})
    best = max(result["candidates"], key=lambda item: item["agreement"], default=None)
    if best and best["agreement"] >= AGREEMENT_THRESHOLD:
        result["matched"] = best
    glossary = names.build_glossary([cue for source in sources for cue in source["cues"]], cleaned)
    corrected, fixes = names.correct_names(cleaned, glossary)
    cues = cue_tools.build_cues(corrected)
    result.update(corrections=[asdict(fix) for fix in fixes], terms=sorted(glossary.names.values())[:60],
                  srt=render(cues, "srt"), cues=len(cues))
    return result


async def run_caption_job(*, job: dict, progress, engine, ledger: Ledger, workdir: Path, roots: dict[str, str] | None = None) -> dict:
    started = time.monotonic()
    request = job["request"]
    force = bool(request.get("force"))
    try:
        video = media.resolve_media_path(request["path"], roots)
    except media.MediaError as exc:
        raise CaptionError(str(exc)) from exc
    sidecar = sidecar_for(video)

    previous = ledger.get(video)
    if previous and not force and ledger.reusable(previous, video, sidecar):
        outcome = "already_generated" if previous["outcome"] == "written" else previous["outcome"]
        details = {**previous["details"], "outcome": outcome, "cached": True}
        return {"result_text": None, "duration_seconds": (details.get("audio") or {}).get("duration_seconds"), "result_json": details}
    if not os.access(video.parent, os.W_OK | os.X_OK):
        raise CaptionError("SpeechDock cannot write to this media folder: mount the library read-write for the container user.")

    timings: dict[str, float] = {}
    mark = time.monotonic()

    def lap(name: str) -> None:
        nonlocal mark
        timings[name] = round(time.monotonic() - mark, 1)
        mark = time.monotonic()

    try:
        probed = await media.probe(video)
    except media.MediaError as exc:
        raise CaptionError(str(exc)) from exc
    lap("probe")
    audio = media.pick_english_audio(probed["streams"])
    result: dict = {"path": str(video), "file": video.name, "sidecar": sidecar.name, "force": force,
                    "threshold": AGREEMENT_THRESHOLD, "model": job["model"], "audio": None, "transcript": None,
                    "candidates": [], "matched": None, "glossary": None, "output": None, "note": None}

    def finish(outcome: str, text: str | None, duration: float | None, *, output: dict | None = None, **extra) -> dict:
        timings["total"] = round(time.monotonic() - started, 1)
        result.update(outcome=outcome, output=output, timings=timings, **extra)
        ledger.record(video, outcome, result, sidecar=sidecar if output else None,
                      sha256=output["sha256"] if output else None, job_id=job["id"])
        return {"result_text": text or None, "duration_seconds": duration, "result_json": result}

    if audio is None:
        result["note"] = "The file has no English audio stream (commentary is ignored)."
        return finish("no_english_audio", None, probed["duration"])
    duration = probed["duration"]
    result["audio"] = {"index": audio["index"], "language": media.language_of(audio), "title": media.tag(audio, "title"),
                       "codec": audio.get("codec_name"), "channels": audio.get("channels"),
                       "default": bool((audio.get("disposition") or {}).get("default")), "duration_seconds": duration}

    tracks = media.english_text_tracks(probed["streams"])
    work = workdir / job["id"]
    work.mkdir(parents=True, exist_ok=True)
    sources: list[dict] = []
    try:
        await progress(0.02, duration)

        async def extracting(fraction: float) -> None:
            await progress(0.02 + 0.33 * fraction, duration)

        try:
            pcm, files, note = await media.extract(video, audio, tracks, work, duration, extracting)
        except media.MediaError as exc:
            raise CaptionError(str(exc)) from exc
        result["note"] = note
        if not pcm.stat().st_size:
            raise CaptionError("The English audio stream contains no audio.")
        by_index = {track["index"]: track for track in tracks}
        for index, path in files.items():
            track = by_index[index]
            cues = parse_subtitle(path.read_bytes(), track["codec_name"])
            sources.append({"meta": {"source": "embedded", "stream": index, "title": media.tag(track, "title"),
                                     "codec": track["codec_name"], "label": _track_label(track)}, "cues": cues})
        lap("extract")
        duration = pcm.stat().st_size / (RATE * 2)
        bounds = await asyncio.to_thread(windows.window_bounds, pcm)
        words: list[Word] = []
        repairs = 0
        for number, (begin, end) in enumerate(bounds, 1):
            wav = await asyncio.to_thread(windows.read_window, pcm, begin, end)
            found, fixed = await engine.transcribe_words(wav, job["model"], "en")
            offset = begin / RATE
            words.extend(replace(word, start=word.start + offset, end=word.end + offset) for word in found)
            repairs += fixed
            await progress(0.35 + 0.55 * number / len(bounds), duration)
        lap("transcribe")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    for path in await asyncio.to_thread(media.sidecar_subtitles, video):
        if ledger.is_owned(path):
            continue  # our own earlier output is derived from the transcript: it proves nothing
        sources.append({"meta": {"source": "sidecar", "file": path.name, "label": f"file {path.name}"},
                        "cues": parse_subtitle(path.read_bytes(), path.suffix.lstrip("."))})

    found = await asyncio.to_thread(analyse, words, sources)
    lap("analyse")
    result["candidates"] = found["candidates"]
    result["transcript"] = {"windows": len(bounds), "words": found["cleaned"], "cues": found["cues"],
                            "dropped": found["stats"], "timing_repairs": repairs}
    result["glossary"] = {"terms": len(found["terms"]), "sample": found["terms"][:30], "corrections": found["corrections"]}
    await progress(0.95, duration)
    if not found["srt"]:
        result["note"] = "Too little intelligible speech was recognized in the English audio."
        return finish("no_speech", None, duration)
    if found["matched"]:
        result["matched"] = found["matched"]
        return finish("matched_existing", found["srt"], duration)

    exists = os.path.lexists(sidecar)
    if exists and not ledger.is_owned(sidecar):
        raise CaptionError(f"{sidecar.name} already exists and was not created by SpeechDock, so it was left untouched.")
    data = found["srt"].encode("utf-8")
    try:
        await asyncio.to_thread(publish, sidecar, data, replace=exists)
    except FileExistsError as exc:
        raise CaptionError(f"{sidecar.name} appeared while the job ran and was left untouched.") from exc
    except OSError as exc:
        raise CaptionError(f"Could not write {sidecar.name}: {exc.strerror or type(exc).__name__}") from exc
    output = {"file": sidecar.name, "bytes": len(data), "cues": found["cues"], "sha256": digest(sidecar), "replaced": exists}
    await progress(1.0, duration)
    return finish("written", found["srt"], duration, output=output)
