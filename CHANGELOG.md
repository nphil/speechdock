# Changelog

## 0.2.0 — 2026-11-01

- Import supported speech-model files by upload or direct public HTTPS URL, with optional expected SHA-256 and size, bounded streaming transfers, cancellation, restart recovery and explicit native registration. Preserve unrelated models and never load a model simply because it was imported.
- Identify supported Whisper GGML and legacy Parakeet/Canary GGUF formats from their contents; reject unsupported architectures and malformed files. Distinguish a supplied checksum match from a locally calculated hash.
- Package a pinned CUDA 12 CrispASR runtime for on-demand execution inside the existing llama-swap container, with verified immutable assets and a fail-closed GPU launcher. Parakeet uses its native word timing; Canary transcription uses explicitly labelled DTW estimates. Parakeet CUDA execution is verified on the Tesla P40; model quality and speed must be judged on representative speech, not hardware compatibility alone. Canary translation is not enabled.
- Keep language detection on its own compatible model when choosing a transcription backend without language identification.
- Add optional English CPU forced alignment for Marquee using exact selected Plex media/audio/subtitle identities, read-only server-side media mappings and a pinned ONNX model. Preserve authored cue bounds and retain estimated reveal for unsupported or mismatched speech; reject obsolete playback and seek generations. This source integration has local fixture proof, not production Plex or TV deployment proof.
- Clarify that a verified source revision does not imply acoustic alignment: consumers must honor each cue’s status and discard old word timing on estimated fallback or source change.

## 0.1.3 — 2026-11-01

- Accept valid subtitle cues when Whisper’s unused internal subword estimates overlap or run backward. Validate the actual cue endpoints, preserve every recognized segment and keep strict final subtitle and physical-audio timing checks.

## 0.1.2 — 2026-11-01

- Use Whisper’s aligned word/token endpoints for subtitle cue boundaries instead of coarse predicted segment times. Fix full-length movie transcription stopping on an overlong segment without dropping dialogue or widening the physical-audio timing guard.
- Preserve complete segment text, including punctuation and zero-duration internal tokens, while rejecting malformed, overlapping or out-of-bounds alignment.
- Document the exact shared SQLite file permissions required when llama-swap runs as root and SpeechDock runs as UID/GID 1000. Keep the database owner, contents and unrelated appdata permissions intact.

## 0.1.1 — 2026-11-01

- Detect speech with a checksum-pinned CPU Silero model before sending bounded audio windows to Whisper. Preserve original movie timestamps and return empty subtitles for genuine silence rather than hallucinated speech.
- Handle the observed one-tick final timestamp rounding at the exact supplied-audio boundary; keep invalid, overlapping and zero-duration subtitle rejection.
- Expose the real mounted llama-swap configuration at its engine runtime path inside the image so its exact validator can check model registration without a false missing-store-directory error.
- Make the GitHub repository, release notes, Unraid template and GHCR image public. Verify anonymous access to both `latest` and `0.1.0`, so image pulls and update checks need no GitHub credentials.
- Point the Unraid template at its public source so native template metadata can refresh. Document movie and series Whisper fallback settings and the score thresholds that keep automatic generation behind ordinary providers.

## 0.1.0 — 2026-10-31

- Add a local speech workspace with activity history, audio/video uploads, subtitle previews and SRT/WebVTT/text downloads.
- Connect Bazarr's Whisper provider to the existing llama-swap engine, including real language detection, timed subtitles and separate English-translation model selection.
- Add a pinned Whisper model catalogue with streamed, size- and checksum-verified downloads, safe cancellation, restart recovery and app-owned model management.
- Validate and back up llama-swap configuration before applying owned definitions; require confirmation for engine-wide reloads and guard model launches with GPU-memory headroom.
- Persist settings and bounded speech work in SQLite; recover interrupted work, clean temporary audio and retain only app-owned history/files.
- Package a non-root CPU-only Unraid app with branding, managed-template metadata, private GHCR releases, image labels and a manually dispatched release pipeline.
