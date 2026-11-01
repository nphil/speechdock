# Changelog

## Unreleased

- Make the GitHub repository, release notes, Unraid template and GHCR image public. Verify anonymous access to both `latest` and `0.1.0`, so image pulls and update checks need no GitHub credentials.
- Point the Unraid template at its public source so native template metadata can refresh. Document movie and series Whisper fallback settings and the score thresholds that keep automatic generation behind ordinary providers.

## 0.1.0 — 2026-10-31

- Add a local speech workspace with activity history, audio/video uploads, subtitle previews and SRT/WebVTT/text downloads.
- Connect Bazarr's Whisper provider to the existing llama-swap engine, including real language detection, timed subtitles and separate English-translation model selection.
- Add a pinned Whisper model catalogue with streamed, size- and checksum-verified downloads, safe cancellation, restart recovery and app-owned model management.
- Validate and back up llama-swap configuration before applying owned definitions; require confirmation for engine-wide reloads and guard model launches with GPU-memory headroom.
- Persist settings and bounded speech work in SQLite; recover interrupted work, clean temporary audio and retain only app-owned history/files.
- Package a non-root CPU-only Unraid app with branding, managed-template metadata, private GHCR releases, image labels and a manually dispatched release pipeline.
