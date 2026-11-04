# Changelog

## 0.3.1 — 2026-11-04

- Wait for a restarting engine instead of failing jobs. Two bursts of 10-20 queued jobs failed within one second ("Model … is not available" or "Speech engine returned HTTP 502"): once when the host rebooted and SpeechDock came up before its reverse proxy and llama-swap, once when llama-swap's container was recreated by an image update. A request that meets a refused or dropped connection or HTTP 502/503/504 now waits with growing pauses for up to 15 minutes (then fails naming the last reason); the waiting job keeps the head of the queue, so the queue behind it is not failed with it. Requests and caption batches are accepted while the engine is down; a model the engine lists as missing and other HTTP errors still fail at once. This covers every caller: Bazarr `/asr`, uploads and captions.
- Re-queue interrupted work. A job running when SpeechDock stops (shutdown or crash) goes back in the queue with its audio instead of failing as "interrupted by a restart"; after three interruptions it fails with that explanation. A cancel that was still pending stays cancelled.
- Recover windows whose timings fail the strict checks. On five failing Bazarr episodes 20 of 371 windows were rejected (a word with no duration, words piled on the last frame, a word starting 40 ms before the previous one ends, punctuation listed after the next word, a word stretched over 10 seconds), always identically, so retrying never helped. The same audio is now cut at quiet points near its middle and every piece goes through the same checks; only if no cut works does the full Whisper translation model get the window, and a window that still cannot be recovered fails the job naming its place in the recording. All 87 rejected windows found on those episodes recovered by cutting. Each recovered window is listed as `recovery` in the job detail. The checks themselves are unchanged.
- Decode xHE-AAC and report real decoder errors. The image used Debian 12's FFmpeg 5.1, which cannot open xHE-AAC (USAC, audio object type 42); ffprobe then exited 1 without a word and the job error read "Could not read the video: ". The image now builds FFmpeg 8.1.3 from the official source archive (pinned by SHA-256), and ffprobe/ffmpeg failures include the exit code and the real stderr (at warning level, where FFmpeg explains unsupported audio).
- Automatic captions only for dubs. The Sonarr/Radarr webhook now queues a video only if it has English audio and at least one audio track in another language; others are listed as `skipped` in the reply and nothing is recorded for them. Manual requests are unfiltered. The setting stays off by default.

## 0.3.0 — 2026-11-03

- Add dub captions. A job takes a video inside the media mounts, picks its English audio stream (the default English stream, never commentary), transcribes it with the configured transcription model in windows cut at quiet points, cleans the transcript (looped repetitions, words squeezed into the last frames of a window and low-confidence stretches such as singing are dropped), repairs misspelled names from the same file's English subtitle, builds cues of at most two 42-character lines at a 20 characters-per-second reading speed and publishes `<video base>.en.cc.srt` atomically beside the video, which Plex lists as English (CC).
- Skip real captions. Every English subtitle, embedded or beside the video, is scored against the transcript (F1 of how many of its words are heard near their cue and how many spoken words it carries). A track at 0.75 or better is a real dub caption: nothing is written and the matching track and scores are recorded. Measured on real episodes, dub captions scored 0.91-0.95 and translation tracks 0.31-0.62; a truncated or out-of-sync track also fails. English-original shows with a foreign dub are skipped by the same test.
- Never overwrite what SpeechDock did not write. Each written file is recorded with its checksum; a file it did not create, or one edited afterwards, is never replaced, even when forced. A rerun keeps SpeechDock's own file unless forced. Earlier skip decisions are reused until the video changes. `GET /api/captions` lists the decisions.
- Add `POST /api/captions`, `POST /api/captions/batch` and a Sonarr/Radarr webhook (`POST /api/captions/webhook`, On Import and On Upgrade) with an auto-captions switch in Settings, off by default. Caption jobs show in Activity with their report and run serially behind Bazarr and uploaded jobs; a caption batch never uses up their queue limit.
- Read Parakeet tokens tolerantly for captions: overlapping or empty token intervals, which fail a whole window in the strict subtitle path used for Bazarr, are repaired and counted. The strict path is unchanged.
- The Unraid template mounts the media folders read-write so captions can be created beside the videos.

## 0.2.2 — 2026-11-02

- Preserve quiet film dialogue by sending every non-digitally-silent, bounded audio window for transcription and translation. Silero remains a language-detection aid, not a filter that can silently discard subtitle audio.
- Read Parakeet native lexical token intervals through the model-specific llama-swap route. Preserve the complete transcript and punctuation without letting a delayed punctuation token turn a short word into a 16-second subtitle; keep strict completeness, timing, physical-audio and readability checks. No speech runtime rebuild or additional model is required.
- Allow an owned model registration to explicitly retain its resident policy with `keep_loaded`. A warm member uses TTL 0 in a nonexclusive, persistent, nonswapping speech group; other members retain their idle timeout. Confirm configuration reloads using a fresh engine-visible revision, preserve unrelated models and roll back unacknowledged changes.
- The retained 888.064-second Sintel prepared-audio check produced a complete 57-cue SRT from 33 real native requests. This verifies delivery and timing bounds, not a perfect transcript or independently measured acoustic accuracy.

## 0.2.1 — 2026-11-01

- Preserve the raw independent CTC token sequence for acoustic scoring. Previously, normalized boundary or repeated separators could be discarded, falsely rejecting valid alignment as weak audio. Exact independent lexical equality and all token-score, source, hash, bounds and playback/seek epoch guards remain unchanged.
- On the same 19 retained film windows, aligned results increased from 1 to 2; the other 17 remain estimated due to lexical mismatch. This does not expand audio context, change translation or model defaults, or replace the CPU known-text aligner with the selected speech-to-text model.

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
