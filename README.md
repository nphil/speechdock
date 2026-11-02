# SpeechDock

A local speech-to-subtitles workspace for Bazarr and other clients. SpeechDock handles uploads, timed subtitle files, activity history and speech-model management; your existing llama-swap service runs the models.

One CPU-only app container. The existing llama-swap service launches model servers on demand; SpeechDock itself needs neither GPU access, a Docker socket nor a production Node.js runtime.

Version 0.2.2 preserves quiet subtitle audio and consumes Parakeet’s native lexical token intervals without stretching words to delayed punctuation. Owned registrations can retain a resident English model while other speech models remain on demand. The optional CPU known-text aligner is separate from speech recognition; backend or local client proof does not establish actual TV consumption.

## Use it

- **Activity:** inspect queued, running and finished work; preview and download SRT, WebVTT or plain text.
- **New transcription:** upload audio or video, keep its spoken language or translate it into English.
- **Models:** download checksum-verified Whisper models, apply them to llama-swap, and remove only SpeechDock-owned models.
- **Settings:** choose the engine, transcription and translation models, language, timeouts, upload limit and history retention.

For a compact setup, keep Parakeet Q5 resident for qualified English voice/subtitle work and full Whisper Large-v3 Q5 on demand for multilingual transcription, English translation and language detection. Choose registered models in Settings; an engine alias alone does not mean a model is installed. Whisper Turbo cannot translate. SpeechDock never silently substitutes Turbo for a translation model or a transcription backend for the separately selected detection model.

## Unraid installation

Use [`unraid/speechdock.xml`](unraid/speechdock.xml). The image is `ghcr.io/nphil/speechdock:latest`. The repository, release notes, template and GHCR image are public; Unraid can pull the image and check for updates without GitHub credentials.

| Container path | Host path | Access |
| --- | --- | --- |
| `/data` | `/mnt/nvme/appdata/speechdock/data` | Read/write, UID/GID 1000 |
| `/models-owned` | `/mnt/nvme/appdata/llama-models/whisper/speechdock` | Read/write, UID/GID 1000 |
| `/llama-config` | `/mnt/nvme/appdata/llama-swap` | Read/write directory containing `config.yaml` |
| `/validator/llama-swap` | `/mnt/nvme/appdata/speechdock/tools/llama-swap` | Read-only executable file |

The existing llama-swap mount exposes the owned-model folder as `/models/whisper/speechdock`. Do not mount a different folder on one side: downloading a model would succeed but the engine could not open it.

The validator must be an executable copy of **the exact llama-swap binary deployed on that host**, not a guessed version. Refresh the copy after upgrading llama-swap. Configuration updates use same-directory temporary files, validation, backups and atomic replacement. Both the configuration directory and file need compatible UID/GID 1000 permissions; back up their original ownership and modes before preparing them. Do not recursively change ownership of unrelated models or app data.

The image maps the engine’s standard `/etc/llama-swap/config` path to the same `/llama-config` mount. The validator therefore checks the real store location, not a fabricated temporary database. If an existing store file is not writable by UID/GID 1000, registration still fails safely; no permission changes are made automatically.

When llama-swap runs as root, keep the existing database owner and grant SpeechDock’s shared group write access to that one file. For this host’s appdata layout:

```sh
chgrp 1000 /mnt/nvme/appdata/llama-swap/store.sqlite
chmod g+w /mnt/nvme/appdata/llama-swap/store.sqlite
```

Record the original group and mode first. Do not delete or recreate the database, grant world-write access, or recursively change appdata permissions to make registration pass. The configuration directory must already allow UID/GID 1000 to create the validator’s temporary configuration files.

The template maps port `3777` to `8000`. Install and recreate through Unraid's native Docker template path (`CreateDocker.php` with `updateContainer`), not `rebuild_container` or an ad-hoc `docker run` that loses managed-app metadata. A restart alone does not install a new image.

Publish internally through HomeLabber at `https://speechdock.nateshome.net`, upstream `http://192.168.1.69:3777`. Do not add per-container Tailscale. The application has no login and belongs on a trusted LAN/tailnet, not a public route.

### Reverse proxy

The UI rejects cross-origin browser changes. Uvicorn must trust the actual reverse-proxy peer before accepting its forwarded HTTPS scheme. The Unraid template provides `FORWARDED_ALLOW_IPS` for loopback and this installation's Caddy address, `192.168.1.71`. Verify the address seen by the container and change that exact list if the network differs. Never fix a proxy problem by trusting every address or disabling the origin check.

## Connect Bazarr

In **Settings → Providers → Whisper**:

| Field | Value |
| --- | --- |
| Whisper ASR Docker Endpoint | `https://speechdock.nateshome.net` |
| Connection/response timeout | `10` seconds |
| Transcription/translation timeout | `3600` seconds |
| Logging level | `INFO` |

Use **Test Connection**, enable the provider and save. Keep ordinary subtitle providers enabled. Enable Bazarr’s native Whisper fallback for both movies (`use_whisper_fallback`) and series (`use_whisper_fallback_series`); provider order alone does not make Whisper a backup.

Keep the normal subtitle-score thresholds above Whisper’s generated-subtitle score. With Bazarr’s standard 90% series / 70% movie thresholds, Whisper cannot pass the ordinary search and runs when that search finds nothing. Lowering thresholds, manually generating subtitles, or forcing an upgrade can bypass that automatic-fallback behavior. SpeechDock does not replace Bazarr’s search policy.

SpeechDock implements the Whisper ASR protocol Bazarr uses: `GET /status`, `POST /detect-language` and `POST /asr`. For `encode=false`, `audio_file` must be raw signed 16-bit little-endian PCM, 16 kHz, mono. Normal web/API uploads accept media files and normalize them with FFmpeg. `video_file` is logging metadata, never a server filesystem path.

`task=translate` means translation **into English**; the `language` query names the spoken source language, not an arbitrary translation target.

## Model safety

- The official Whisper catalogue pins exact revisions, sizes and SHA-256 hashes. Manual imports additionally accept supported files or direct public HTTPS links, with optional expected hash and size; a calculated hash alone does not establish who published a model.
- Existing shared models, including the `whisper-1` Turbo alias, are read-only. SpeechDock edits only its own `speechdock-*` definitions and `speechdock` group, preserving unrelated YAML and detecting conflicting edits.
- **Applying or removing model definitions reloads llama-swap and interrupts its loaded models.** The UI requires a separate confirmation and keeps a configuration backup. The engine must be running with configuration watching enabled.
- Loading a managed speech model can evict warmed chat models. A free-GPU-memory launch guard reserves conservative headroom; its thresholds are not measured performance or memory benchmarks. Other GPU services retain priority.
- Model-file size is not GPU-memory consumption. Full Large-v3 is about 3.1 GB on disk, and has a higher memory requirement than its quantized variants.

### Manual imports and on-demand runtimes

In **Models → Import a model**, choose a local file or a direct download URL. Supported formats are Whisper GGML and the validated legacy Parakeet/Canary GGUF variants supported by the pinned CrispASR runtime—not arbitrary chat-model GGUF, Safetensors, archives or newer incompatible speech architectures. Files are streamed into private staging, limited to 8 GiB and checked structurally before publication. URL downloads allow only public HTTPS on port 443, validate and pin DNS addresses, and recheck redirects. Signed URLs stay transient; only a source hostname and optional display filename are saved.

Supplying a publisher SHA-256 gives a **checksum match**. Without it, the UI says the checksum was calculated, not independently verified. Unknown transfer lengths show indeterminate progress. Cancellation and interrupted-start recovery never publish partial files. Removing an import affects only its verified owned file and record, not the original upload or media library.

Importing does **not** register or load a model. Review its detected format, runtime and task capabilities, then explicitly register it. The packaged CrispASR CUDA 12 runtime is copied atomically to an immutable versioned directory beneath the existing shared owned-model mount; no new container, port or always-running service is added. The GPU launcher fails rather than silently running on the CPU when CUDA is unavailable.

Parakeet subtitle transcription reads explicit lexical token intervals from the pinned runtime’s native endpoint. These remain model estimates, not independently measured word onsets. Canary uses source-validated DTW estimates and requires an explicit spoken language. Neither backend supplies verified language detection here; Canary translation is disabled. Full Whisper remains available for multilingual transcription, language detection and translation into English. Translating existing subtitle text into other languages is a separate text-translation task, such as Lingarr; it does not repair recognition or source timing.

### Local English voice deployment

The two-model deployment uses one SpeechDock-managed Parakeet Q5 definition for English voice and English subtitle generation, rather than keeping a separate `parakeet-voice` definition of the same weights. Point the existing `wyoming-openai` bridge’s `STT_MODELS` and SpeechDock’s English transcription selection at that canonical registered model. No extra app container or duplicate model download is needed.

Register the owned Parakeet model with `confirm_reload: true, keep_loaded: true`, then use its same canonical ID in the startup preload hook and consumers while preserving unrelated preload entries. Explicit `keep_loaded: false` restores the 180-second idle timeout; omitting the field preserves the existing policy. A warm member makes the `speechdock` group nonexclusive, persistent and nonswapping, while cold members keep their own idle timeout. This prevents normal idle unloading/eviction, not explicit unloads, process failure or engine reloads. The fresh revision acknowledgement proves the active configuration, not loaded readiness or audio inference. Startup preload is asynchronous; its generic `GET /` may log a 404 even when an ASR process is healthy.

Only the bridge's `STT_MODELS` changed; its Whisper-specific `STT_EXTRA_BODY` prompt was removed. TTS, MQTT, early transcription, satellite choices, the preferred Cloud assistant and the always-on CPU backup are unchanged. The existing local assistant retains its `Local (Kokoro + Whisper)` label because satellite selections and failover rules reference that name. The label no longer identifies its current recognition model.

Keep full Whisper Large-v3 Q5 on demand for language detection, multilingual transcription, English translation and VoiceStudio reference-clip ASR. Choosing Parakeet for English subtitles does not add an automatic error-triggered Parakeet-to-Whisper retry; select the appropriate model before non-English transcription. Remove Turbo and other unused models only after auditing their actual callers. Model cleanup preserves failed job history and does not replace Marquee’s separate CPU known-text aligner.

The initial voice-route check streamed three retained public command recordings through the actual Home Assistant local pipeline, Wyoming bridge and llama-swap: all 12 reference words matched, stopping at speech recognition without intent execution or TTS. This is server-side proof, not a physical microphone test or evidence of LumaShow’s separately selectable voice mode. LumaShow uses the local route only when actually configured for it; do not enable a disabled device or delete a possible legacy dependency based on an unreachable device’s assumed settings.

### Optional Marquee CPU alignment

The optional `POST /api/alignment/windows` service, introduced in 0.2.0, aligns exact English subtitle text to selected audio using a pinned quantized wav2vec2 CTC model on the CPU. An independent, unprompted recognition check rejects mismatched text, silence and unsupported characters. It preserves authored cue bounds and returns estimated fallback, never invented word times, when alignment is unsafe. Requests and responses bind the exact playback, seek generation, Plex media/part/audio/subtitle identity, subtitle digest and cue manifest. The Marquee client rejects stale responses and requires the actual selected renderer-ready subtitle source.

Version 0.2.1 preserves the raw independently recognized CTC tokens for acoustic scoring, avoiding false weak-acoustic rejection when normalization removes boundary or repeated separators. Exact independent lexical equality and all token-score, source, hash, bounds and playback/seek epoch guards remain unchanged. On the same 19 retained film windows, alignment improved from 1 to 2; the other 17 remained estimated because of lexical mismatch. Audio context, translation and model defaults are unchanged. The selected speech-to-text model does not replace the separate CPU known-text aligner.

For a fresh installation, configure these trusted server-side variables and mount media **read-only** before enabling the client endpoint. An already-configured production installation does not need these settings re-entered just for this update:

- `SPDOCK_ALIGNMENT_PLEX_URL`: the trusted Plex server origin, without token, query or path.
- `SPDOCK_ALIGNMENT_PLEX_TOKEN_FILE`: a private readable token file; never send it from the client or include it in a URL.
- `SPDOCK_ALIGNMENT_ROOTS`: JSON mappings such as `[{"plex":"/PMS/movies","local":"/media/movies"}]` matching Plex paths to read-only container mounts.
- `SPDOCK_ALIGNMENT_MODEL_DIR`: `/opt/speechdock/models/alignment` in the packaged image. Its model, vocabulary, configuration, provenance and Apache-2.0 license are baked in and checksummed; requests never download assets.

Missing configuration does not break ordinary SpeechDock jobs: alignment reports an explicit estimated result. Alignment admits one active CPU request and one waiting request, uses bounded windows and has no persistent result cache. Marquee diagnostics require separate session-only consent and default off. Local fixtures prove real CPU alignment and Rust HTTP exchange, including stale-generation rejection; they do not by themselves prove TV activation or deployment.

A non-null `source_revision` means the source was verified, not that a cue was acoustically aligned. Consumers must check each cue’s `status`, use word timings only for `aligned` cues, and clear previously accepted timings when a cue falls back to `estimated` or its source changes. Strict acoustic checks can reject ordinary film dialogue even with correct server configuration; an estimated response is not positive alignment proof.


## Processing and storage

Speech work runs one job at a time with a bounded waiting queue. Uploads are spooled to disk and normalized in five-minute PCM blocks. Transcription and translation send every bounded non-digitally-silent window, including quiet dialogue and background audio; they do not discard samples on a speech-detector prediction. This can send substantially more audio to the model than VAD-only processing. The checksum-pinned CPU Silero detector is retained for language detection. Original movie offsets, including long openings, are preserved. Digital silence produces no subtitle cues; language detection reports when no speech is available. Recognition can still miss, mishear or hallucinate content. Cancellation takes effect at safe processing boundaries, not necessarily during an engine request.

Subtitle text is preserved from each recognized segment. Parakeet uses the existing runtime’s explicit native token intervals, grouped by canonical English word boundaries; punctuation stays in the text but cannot extend a lexical word’s timing. Whisper uses its reported first/last token endpoint estimates, which are not independently verified acoustic word boundaries. Its unused intermediate estimates can overlap or run backward and are not exposed as word intervals. Every finished cue must still have positive duration, remain non-overlapping and fit the physical audio. Unsupported or malformed timing fails clearly rather than being clipped, dropped or replaced with invented times.

Defaults: 1 GiB upload limit, six-hour audio limit and 14-day retained history. Upload limits and retention are configurable. Failed/interrupted jobs remain visible with an explanation. Removing a job or expiring its history only removes SpeechDock's own records and files, never the original media library.

Back up `/data` for settings and history. Back up the owned-model folder if avoiding model re-downloads matters. Keep llama-swap configuration backups with its normal operational backups.

## API

The UI uses the same-origin `/api` endpoints. Interactive API documentation is available at `/docs`.

```sh
curl -F file=@recording.flac -F task=transcribe -F language=auto \
  https://speechdock.nateshome.net/api/jobs
```

The response is a queued job. Read `GET /api/jobs/{id}` for status and download `GET /api/jobs/{id}/result?format=srt` after completion. Detection-job detail includes `detection: {language_code, detected_language}`. Failures return real error statuses/messages, not empty fake subtitle files.

Settings are persisted after the first start. `SPDOCK_ENGINE_URL` supplies the initial engine URL; change saved settings in the UI afterward. Advanced deployment variables include `SPDOCK_DATA_DIR`, `SPDOCK_MODELS_DIR`, `SPDOCK_LLAMA_CONFIG`, `SPDOCK_ENGINE_MODELS_DIR`, `SPDOCK_LLAMA_VALIDATOR`, `SPDOCK_VAD_MODEL` and `SPDOCK_FRONTEND_DIR`. The production image bundles the read-only VAD model at `/opt/speechdock/models/silero-vad-v5.1.onnx`; it never downloads models on a transcription request.

## Development

Python 3.12+, uv, Node.js and FFmpeg are required for local development. Node is needed only to build the frontend.

```sh
uv sync --locked --all-extras
npm ci --include=dev --prefix ui
npm run check --prefix ui
npm run build --prefix ui
mkdir -p .data
curl --fail --location https://raw.githubusercontent.com/snakers4/silero-vad/v5.1/src/silero_vad/data/silero_vad.onnx --output .data/silero-vad-v5.1.onnx
printf '2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f  .data/silero-vad-v5.1.onnx\n' | sha256sum --check
SPDOCK_DATA_DIR=.data SPDOCK_VAD_MODEL=.data/silero-vad-v5.1.onnx uv run uvicorn speechdock.app:app --host 0.0.0.0 --port 8000
uv run pytest
```

Model registration requires a compatible real validator. To include validator-backed integration checks, set `SPDOCK_TEST_LLAMA_VALIDATOR` to that binary before running the suite.

## Releases and updates

CI checks the Python service and frontend without publishing. The manually dispatched **Release** workflow requires a stable `X.Y.Z` matching `pyproject.toml` and human-readable release notes. It checks the app, builds and smoke-tests the exact image, publishes the versioned GHCR image, creates the GitHub release/tag and moves `latest`. A push to `main` is not a release.

Images carry source, version and revision labels. Unraid's update check must resolve both local and remote image versions; verify it after installation, along with ShipLog's access to the public release notes and the app-served icon.

If a release fails partway, inspect the image tags and GitHub release before retrying. A failed health check is not enough reason for automatic rollback. Capture container health, exit/restart state and logs first; rollback by pinning a known-good image digest in the Unraid template and recreating through its native update path.
