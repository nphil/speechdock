# SpeechDock

A local speech-to-subtitles workspace for Bazarr and other clients. SpeechDock handles uploads, timed subtitle files, activity history and speech-model management; your existing llama-swap service runs the models.

One CPU-only app container. The existing llama-swap service launches model servers on demand; SpeechDock itself needs neither GPU access, a Docker socket nor a production Node.js runtime.

Version 0.2.0 adds manual model imports, an optional GGUF speech runtime and CPU subtitle alignment. Parakeet CUDA execution is verified on the Tesla P40. CPU alignment has real local Plex-fixture and Marquee HTTP proof; it still requires production Plex credentials/read-only mounts and the separately owned TV integration before use.

## Use it

- **Activity:** inspect queued, running and finished work; preview and download SRT, WebVTT or plain text.
- **New transcription:** upload audio or video, keep its spoken language or translate it into English.
- **Models:** download checksum-verified Whisper models, apply them to llama-swap, and remove only SpeechDock-owned models.
- **Settings:** choose the engine, transcription and translation models, language, timeouts, upload limit and history retention.

Whisper Turbo is the fast transcription default. **Turbo cannot translate.** English audio translation uses a separate translation-capable model and fails clearly until that model is installed and registered; it never silently substitutes Turbo. Language detection has its own compatible model selection, so choosing a transcription backend without language identification does not silently change Bazarr detection.

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

Parakeet transcription uses native word timing. Canary transcription uses source-validated DTW estimates, not native word alignment, and requires an explicit spoken language. Neither backend provides verified language detection here; Canary translation is disabled. Whisper remains available. For translation of existing subtitles into other languages, use a subtitle-text translator such as Lingarr with a suitable text model; it is separate from SpeechDock and does not repair source transcription or timing errors.

### Optional Marquee CPU alignment

The unreleased `POST /api/alignment/windows` service aligns exact English subtitle text to selected audio using a pinned quantized wav2vec2 CTC model on the CPU. An independent, unprompted recognition check rejects mismatched text, silence and unsupported characters. It preserves authored cue bounds and returns estimated fallback, never invented word times, when alignment is unsafe. Requests and responses bind the exact playback, seek generation, Plex media/part/audio/subtitle identity, subtitle digest and cue manifest. The Marquee client rejects stale responses and requires the actual selected renderer-ready subtitle source.

Configure these trusted server-side variables and mount media **read-only** before enabling the client endpoint:

- `SPDOCK_ALIGNMENT_PLEX_URL`: the trusted Plex server origin, without token, query or path.
- `SPDOCK_ALIGNMENT_PLEX_TOKEN_FILE`: a private readable token file; never send it from the client or include it in a URL.
- `SPDOCK_ALIGNMENT_ROOTS`: JSON mappings such as `[{"plex":"/PMS/movies","local":"/media/movies"}]` matching Plex paths to read-only container mounts.
- `SPDOCK_ALIGNMENT_MODEL_DIR`: `/opt/speechdock/models/alignment` in the packaged image. Its model, vocabulary, configuration, provenance and Apache-2.0 license are baked in and checksummed; requests never download assets.

Missing configuration does not break ordinary SpeechDock jobs: alignment reports an explicit estimated result. Alignment admits one active CPU request and one waiting request, uses bounded windows and has no persistent result cache. Marquee diagnostics require separate session-only consent and default off. Local fixtures prove real CPU alignment and Rust HTTP exchange, including stale-generation rejection; no production Plex mount, TV activation or deployment is implied.


## Processing and storage

Speech work runs one job at a time with a bounded waiting queue. Uploads are spooled to disk, not retained as whole media files in memory. A checksum-pinned Silero VAD model detects speech on the CPU, one five-minute PCM block at a time. Only bounded voiced windows are sent to llama-swap; subtitles retain their original movie timeline, including long silent openings. Digital silence produces no subtitle cues; language detection reports when no speech is available. Whisper may still miss or misrecognize speech. Cancellation takes effect at safe processing boundaries, not necessarily in the middle of an engine request.

Subtitle text is preserved from each recognized segment. Cue boundaries use the first token’s onset and the last token’s end rather than Whisper’s coarse segment-time predictions. Intermediate token estimates can overlap or run backward and are not exposed as word intervals. Every finished cue must still have positive duration, remain non-overlapping and fit the physical audio.

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
