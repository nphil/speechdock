# SpeechDock

A local speech-to-subtitles workspace for Bazarr and other clients. SpeechDock handles uploads, timed subtitle files, activity history and speech-model management; your existing llama-swap service runs the models.

One CPU-only container. No second inference server, GPU access, Docker socket or production Node.js runtime.

## Use it

- **Activity:** inspect queued, running and finished work; preview and download SRT, WebVTT or plain text.
- **New transcription:** upload audio or video, keep its spoken language or translate it into English.
- **Models:** download checksum-verified Whisper models, apply them to llama-swap, and remove only SpeechDock-owned models.
- **Settings:** choose the engine, transcription and translation models, language, timeouts, upload limit and history retention.

Whisper Turbo is the fast transcription default. **Turbo cannot translate.** English translation uses a separate translation-capable model, with full Whisper Large-v3 as the initial selection. Translation fails clearly until that model is installed and applied; it never silently substitutes Turbo.

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

- Downloads come from a fixed official Whisper catalogue, pinned to a revision with exact size and SHA-256 verification. Arbitrary download URLs are not accepted.
- Existing shared models, including the `whisper-1` Turbo alias, are read-only. SpeechDock edits only its own `speechdock-*` definitions and `speechdock` group, preserving unrelated YAML and detecting conflicting edits.
- **Applying or removing model definitions reloads llama-swap and interrupts its loaded models.** The UI requires a separate confirmation and keeps a configuration backup. The engine must be running with configuration watching enabled.
- Loading a managed speech model can evict warmed chat models. A free-GPU-memory launch guard reserves conservative headroom; its thresholds are not measured performance or memory benchmarks. Other GPU services retain priority.
- Model-file size is not GPU-memory consumption. Full Large-v3 is about 3.1 GB on disk, and has a higher memory requirement than its quantized variants.

## Processing and storage

Speech work runs one job at a time with a bounded waiting queue. Uploads are spooled to disk, not retained as whole media files in memory. A checksum-pinned Silero VAD model detects speech on the CPU, one five-minute PCM block at a time. Only bounded voiced windows are sent to llama-swap; subtitles retain their original movie timeline, including long silent openings. Digital silence produces no subtitle cues; language detection reports when no speech is available. Whisper may still miss or misrecognize speech. Cancellation takes effect at safe processing boundaries, not necessarily in the middle of an engine request.

Subtitle text is preserved from each recognized segment. Cue boundaries use Whisper’s finer word/token alignment rather than its coarse segment-time predictions. Malformed, overlapping, zero-duration and physically out-of-bounds cues remain errors.

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
