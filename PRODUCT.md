# SpeechDock

## Product
SpeechDock is Nitin’s local speech-and-subtitle workspace: one small Unraid container that turns files into timed subtitles, exposes the Whisper ASR endpoints Bazarr expects, and manages supported speech models in his existing llama-swap service. It does not run a second speech inference engine.

## Users and workflows
Nitin uses Bazarr to find subtitles and wants local AI generation as a backup. He also wants to upload recordings independently of Bazarr, see recent work and errors, download/select speech models, and change relevant application settings without editing files.

The primary workflows are: check activity; transcribe a file or translate speech into English; install and select a supported model; inspect a result or error; configure the shared engine and integration. Honest empty, unavailable, pending, download, and error states are required. No invented activity or performance numbers.

## Platform
Responsive web application for desktop, tablet, and phone, including touch and keyboard input. Both light and dark themes, 320px reflow, 200% text zoom, reduced motion, readable contrast, and accessible named controls are required. One Unraid container serves both frontend and backend.

## Stack
Implementation approach delegated by Nitin after an explicit choice between a lightweight recommendation and React. Following his clarification that he wants a good UI framework, use Svelte 5, TypeScript, and Vite for the compiled frontend, with a Python/FastAPI backend. No Node process is needed in the runtime image. SQLite stores settings and activity locally.

## Capabilities and constraints
- Reuse llama-swap and its whisper-server. Preserve existing VoiceStudio and the whisper-1 alias.
- Support Bazarr’s /status, /detect-language, and /asr interfaces, including headerless 16kHz mono PCM uploads.
- Explicitly request valid timestamps. Transcription keeps the spoken language; translation produces English and must use a translation-capable full Whisper model, not Turbo.
- Curate official Whisper artifacts with pinned sizes and checksums. Manage only app-owned files and registered model entries. Existing shared models remain protected.
- Registration can reload llama-swap and interrupt active models; disclose that before registration. Do not silently rewrite unrelated model/group definitions.
- Bounded upload/storage limits, a serial speech job queue, download progress, recoverable errors, persistent history, result downloads, and explicit destructive confirmations.
- No Docker socket, GPU runtime, per-container Tailscale, public hostname exposure, or unrestricted arbitrary-model downloads.
- Scrypted’s GPU priority and existing llama-swap workload must remain respected.

## Identity and deployment
Working product name SpeechDock, confirmed in the implementation proposal. Own public GitHub repository under nphil, as requested on 2026-10-31; branded icon; versioned container releases; proper Unraid template, Docker-tab icon/WebUI, update metadata, persistent app data, and internal HomeLabber hostname. Container-package visibility is independent; pulls, update checks and release notes must all work.

## Brand commitments
Inherit Nitin’s active Lucent v2 design language and Rosé Pine palette, with self-hosted Manrope and one token layer. The interface should be clean, modern, calm, and functional, not decorative. Operate mode: the work and its state lead, with concise copy and progressive settings.

## Success
SpeechDock is running cleanly on Unraid; its UI reflects real activity, downloads, model state, settings, and results; full-model translation works; Bazarr’s connection test succeeds and real generated subtitles have valid timings. Existing speech/voice and unrelated llama-swap configuration still work.
