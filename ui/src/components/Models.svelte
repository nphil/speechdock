<script lang="ts">
  import { onMount } from 'svelte';
  import { api, HttpError, type Model, type Download } from '../api';

  let models: Model[] = [];
  let downloads: Download[] = [];
  let loading = true;
  let error = '';
  let busy = '';
  let refreshing = false;
  let disposed = false;
  let importSource: 'upload' | 'url' = 'upload';
  let importName = '';
  let importUrl = '';
  let importFilename = '';
  let importSha = '';
  let importSize = '';
  let importFile: File | null = null;
  let importFileInput: HTMLInputElement | undefined;
  let importBusy = false;
  let uploadController: AbortController | null = null;
  let uploadingId = '';
  let uploadProgress: number | null = null;
  let importNotice = '';
  let runtimes: Record<string, {available: boolean; version: string | null; error: string | null}> = {};

  const active = (download: Download | null | undefined) => !!download && ['awaiting_upload', 'queued', 'downloading', 'uploading', 'verifying'].includes(download.status);
  const imported = (model: Model) => model.source === 'upload' || model.source === 'url';
  const checksum = (model: Model) => model.checksum_status === 'pinned' ? (model.verified ? 'Checksum verified against pinned value' : 'Pinned checksum; verification pending') : model.checksum_status === 'matched' ? 'Checksum matches supplied SHA-256' : model.checksum_status === 'computed' ? 'Checksum calculated; not independently verified' : 'Checksum not yet verified';
  const bytes = (value: number | null | undefined) => value == null ? 'Size unavailable' : value >= 1_000_000_000 ? `${(value / 1_000_000_000).toFixed(2)} GB` : `${Math.round(value / 1_000_000)} MB`;
  const message = (reason: unknown) => reason instanceof Error ? reason.message : 'Something went wrong. Please try again.';

  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    try {
      const result = await api.models();
      if (!disposed) { models = result.items; downloads = result.downloads; runtimes = result.runtimes; }
    } catch (reason) {
      if (!disposed) error = message(reason);
    } finally {
      refreshing = false;
      if (!disposed) loading = false;
    }
  }
  async function cancelUpload() {
    const id = uploadingId;
    const controller = uploadController;
    if (!id || !controller) return;
    const cancellation = api.cancelDownload(id);
    controller.abort();
    try { await cancellation; }
    catch (reason) { if (!(reason instanceof HttpError && reason.status === 409) && !disposed) error = message(reason); }
    if (!disposed) await refresh();
  }
  async function startImport(event: SubmitEvent) {
    event.preventDefault();
    if (importBusy || (importSource === 'upload' && !importFile)) return;
    const file = importFile;
    importBusy = true; error = ''; importNotice = ''; uploadProgress = null;
    try {
      const result = await api.createImport({ name: importName.trim(), source: importSource,
        ...(importSource === 'url' ? { url: importUrl.trim(), ...(importFilename.trim() ? { source_filename: importFilename.trim() } : {}) } : { source_filename: file!.name }),
        ...(importSha.trim() ? { expected_sha256: importSha.trim() } : {}),
        ...(importSize ? { expected_size_bytes: Number(importSize) } : {}) });
      await refresh();
      if (importSource === 'upload' && file) {
        uploadingId = result.model.id;
        uploadController = new AbortController();
        await api.uploadModel(result.model.id, file, uploadController.signal, value => { uploadProgress = value; });
        importNotice = 'Upload received. The model is not registered; review it before registering.';
      } else importNotice = 'Import started. The model will not be registered automatically.';
      importName = ''; importUrl = ''; importFilename = ''; importSha = ''; importSize = ''; importFile = null;
      if (importFileInput) importFileInput.value = "";
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === 'AbortError') importNotice = 'Upload stopped; check the model status below.';
      else error = message(reason);
    } finally {
      uploadController = null; uploadingId = ''; importBusy = false; await refresh();
    }
  }


  async function act(model: Model, action: 'download' | 'cancel' | 'register' | 'unregister' | 'delete' | 'remove') {
    if (busy || model.read_only) return;
    if (action === 'register' && !window.confirm(`Register ${model.name}? This reloads llama-swap and interrupts loaded models. Using it can evict the daily and big groups; other GPU use is not controlled.`)) return;
    if (action === 'unregister' && !window.confirm(`Unregister ${model.name}? This reloads llama-swap and interrupts loaded models. The downloaded file will remain.`)) return;
    if (action === 'delete' && !window.confirm(`Delete the downloaded file for ${model.name}? This cannot be undone. Download it again to restore it.`)) return;
    if (action === 'remove' && !window.confirm(`Remove import ${model.name} and its owned file? This cannot be undone.`)) return;
    busy = model.id;
    error = '';
    try {
      if (action === 'download') await api.downloadModel(model.id);
      else if (action === 'cancel') await api.cancelDownload(model.id);
      else if (action === 'register') await api.registerModel(model.id);
      else if (action === 'unregister') await api.unregisterModel(model.id);
      else if (action === 'remove') await api.removeImport(model.id);
      else await api.deleteModel(model.id);
      await refresh();
    } catch (reason) { error = message(reason); }
    finally { busy = ''; }
  }

  onMount(() => {
    disposed = false;
    void refresh();
    const timer = window.setInterval(() => { if (downloads.some(active)) void refresh(); }, 3000);
    return () => { disposed = true; window.clearInterval(timer); if (uploadController) void cancelUpload(); };
  });
</script>

<section class="models" aria-labelledby="models-heading">
  <header class="heading">
    <div><h1 id="models-heading">Models</h1><p>Choose the speech models available to your engine.</p></div>
    <button class="secondary" onclick={() => refresh()} disabled={refreshing}>Refresh</button>
  </header>
  <p class="notice">Registering or removing a registration reloads llama-swap and interrupts loaded models. Loading a model can also evict other workloads.</p>
  <form class="import-form" onsubmit={startImport} aria-labelledby="import-heading">
    <div><h2 id="import-heading">Import a model</h2><p class="muted">Use a trusted source. Supported files are Whisper GGML and validated Parakeet or Canary GGUF models, not arbitrary GGUF files. Importing does not register the model.</p></div>
    <div class="import-fields">
      <label>Model name<input required maxlength="100" bind:value={importName} disabled={importBusy} /></label>
      <label>Source<select bind:value={importSource} disabled={importBusy}><option value="upload">Upload a file</option><option value="url">Direct URL</option></select></label>
      {#if importSource === 'upload'}<label>Model file<input bind:this={importFileInput} type="file" required disabled={importBusy} onchange={(event) => { importFile = event.currentTarget.files?.[0] ?? null; }} /></label>
      {:else}<label>Direct URL<input type="url" required bind:value={importUrl} disabled={importBusy} /></label><label>Source filename <small>Optional</small><input bind:value={importFilename} disabled={importBusy} placeholder="model.gguf" /></label>{/if}
    </div>
    <details class="advanced"><summary>Advanced verification options</summary><p class="muted">A supplied SHA-256 is checked against the file. Without one, SpeechDock calculates a checksum but cannot independently verify the source.</p><div class="import-fields"><label>Expected SHA-256 <small>Optional</small><input bind:value={importSha} disabled={importBusy} pattern={'[A-Fa-f0-9]{64}'} title="Enter 64 hexadecimal characters" spellcheck="false" /></label><label>Expected size in bytes <small>Optional</small><input type="number" min="1" step="1" bind:value={importSize} disabled={importBusy} /></label></div></details>
    <div class="actions"><button class="primary" type="submit" disabled={importBusy}>{importBusy ? 'Importing…' : 'Import model'}</button>{#if uploadController}<button class="secondary" type="button" onclick={cancelUpload}>Cancel upload</button>{/if}</div>
    {#if uploadController}<div class="download" role="status"><span>Uploading{uploadProgress == null ? '' : ` ${Math.round(uploadProgress * 100)}%`}</span>{#if uploadProgress == null}<progress aria-label="Upload progress"></progress>{:else}<progress max="1" value={uploadProgress} aria-label="Upload progress"></progress>{/if}</div>{/if}
    {#if importNotice}<p class="muted" role="status">{importNotice}</p>{/if}
  </form>
  {#if Object.keys(runtimes).length}<div class="facts" aria-label="Runtime availability">{#each Object.entries(runtimes) as [name, runtime]}<span>{name}: {runtime.available ? 'Available' : 'Unavailable'}{runtime.version ? ` · ${runtime.version}` : ''}{runtime.error ? ` · ${runtime.error}` : ''}</span>{/each}</div>{/if}
  {#if error}<p class="error" role="alert">{error}</p>{/if}
  {#if loading}
    <p class="muted" role="status">Loading models…</p>
  {:else if models.length === 0}
    <p class="muted">No models are available.</p>
  {:else}
    <div class="list">
      {#each models as model (model.id)}
        {@const download = downloads.find(item => item.id === model.id) ?? model.download}
        <article class="model">
          <div class="model-top">
            <div class="identity"><h2>{model.name}</h2><p>{model.description}</p></div>
            <span class:registered={model.registered} class="status">{model.registered ? 'Registered' : model.read_only ? 'Not detected' : model.can_register ? 'Ready to register' : model.installed ? 'Not ready to register' : 'Not installed'}</span>
          </div>
          <div class="facts"><span>{bytes(model.size_bytes)}</span>{#if model.quality}<span>Quality: {model.quality}</span>{/if}{#if model.speed}<span>Speed: {model.speed}</span>{/if}<span>{model.translation_capable ? 'Translation supported' : 'Translation not supported'}</span>{#if model.read_only}<span>Shared · read-only</span>{/if}</div>
          <div class="facts"><span>{model.runtime && model.architecture && model.format ? `${model.runtime} · ${model.architecture} · ${model.format}` : 'Model format not yet identified'}</span><span>{checksum(model)}</span><span>{model.language_detection_capable ? 'Language detection supported' : 'No language detection'}</span>{#if model.requires_language}<span>Explicit language required</span>{/if}{#if imported(model)}<span>Source: {model.source}{model.source_host ? ` · ${model.source_host}` : ''}{model.source_filename ? ` · ${model.source_filename}` : ''}</span>{/if}</div>
          {#if model.runtime_error}<p class="error">Runtime unavailable: {model.runtime_error}</p>{/if}
          {#if model.read_only}<p class="muted">{model.warning}</p>{/if}
          {#if download && (active(download) || download.status === 'failed' || download.status === 'cancelled')}
            <div class="download" aria-live="polite">
              <div class="download-label"><span>{download.status === 'verifying' ? 'Verifying' : download.status === 'awaiting_upload' ? 'Waiting for upload' : download.status === 'queued' ? 'Waiting' : download.status === 'uploading' ? 'Uploading' : download.status === 'downloading' ? 'Downloading' : download.status === 'failed' ? 'Import failed' : 'Cancelled'}</span>{#if active(download) && download.progress != null}<span>{Math.round(download.progress * 100)}%</span>{/if}</div>
              {#if active(download)}{#if download.progress == null}<progress aria-label={`Progress for ${model.name}`}></progress>{:else}<progress max="1" value={download.progress} aria-label={`Progress for ${model.name}`}></progress>{/if}<small>{bytes(download.received_bytes)}{download.total_bytes == null ? ' received' : ` of ${bytes(download.total_bytes)}`}</small>{/if}
              {#if download.error}<p class="error">{download.error}</p>{/if}
            </div>
          {/if}
          {#if !model.read_only}
            <div class="actions">
              {#if active(download)}
                <button class="secondary" onclick={() => uploadingId === model.id ? cancelUpload() : act(model, 'cancel')} disabled={!!busy}>Cancel {download?.status === 'uploading' || download?.status === 'awaiting_upload' ? 'upload' : 'download'}</button>
              {:else if !model.installed && !imported(model)}
                <button class="primary" onclick={() => act(model, 'download')} disabled={!!busy}>{download?.status === 'failed' || download?.status === 'cancelled' ? 'Download again' : 'Download'}</button>
              {/if}
              {#if model.can_register && !model.registered && !active(download)}<button class="primary" onclick={() => act(model, 'register')} disabled={!!busy}>Register</button>{/if}
              {#if model.registered}<button class="secondary" onclick={() => act(model, 'unregister')} disabled={!!busy}>Unregister</button>{/if}
              {#if imported(model)}<button class="danger" onclick={() => act(model, 'remove')} disabled={!!busy || model.registered || active(download)} title={model.registered ? 'Unregister this model before removing it' : undefined}>Remove import</button>
              {:else if model.installed}<button class="danger" onclick={() => act(model, 'delete')} disabled={!!busy || model.registered || active(download)} title={model.registered ? 'Unregister this model before deleting its file' : undefined}>Delete file</button>{/if}
              {#if busy === model.id}<span class="muted" role="status">Working…</span>{/if}
            </div>
          {/if}
        </article>
      {/each}
    </div>
  {/if}
</section>

<style>
  .models { display: grid; gap: 1.25rem; min-width: 0; }
  .heading, .model-top, .download-label { display: flex; justify-content: space-between; gap: 1rem; align-items: start; }
  h1, h2, p { margin: 0; } h1 { font-size: clamp(1.6rem, 3vw, 2.2rem); letter-spacing: -.03em; } h2 { font-size: 1.05rem; }
  .heading p, .identity p { color: var(--lu-ink-2); margin-top: .35rem; line-height: 1.5; }
  .notice { background: var(--lu-glass); border: 1px solid var(--lu-edge); border-radius: var(--lu-radius-card); padding: 1rem; color: var(--lu-ink-2); line-height: 1.5; }
  .import-form { background: var(--lu-sheet); border: 1px solid var(--lu-edge); border-radius: var(--lu-radius-card); padding: 1.25rem; min-width: 0; }
  .advanced { margin-top: 1rem; } .advanced summary { cursor: pointer; font-weight: 700; } .advanced summary:focus-visible { outline: 2px solid var(--lu-accent); outline-offset: 3px; }
  .import-fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 15rem), 1fr)); gap: 1rem; margin-top: 1rem; }
  .import-fields label { display: grid; gap: .45rem; min-width: 0; font-size: .9rem; font-weight: 700; }
  .import-fields input, .import-fields select { box-sizing: border-box; width: 100%; min-width: 0; min-height: var(--lu-target); padding: .55rem .7rem; font: inherit; font-weight: 400; color: var(--lu-ink); background: var(--lu-glass); border: 1px solid var(--lu-edge); border-radius: var(--lu-radius-control); }
  .import-fields input:focus-visible, .import-fields select:focus-visible { outline: 2px solid var(--lu-accent); outline-offset: 3px; }
  .list { background: var(--lu-sheet); border: 1px solid var(--lu-edge); border-radius: var(--lu-radius-card); padding: 0 1.25rem; }
  .model { padding: 1.3rem 0; } .model + .model { border-top: 1px solid var(--lu-edge); }
  .status { white-space: nowrap; color: var(--lu-ink-2); font-size: .8rem; font-weight: 700; background: var(--lu-glass); border-radius: var(--lu-radius-control); padding: .4rem .65rem; }
  .status.registered { color: var(--lu-positive); }
  .facts { display: flex; flex-wrap: wrap; gap: .4rem 1.2rem; margin-top: .9rem; color: var(--lu-ink-2); font-size: .86rem; }
  .muted { color: var(--lu-ink-2); font-size: .88rem; line-height: 1.5; margin-top: .7rem; }
  .error { color: var(--lu-danger); overflow-wrap: anywhere; } .download { margin-top: 1rem; max-width: 35rem; } .download-label, small { color: var(--lu-ink-2); font-size: .85rem; } progress { width: 100%; height: .6rem; accent-color: var(--lu-accent); }
  .actions { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center; margin-top: 1rem; }
  button { min-height: var(--lu-target); border: 1px solid var(--lu-edge); border-radius: var(--lu-radius-control); padding: .55rem .9rem; font: inherit; font-weight: 700; cursor: pointer; }
  button:focus-visible { outline: 2px solid var(--lu-accent); outline-offset: 3px; } button:disabled { opacity: .55; cursor: not-allowed; }
  .primary { background: var(--lu-accent); color: var(--lu-accent-ink); } .secondary { background: var(--lu-glass); color: var(--lu-ink); } .danger { background: var(--lu-glass); color: var(--lu-danger); }
  @media (max-width: 520px) { .heading, .model-top { flex-wrap: wrap; } .list { padding: 0 1rem; } .status { white-space: normal; } }
  @media (max-width: 400px) { .list { padding: 0 12px; } .model, .identity, .heading > div { min-width: 0; overflow-wrap: anywhere; } .download-label { flex-wrap: wrap; } .actions button { max-width: 100%; overflow-wrap: anywhere; } }
</style>
