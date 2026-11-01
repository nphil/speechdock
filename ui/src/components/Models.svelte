<script lang="ts">
  import { onMount } from 'svelte';
  import { api, type Model, type Download } from '../api';

  let models: Model[] = [];
  let downloads: Download[] = [];
  let loading = true;
  let error = '';
  let busy = '';
  let refreshing = false;
  let disposed = false;

  const active = (download: Download | null | undefined) => !!download && ['queued', 'downloading', 'verifying'].includes(download.status);
  const bytes = (value: number | null | undefined) => value == null ? 'Size unavailable' : value >= 1_000_000_000 ? `${(value / 1_000_000_000).toFixed(2)} GB` : `${Math.round(value / 1_000_000)} MB`;
  const message = (reason: unknown) => reason instanceof Error ? reason.message : 'Something went wrong. Please try again.';

  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    try {
      const result = await api.models();
      if (!disposed) { models = result.items; downloads = result.downloads; error = ''; }
    } catch (reason) {
      if (!disposed) error = message(reason);
    } finally {
      refreshing = false;
      if (!disposed) loading = false;
    }
  }

  async function act(model: Model, action: 'download' | 'cancel' | 'register' | 'unregister' | 'delete') {
    if (busy || model.read_only) return;
    if (action === 'register' && !window.confirm(`Register ${model.name}? This reloads llama-swap and interrupts loaded models. Using it can evict the daily and big groups; other GPU use is not controlled.`)) return;
    if (action === 'unregister' && !window.confirm(`Unregister ${model.name}? This reloads llama-swap and interrupts loaded models. The downloaded file will remain.`)) return;
    if (action === 'delete' && !window.confirm(`Delete the downloaded file for ${model.name}? This cannot be undone. Download it again to restore it.`)) return;
    busy = model.id;
    error = '';
    try {
      if (action === 'download') await api.downloadModel(model.id);
      else if (action === 'cancel') await api.cancelDownload(model.id);
      else if (action === 'register') await api.registerModel(model.id);
      else if (action === 'unregister') await api.unregisterModel(model.id);
      else await api.deleteModel(model.id);
      await refresh();
    } catch (reason) { error = message(reason); }
    finally { busy = ''; }
  }

  onMount(() => {
    disposed = false;
    void refresh();
    const timer = window.setInterval(() => { if (downloads.some(active)) void refresh(); }, 3000);
    return () => { disposed = true; window.clearInterval(timer); };
  });
</script>

<section class="models" aria-labelledby="models-heading">
  <header class="heading">
    <div><h1 id="models-heading">Models</h1><p>Choose the speech models available to your engine.</p></div>
    <button class="secondary" onclick={() => refresh()} disabled={refreshing}>Refresh</button>
  </header>
  <p class="notice">Registering or removing a registration reloads llama-swap and interrupts loaded models. Loading a model can also evict other workloads.</p>
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
            <span class:registered={model.registered} class="status">{model.registered ? 'Registered' : model.read_only ? 'Not detected' : model.verified ? 'Ready to register' : model.installed ? 'Verification required' : 'Not downloaded'}</span>
          </div>
          <div class="facts"><span>{bytes(model.size_bytes)}</span><span>Quality: {model.quality}</span><span>Speed: {model.speed}</span><span>{model.translation_capable ? 'Translation supported' : 'Transcription only'}</span>{#if model.read_only}<span>Shared · read-only</span>{/if}</div>
          {#if model.read_only}<p class="muted">{model.warning}</p>{/if}
          {#if download && (active(download) || download.status === 'failed' || download.status === 'cancelled')}
            <div class="download" aria-live="polite">
              <div class="download-label"><span>{download.status === 'verifying' ? 'Verifying download' : download.status === 'queued' ? 'Waiting to download' : download.status === 'downloading' ? 'Downloading' : download.status === 'failed' ? 'Download failed' : 'Download cancelled'}</span>{#if active(download)}<span>{Math.round(download.progress * 100)}%</span>{/if}</div>
              {#if active(download)}<progress max="1" value={download.progress} aria-label={`Download progress for ${model.name}`}></progress><small>{bytes(download.received_bytes)} of {bytes(download.total_bytes)}</small>{/if}
              {#if download.error}<p class="error">{download.error}</p>{/if}
            </div>
          {/if}
          {#if !model.read_only}
            <div class="actions">
              {#if active(download)}
                <button class="secondary" onclick={() => act(model, 'cancel')} disabled={!!busy}>Cancel download</button>
              {:else if !model.installed}
                <button class="primary" onclick={() => act(model, 'download')} disabled={!!busy}>{download?.status === 'failed' || download?.status === 'cancelled' ? 'Download again' : 'Download'}</button>
              {/if}
              {#if model.can_register && !model.registered && !active(download)}<button class="primary" onclick={() => act(model, 'register')} disabled={!!busy}>Register</button>{/if}
              {#if model.registered}<button class="secondary" onclick={() => act(model, 'unregister')} disabled={!!busy}>Unregister</button>{/if}
              {#if model.installed}<button class="danger" onclick={() => act(model, 'delete')} disabled={!!busy || model.registered || active(download)} title={model.registered ? 'Unregister this model before deleting its file' : undefined}>Delete file</button>{/if}
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
