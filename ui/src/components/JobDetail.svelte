<script lang="ts">
  import { onMount } from 'svelte';
  import { api, type Job } from '../api';

  export let id: string;
  export let onChanged: (() => void) | undefined = undefined;
  export let onDeleted: (() => void) | undefined = undefined;
  export let onClose: (() => void) | undefined = undefined;

  let job: Job | null = null;
  let loading = true;
  let error = '';
  let busy = false;
  let confirmAction: 'cancel' | 'delete' | null = null;
  let currentId = '';
  let sequence = 0;
  let mounted = false;
  let previewFormat: 'srt' | 'txt' = 'srt';
  let plainText: string | null = null;
  let previewError = '';
  let previewLoading = false;

  async function selectPreview(format: 'srt' | 'txt') {
    previewFormat = format;
    if (format !== 'txt' || plainText !== null || previewLoading) return;
    const requestedId = id;
    previewLoading = true;
    previewError = '';
    try {
      const response = await fetch(api.resultUrl(requestedId, 'txt'));
      if (!response.ok) throw new Error('Could not load the text preview. Please try again.');
      const text = await response.text();
      if (id === requestedId) plainText = text;
    } catch (reason) {
      if (id === requestedId) previewError = reason instanceof Error ? reason.message : 'Could not load the text preview.';
    } finally {
      if (id === requestedId) previewLoading = false;
    }
  }

  async function load(showLoading = false) {
    const requestedId = id;
    const request = ++sequence;
    if (showLoading) loading = true;
    try {
      const result = await api.job(requestedId);
      if (request !== sequence) return;
      job = result;
      error = '';
    } catch (e) {
      if (request !== sequence) return;
      error = e instanceof Error ? e.message : 'Could not load this job.';
    } finally {
      if (request === sequence) loading = false;
    }
  }

  $: if (mounted && id !== currentId) {
    currentId = id;
    job = null;
    confirmAction = null;
    previewFormat = 'srt';
    plainText = null;
    previewError = '';
    previewLoading = false;
    void load(true);
  }

  onMount(() => {
    mounted = true;
    const timer = window.setInterval(() => {
      if (job && ['queued', 'running', 'cancelling'].includes(job.status) && !busy) void load();
    }, 5000);
    return () => { window.clearInterval(timer); sequence++; };
  });

  async function performAction() {
    if (!confirmAction || busy) return;
    const action = confirmAction;
    busy = true;
    error = '';
    try {
      if (action === 'cancel') {
        await api.cancelJob(id);
        confirmAction = null;
        await load();
        onChanged?.();
      } else {
        await api.deleteJob(id);
        confirmAction = null;
        onChanged?.();
        onDeleted?.();
      }
    } catch (e) {
      error = e instanceof Error ? e.message : `Could not ${action} this job.`;
    } finally { busy = false; }
  }

  function date(value: string | null | undefined) {
    if (!value) return '—';
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
  }
</script>

<section class="detail" aria-label="Job details">
  <div class="heading">
    <div><h2>Job details</h2><p>Inspect the request and its result.</p></div>
    {#if onClose}<button class="quiet" type="button" onclick={onClose}>Close</button>{/if}
  </div>
  {#if loading}<p class="message" role="status">Loading job…</p>{/if}
  {#if error}<div class="error" role="alert">{error} <button class="text-button" onclick={() => load(true)}>Try again</button></div>{/if}
  {#if job}
    <div class="job-title"><h3>{job.filename}</h3><span class:failed={job.status === 'failed'} class:complete={job.status === 'completed'} class="status">{job.status}</span></div>
    <dl>
      <div><dt>Task</dt><dd>{job.task === 'translate' ? 'Translate to English' : job.task === 'detect' ? 'Detect language' : 'Transcribe'}</dd></div>
      <div><dt>Language</dt><dd>{job.language}</dd></div>
      <div><dt>Model</dt><dd>{job.model}</dd></div>
      <div><dt>Source</dt><dd>{job.source}</dd></div>
      <div><dt>Created</dt><dd>{date(job.created_at)}</dd></div>
      <div><dt>Finished</dt><dd>{date(job.finished_at)}</dd></div>
      {#if job.duration_seconds != null}<div><dt>Duration</dt><dd>{Math.round(job.duration_seconds)} seconds</dd></div>{/if}
      {#if job.progress != null && ['running', 'cancelling'].includes(job.status)}<div><dt>Progress</dt><dd>{Math.round(job.progress * 100)}%</dd></div>{/if}
    </dl>
    {#if job.error}<div class="error" role="alert">{job.error}</div>{/if}
    {#if job.result_available && job.task === 'detect' && job.detection}
      <div class="result"><h3>Detected language</h3><p>{job.detection.detected_language} <span>({job.detection.language_code})</span></p></div>
    {:else if job.result_available && job.task !== 'detect'}
      <div class="result">
        <h3>Result preview</h3>
        <div class="actions" aria-label="Preview format">
          <button type="button" class="quiet" aria-pressed={previewFormat === 'srt'} onclick={() => selectPreview('srt')}>Subtitles (SRT)</button>
          <button type="button" class="quiet" aria-pressed={previewFormat === 'txt'} onclick={() => selectPreview('txt')}>Text</button>
        </div>
        {#if previewFormat === 'srt'}
          {#if job.result_text}<pre class="preview">{job.result_text}</pre>{:else}<p>No subtitle text is available.</p>{/if}
        {:else if previewLoading}<p role="status">Loading text preview…</p>
        {:else if previewError}<p class="error" role="alert">{previewError} <button type="button" class="text-button" onclick={() => selectPreview('txt')}>Try again</button></p>
        {:else if plainText}<pre class="preview">{plainText}</pre>
        {:else}<p>No text is available.</p>{/if}
        <h3 class="download-heading">Download result</h3>
        <div class="actions">{#each ['srt', 'vtt', 'txt'] as format}<a class="button" href={api.resultUrl(job.id, format as 'srt' | 'vtt' | 'txt')} download>{format.toUpperCase()}</a>{/each}</div>
      </div>
    {/if}
    <div class="actions bottom">
      {#if ['queued', 'running'].includes(job.status)}<button type="button" class="quiet" disabled={busy} onclick={() => confirmAction = 'cancel'}>Cancel job</button>{/if}
      {#if ['completed', 'failed', 'cancelled'].includes(job.status)}<button type="button" class="danger" disabled={busy} onclick={() => confirmAction = 'delete'}>Delete job</button>{/if}
    </div>
    {#if confirmAction}
      <div class="confirmation" role="group" aria-label="Confirm action">
        <p>{confirmAction === 'delete' ? 'Delete this job and its result? This cannot be undone.' : 'Cancel this job? Processing will stop.'}</p>
        <div class="actions"><button type="button" class="danger" disabled={busy} onclick={performAction}>{busy ? 'Working…' : confirmAction === 'delete' ? 'Delete job' : 'Cancel job'}</button><button type="button" class="quiet" disabled={busy} onclick={() => confirmAction = null}>Keep job</button></div>
      </div>
    {/if}
  {/if}
</section>

<style>
  .detail{background:var(--lu-sheet);border:1px solid var(--lu-edge);border-radius:var(--lu-radius-card);padding:clamp(1.1rem,3vw,1.6rem);color:var(--lu-ink)}
  .heading,.job-title{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem}.heading{margin-bottom:1.4rem}h2,h3,p{margin:0}h2{font-size:1.2rem}h3{font-size:1rem;overflow-wrap:anywhere}.heading p,.message{color:var(--lu-ink-2);margin-top:.35rem;font-size:.9rem}.job-title{align-items:center}.status{color:var(--lu-ink-2);text-transform:capitalize;white-space:nowrap;font-size:.8rem}.status.complete{color:var(--lu-positive)}.status.failed{color:var(--lu-danger)}
  dl{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem;margin:1.4rem 0}dt{color:var(--lu-ink-2);font-size:.78rem}dd{margin:.25rem 0 0;overflow-wrap:anywhere;font-size:.9rem} .result,.bottom,.confirmation{border-top:1px solid var(--lu-edge);padding-top:1rem;margin-top:1rem}.result p{margin-top:.6rem}.result span{color:var(--lu-ink-2)}.actions{display:flex;flex-wrap:wrap;gap:.55rem}.result .actions{margin-top:.75rem}
  .preview{white-space:pre-wrap;overflow-wrap:anywhere;max-height:22rem;overflow:auto;font:inherit;line-height:1.6;margin:1rem 0 0;padding:1rem;background:var(--lu-material-well);border-radius:var(--lu-radius-control)}.download-heading{margin-top:1.25rem}.actions button[aria-pressed="true"]{background:var(--lu-material-selected-wash);box-shadow:inset 0 -2px var(--lu-accent)}
  button,.button{font:inherit;min-height:var(--lu-target);border-radius:var(--lu-radius-control);border:1px solid var(--lu-edge);padding:.55rem .9rem;background:var(--lu-glass);color:var(--lu-ink);cursor:pointer;text-decoration:none;display:inline-flex;align-items:center;justify-content:center}button:disabled{opacity:.55;cursor:not-allowed}.danger{color:var(--lu-danger)}.text-button{border:0;background:none;padding:0;min-height:0;color:inherit;text-decoration:underline}.error{color:var(--lu-danger);margin:1rem 0;overflow-wrap:anywhere}.confirmation p{margin-bottom:.8rem}.quiet:hover,.button:hover{background:var(--lu-sheet)}button:focus-visible,.button:focus-visible{outline:2px solid var(--lu-accent);outline-offset:2px}
  @media(max-width:480px){dl{grid-template-columns:1fr 1fr}.heading{align-items:center}}
  @media(max-width:400px){.detail{padding:12px;min-width:0}.heading,.job-title{flex-wrap:wrap}.heading > div{min-width:0}.preview{padding:12px}.actions button,.actions a{max-width:100%;overflow-wrap:anywhere}dl{grid-template-columns:minmax(0,1fr)}}
</style>
