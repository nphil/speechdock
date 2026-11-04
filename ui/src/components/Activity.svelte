<script lang="ts">
  import { onMount } from 'svelte';
  import { api, type CaptionOutcome, type Job } from '../api';
  import JobDetail from './JobDetail.svelte';

  export let onChanged: (() => void) | undefined = undefined;
  export let selectedId: string | undefined = undefined;

  const limit = 20;
  let items: Job[] = [];
  let total = 0;
  let offset = 0;
  let status = '';
  let loading = true;
  let error = '';
  let selected: string | undefined = selectedId;
  let previousSelectedId = selectedId;
  let sequence = 0;

  $: if (selectedId !== previousSelectedId) {
    selected = selectedId;
    previousSelectedId = selectedId;
  }

  async function load(showLoading = false) {
    const request = ++sequence;
    if (showLoading) loading = true;
    try {
      const page = await api.jobs(limit, offset, status || undefined);
      if (request !== sequence) return;
      if (offset > 0 && page.total <= offset) {
        offset = Math.max(0, Math.ceil(page.total / limit) - 1) * limit;
        await load(showLoading);
        return;
      }
      items = page.items;
      total = page.total;
      error = '';
    } catch (e) {
      if (request !== sequence) return;
      error = e instanceof Error ? e.message : 'Could not load activity.';
    } finally {
      if (request === sequence) loading = false;
    }
  }
  onMount(() => {
    void load(true);
    const timer = window.setInterval(() => void load(), 5000);
    return () => { window.clearInterval(timer); sequence++; };
  });
  function filterChanged(event: Event) { status = (event.currentTarget as HTMLSelectElement).value; offset = 0; void load(true); }
  function page(delta: number) { offset = Math.max(0, offset + delta * limit); void load(true); }
  function changed() { void load(); onChanged?.(); }
  function date(value: string) {
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
  }
  const outcomes: Record<CaptionOutcome, string> = { written: 'captions written', matched_existing: 'skipped: real captions exist', already_generated: 'already generated', no_english_audio: 'no English audio', no_speech: 'no speech found' };
</script>

<section class="activity" aria-label="Activity">
  <div class="heading"><div><h1>Activity</h1><p>Track requests and retrieve completed results.</p></div><button type="button" class="button" onclick={() => load(true)}>Refresh</button></div>
  <div class="layout" class:has-detail={!!selected}>
    <div class="history">
      <div class="toolbar"><h2>Job history</h2><label><span>Status</span><select bind:value={status} onchange={filterChanged}><option value="">All statuses</option><option value="queued">Queued</option><option value="running">Running</option><option value="cancelling">Cancelling</option><option value="completed">Completed</option><option value="failed">Failed</option><option value="cancelled">Cancelled</option></select></label></div>
      {#if error}<div class="error" role="alert">{error} <button class="text-button" onclick={() => load(true)}>Try again</button></div>{/if}
      {#if loading}<p class="message" role="status">Loading activity…</p>
      {:else if items.length === 0 && !error}<div class="empty"><h3>No jobs {status ? `with status ${status}` : 'yet'}</h3><p>{status ? 'Choose another status to see more activity.' : 'Submitted files and API requests will appear here.'}</p></div>
      {:else}
        <div class="rows">{#each items as job (job.id)}
          <button type="button" class="row" class:selected={selected === job.id} aria-pressed={selected === job.id} onclick={() => selected = job.id}>
            <span class="row-main"><strong>{job.filename}</strong><span>{job.task === 'captions' ? 'Dub captions' : job.task === 'translate' ? 'Translate to English' : job.task === 'detect' ? 'Detect language' : 'Transcribe'} · {date(job.created_at)}{#if job.captions?.outcome} · {outcomes[job.captions.outcome]}{/if}</span></span>
            <span class="status" class:complete={job.status === 'completed'} class:failed={job.status === 'failed'}>{job.status}{#if job.status === 'running' && job.progress != null} · {Math.round(job.progress * 100)}%{/if}</span>
          </button>
        {/each}</div>
      {/if}
      {#if total > 0}<div class="pagination"><span>{offset + 1}–{Math.min(offset + limit, total)} of {total}</span><div><button class="button" type="button" disabled={offset === 0 || loading} onclick={() => page(-1)}>Previous</button><button class="button" type="button" disabled={offset + limit >= total || loading} onclick={() => page(1)}>Next</button></div></div>{/if}
    </div>
    {#if selected}<JobDetail id={selected} onChanged={changed} onDeleted={() => selected = undefined} onClose={() => selected = undefined} />{/if}
  </div>
</section>

<style>
  .activity{color:var(--lu-ink)}.heading{display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-bottom:1.5rem}.heading h1,.heading p,h2,h3,p{margin:0}.heading h1{font-size:clamp(1.65rem,3vw,2.1rem);letter-spacing:-.025em}.heading p{color:var(--lu-ink-2);margin-top:.35rem}.layout{display:grid;grid-template-columns:minmax(0,1fr);gap:1rem;align-items:start}.history{background:var(--lu-sheet);border:1px solid var(--lu-edge);border-radius:var(--lu-radius-card);overflow:hidden}.toolbar{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:1.2rem 1.3rem}.toolbar h2{font-size:1.05rem}.toolbar label{display:flex;gap:.55rem;align-items:center;color:var(--lu-ink-2);font-size:.85rem}select,.button{min-height:var(--lu-target);font:inherit;color:var(--lu-ink);background:var(--lu-glass);border:1px solid var(--lu-edge);border-radius:var(--lu-radius-control);padding:.45rem .8rem}button{cursor:pointer}.rows{border-top:1px solid var(--lu-edge)}.row{border:0;border-bottom:1px solid var(--lu-edge);background:transparent;color:inherit;text-align:left;width:100%;display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:1rem 1.3rem;font:inherit}.row:hover,.row.selected{background:var(--lu-glass)}.row-main{display:grid;gap:.3rem;min-width:0}.row-main strong{font-size:.93rem;overflow-wrap:anywhere}.row-main span{color:var(--lu-ink-2);font-size:.8rem}.status{font-size:.8rem;text-transform:capitalize;color:var(--lu-ink-2);white-space:nowrap}.status.complete{color:var(--lu-positive)}.status.failed,.error{color:var(--lu-danger)}.error{padding:1rem 1.3rem;overflow-wrap:anywhere}.text-button{background:none;border:0;color:inherit;text-decoration:underline;font:inherit}.empty,.message{padding:2.5rem 1.3rem;color:var(--lu-ink-2)}.empty h3{color:var(--lu-ink);font-size:1rem}.empty p{margin-top:.4rem;font-size:.9rem}.pagination{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:1rem 1.3rem;color:var(--lu-ink-2);font-size:.85rem}.pagination div{display:flex;gap:.5rem}.button:disabled{opacity:.5;cursor:not-allowed}.button:hover:not(:disabled){background:var(--lu-sheet)}button:focus-visible,select:focus-visible{outline:2px solid var(--lu-accent);outline-offset:2px}
  @media(max-width:520px){.heading{flex-wrap:wrap;align-items:flex-start}.toolbar label{flex-wrap:wrap;min-width:0;max-width:100%}.toolbar select{min-width:0;max-width:100%}.history{min-width:0}.pagination div{display:flex;flex-wrap:wrap;gap:.4rem}}
  @media(max-width:520px){.toolbar{align-items:flex-start;flex-direction:column}.row{align-items:flex-start;flex-direction:column;gap:.5rem}.pagination{flex-wrap:wrap}}
</style>
