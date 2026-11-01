<script lang="ts">
  import { onMount } from 'svelte';
  import { api, type Model, type Settings } from '../api';

  export let onChanged: (() => void) | undefined = undefined;

  let models: Model[] = [];
  let settings: Settings | null = null;
  let loading = true;
  let submitting = false;
  let error = '';
  let success = '';
  let file: File | null = null;
  let fileInput: HTMLInputElement;
  let task: 'transcribe' | 'translate' | 'detect' = 'transcribe';
  let language = 'auto';
  let model = '';
  let available: Model[] = [];

  $: available = models.filter(item => item.registered && (task !== 'translate' || item.translation_capable));

  async function load() {
    loading = true;
    error = '';
    try {
      const [modelResponse, preferences] = await Promise.all([api.models(), api.settings()]);
      models = modelResponse.items;
      settings = preferences;
      language = preferences.default_language;
      chooseDefault();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Could not load upload options.';
    } finally { loading = false; }
  }
  onMount(() => { void load(); });

  function chooseDefault() {
    const preferred = task === 'translate' ? settings?.translation_model : settings?.transcription_model;
    const options = models.filter(item => item.registered && (task !== 'translate' || item.translation_capable));
    model = options.find(item => item.model_id === preferred)?.model_id ?? '';
  }
  function taskChanged(event: Event) {
    task = (event.currentTarget as HTMLSelectElement).value as typeof task;
    chooseDefault();
  }
  function fileChanged(event: Event) {
    file = (event.currentTarget as HTMLInputElement).files?.[0] ?? null;
    success = '';
  }
  async function submit(event: SubmitEvent) {
    event.preventDefault();
    if (!file || !model || submitting) return;
    submitting = true;
    error = '';
    success = '';
    try {
      const submittedName = file.name;
      await api.submitJob(file, task, language.trim() || 'auto', model);
      success = `${submittedName} was added to the queue.`;
      file = null;
      if (fileInput) fileInput.value = '';
      onChanged?.();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Could not submit this file.';
    } finally { submitting = false; }
  }
</script>

<section class="upload" aria-label="Upload a file">
  <div class="heading"><h1>New job</h1><p>Turn audio or video into a transcript, English translation, or language detection.</p></div>
  <form onsubmit={submit}>
    <div class="form-heading"><h2>Choose a file</h2><p>Files are processed by your configured speech engine.</p></div>
    {#if loading}<p class="message" role="status">Loading upload options…</p>{/if}
    {#if error}<div class="error" role="alert">{error}{#if !settings} <button class="text-button" type="button" onclick={load}>Try again</button>{/if}</div>{/if}
    {#if success}<div class="success" role="status">{success}</div>{/if}
    <label class="file-field"><span>Audio or video file</span><input bind:this={fileInput} type="file" accept="audio/*,video/*" onchange={fileChanged} required disabled={loading || submitting} /><small>{settings ? `Maximum upload size: ${settings.max_upload_mb} MB.` : 'Select a file from your device.'}</small></label>
    <div class="fields">
      <label><span>Task</span><select bind:value={task} onchange={taskChanged} disabled={loading || submitting}><option value="transcribe">Transcribe</option><option value="translate">Translate to English</option><option value="detect">Detect language</option></select><small>{task === 'translate' ? 'Translation output is always in English.' : task === 'detect' ? 'Identify the language spoken in the file.' : 'Keep the spoken language in the transcript.'}</small></label>
      <label><span>Spoken language</span><input type="text" bind:value={language} disabled={loading || submitting} placeholder="auto" /><small>Use auto, a language name, or a Whisper language code.</small></label>
      <label class="model"><span>Model</span><select bind:value={model} disabled={loading || submitting || available.length === 0} required><option value="" disabled>Choose a model</option>{#each available as item (item.model_id)}<option value={item.model_id}>{item.name}</option>{/each}</select><small>{available.length === 0 && !loading ? task === 'translate' ? 'No registered translation-capable model. Install and register one in Models.' : 'No registered model is available. Install and register one in Models.' : !loading && !model ? 'The configured default is unavailable for this task. Choose a registered model or update the default in Settings.' : task === 'translate' ? 'Turbo models cannot translate and are not listed.' : 'Only registered models are listed.'}</small></label>
    </div>
    <div class="footer"><p>Loading a speech model may unload another model in the engine.</p><button class="primary" type="submit" disabled={loading || submitting || !settings || !file || !model}>{submitting ? 'Submitting…' : 'Submit job'}</button></div>
  </form>
</section>

<style>
  .upload{color:var(--lu-ink);max-width:64rem}.heading{margin-bottom:1.5rem}.heading h1,.heading p,h2,p{margin:0}.heading h1{font-size:clamp(1.65rem,3vw,2.1rem);letter-spacing:-.025em}.heading p,.form-heading p{color:var(--lu-ink-2);margin-top:.4rem;line-height:1.5}form{background:var(--lu-sheet);border:1px solid var(--lu-edge);border-radius:var(--lu-radius-card);padding:clamp(1.1rem,3vw,1.6rem)}.form-heading{margin-bottom:1.4rem}.form-heading h2{font-size:1.1rem}.form-heading p{font-size:.9rem}label{display:grid;gap:.5rem;min-width:0;font-size:.9rem;font-weight:700}input,select{font:inherit;font-weight:400;min-height:var(--lu-target);box-sizing:border-box;width:100%;color:var(--lu-ink);background:var(--lu-glass);border:1px solid var(--lu-edge);border-radius:var(--lu-radius-control);padding:.6rem .8rem}input[type=file]{padding:.55rem;cursor:pointer}input[type=file]::file-selector-button{font:inherit;color:var(--lu-ink);background:var(--lu-sheet);border:1px solid var(--lu-edge);border-radius:var(--lu-radius-control);padding:.45rem .7rem;margin-right:.8rem;cursor:pointer}small{color:var(--lu-ink-2);font-weight:400;line-height:1.45}.fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1.3rem;margin-top:1.4rem}.model{grid-column:1/-1}.footer{border-top:1px solid var(--lu-edge);margin-top:1.5rem;padding-top:1.2rem;display:flex;align-items:center;justify-content:space-between;gap:1rem}.footer p,.message{color:var(--lu-ink-2);font-size:.85rem;line-height:1.45}.primary{font:inherit;font-weight:700;min-height:var(--lu-target);padding:.6rem 1.15rem;background:var(--lu-accent);color:var(--lu-accent-ink);border:0;border-radius:var(--lu-radius-control);cursor:pointer;white-space:nowrap}.primary:disabled{opacity:.5;cursor:not-allowed}.error,.success{margin:1rem 0;overflow-wrap:anywhere}.error{color:var(--lu-danger)}.success{color:var(--lu-positive)}.text-button{font:inherit;color:inherit;border:0;padding:0;background:none;text-decoration:underline;cursor:pointer}input:focus-visible,select:focus-visible,button:focus-visible{outline:2px solid var(--lu-accent);outline-offset:2px}
  @media(max-width:600px){.fields{grid-template-columns:1fr}.model{grid-column:auto}.footer{align-items:stretch;flex-direction:column}}
  @media(max-width:400px){form{padding:12px;min-width:0}input,select{min-width:0;max-width:100%}.footer button{max-width:100%;overflow-wrap:anywhere}}
</style>
