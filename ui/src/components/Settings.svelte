<script lang="ts">
  import { onMount } from 'svelte';
  import { api, type Model, type Settings, type EngineProbe } from '../api';

  let settings: Settings | null = null;
  let original: Settings | null = null;
  let models: Model[] = [];
  let loading = true;
  let saving = false;
  let testing = false;
  let error = '';
  let modelError = '';
  let notice = '';
  let probe: EngineProbe | null = null;
  let theme: 'system' | 'light' | 'dark' = 'system';
  let endpoint = '';
  let copied = '';
  let webhookCopied = '';
  let disposed = false;

  const message = (reason: unknown) => reason instanceof Error ? reason.message : 'Something went wrong. Please try again.';
  $: transcriptionModels = models.filter(model => model.registered);
  $: webhook = endpoint + '/api/captions/webhook';
  $: translationModels = models.filter(model => model.registered && model.translation_capable);
  $: detectionModels = models.filter(model => model.registered && model.language_detection_capable);

  function applyTheme(value: 'system' | 'light' | 'dark') {
    theme = value;
    if (value === 'system') delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = value;
    try { localStorage.setItem('speechdock-theme', value); } catch { notice = 'Theme changed for this session, but could not be saved in this browser.'; }
  }

  async function save() {
    if (!settings || !original || saving) return;
    const defaultModel = models.find(model => model.model_id === settings?.transcription_model);
    if (defaultModel?.requires_language && (!settings.default_language.trim() || settings.default_language.trim().toLowerCase() === 'auto')) {
      error = 'The default transcription model needs an explicit default language.'; return;
    }
    const patch: Partial<Settings> = {};
    for (const key of Object.keys(settings) as (keyof Settings)[]) {
      if (settings[key] !== original[key]) Object.assign(patch, { [key]: settings[key] });
    }
    if (!Object.keys(patch).length) { notice = 'No changes to save.'; return; }
    saving = true; error = ''; notice = '';
    try {
      const result = await api.saveSettings(patch);
      settings = { ...result }; original = { ...result };
      probe = null; notice = 'Settings saved.';
    } catch (reason) { error = message(reason); }
    finally { saving = false; }
  }

  async function testEngine() {
    testing = true; probe = null; error = '';
    try { probe = await api.testEngine(); }
    catch (reason) { error = message(reason); }
    finally { testing = false; }
  }

  async function copyEndpoint() {
    try { await navigator.clipboard.writeText(endpoint); copied = 'Copied.'; }
    catch { copied = 'Could not copy automatically. Select and copy the address instead.'; }
  }

  async function copyWebhook() {
    try { await navigator.clipboard.writeText(webhook); webhookCopied = 'Copied.'; }
    catch { webhookCopied = 'Could not copy automatically. Select and copy the address instead.'; }
  }

  onMount(() => {
    endpoint = window.location.origin;
    try {
      const saved = localStorage.getItem('speechdock-theme');
      if (saved === 'light' || saved === 'dark' || saved === 'system') theme = saved;
    } catch { /* Theme remains usable for this session. */ }
    void Promise.allSettled([api.settings(), api.models()]).then(([settingsResult, modelsResult]) => {
      if (disposed) return;
      if (settingsResult.status === 'fulfilled') { settings = { ...settingsResult.value }; original = { ...settingsResult.value }; }
      else error = message(settingsResult.reason);
      if (modelsResult.status === 'fulfilled') models = modelsResult.value.items;
      else modelError = `Model choices could not be loaded: ${message(modelsResult.reason)}`;
      loading = false;
    });
    return () => { disposed = true; };
  });
</script>

<section class="settings" aria-labelledby="settings-heading">
  <header><h1 id="settings-heading">Settings</h1><p>Configure your speech engine and defaults.</p></header>
  {#if error}<p class="error" role="alert">{error}</p>{/if}
  {#if notice}<p class="notice" role="status">{notice}</p>{/if}
  {#if loading}<p class="muted" role="status">Loading settings…</p>{/if}
  {#if settings}
    <form onsubmit={(event) => { event.preventDefault(); void save(); }}>
      <section class="group" aria-labelledby="engine-heading">
        <div class="group-intro"><h2 id="engine-heading">Speech engine</h2><p>SpeechDock connects to your existing llama-swap service.</p></div>
        <div class="fields">
          <label>Engine URL<input type="url" required bind:value={settings.engine_url} autocomplete="url" /></label>
          <div class="test-row"><button type="button" class="secondary" onclick={testEngine} disabled={testing}>{testing ? 'Testing…' : 'Test connection'}</button><span class="muted">Tests the saved URL. Save changes first.</span></div>
          {#if probe}<p role="status" class:error={!probe.reachable} class="probe">{probe.reachable ? `Connected. ${probe.models.length} model${probe.models.length === 1 ? '' : 's'} reported by the engine.` : probe.error || 'The engine could not be reached.'}</p>{/if}
          <div class="columns"><label>Connection timeout <span>seconds</span><input type="number" required min="1" max="300" step="1" bind:value={settings.connect_timeout_seconds} /></label><label>Processing timeout <span>seconds</span><input type="number" required min="1" max="86400" step="1" bind:value={settings.inference_timeout_seconds} /></label></div>
        </div>
      </section>
      <section class="group" aria-labelledby="defaults-heading">
        <div class="group-intro"><h2 id="defaults-heading">Defaults</h2><p>Used when a request does not specify its own options.</p></div>
        <div class="fields">
          {#if modelError}<p class="error" role="alert">{modelError}</p>{/if}
          <label>Transcription model
            <select bind:value={settings.transcription_model} disabled={!!modelError}>
              {#if !transcriptionModels.some(model => model.model_id === settings?.transcription_model)}<option value={settings.transcription_model}>{settings.transcription_model} (not currently available)</option>{/if}
              {#each transcriptionModels as model (model.id)}<option value={model.model_id}>{model.name}</option>{/each}
            </select>
          </label>
          <label>Translation model
            <select bind:value={settings.translation_model} disabled={!!modelError}>
              {#if !translationModels.some(model => model.model_id === settings?.translation_model)}<option value={settings.translation_model}>{settings.translation_model} (not currently available)</option>{/if}
              {#each translationModels as model (model.id)}<option value={model.model_id}>{model.name}</option>{/each}
            </select>
            <span>Translation needs a registered full Whisper model; Turbo models do not support it.</span>
          </label>
          <label>Language detection model
            <select bind:value={settings.detection_model} disabled={!!modelError}>
              {#if !detectionModels.some(model => model.model_id === settings?.detection_model)}<option value={settings.detection_model}>{settings.detection_model} (not currently available)</option>{/if}
              {#each detectionModels as model (model.id)}<option value={model.model_id}>{model.name}</option>{/each}
            </select>
            <span>Only registered models capable of detecting language are listed.</span>
          </label>
          <label>Default language<input required type="text" bind:value={settings.default_language} spellcheck="false" /><span>Use auto for detection, or a supported language name or code, such as English or en.</span></label>
          <div class="columns"><label>Maximum upload <span>MB</span><input type="number" required min="1" max="10240" step="1" bind:value={settings.max_upload_mb} /></label><label>History retention <span>days</span><input type="number" required min="1" max="3650" step="1" bind:value={settings.retention_days} /></label></div>
        </div>
      </section>
      <section class="group" aria-labelledby="captions-heading">
        <div class="group-intro"><h2 id="captions-heading">Dub captions</h2><p>Create English captions from a video's English audio when Sonarr or Radarr imports it.</p></div>
        <div class="fields">
          <label class="toggle"><input type="checkbox" bind:checked={settings.auto_captions} /><span><strong>Create dub captions for new imports</strong>When on, Sonarr/Radarr imports and upgrades sent to the webhook below are queued behind Bazarr requests and captioned from the English audio. Files SpeechDock did not create are never overwritten.</span></label>
          <label for="captions-webhook">Webhook address</label>
          <div class="copy-row"><input id="captions-webhook" value={webhook} readonly onclick={(event) => event.currentTarget.select()} /><button type="button" class="secondary" onclick={copyWebhook} disabled={!endpoint}>Copy</button></div>
          {#if webhookCopied}<p class="muted" role="status">{webhookCopied}</p>{/if}
          <p class="muted">In Sonarr or Radarr, add Connect → Webhook with this address, method POST, and the triggers On Import and On Upgrade.</p>
        </div>
      </section>
      <div class="save"><button class="primary" type="submit" disabled={saving}>{saving ? 'Saving…' : 'Save settings'}</button></div>
    </form>
  {/if}
  <section class="group" aria-labelledby="bazarr-heading">
    <div class="group-intro"><h2 id="bazarr-heading">Bazarr</h2><p>Use this SpeechDock address when connecting Bazarr to the Whisper ASR service.</p></div>
    <div class="fields"><label for="bazarr-address">Service address</label><div class="copy-row"><input id="bazarr-address" value={endpoint} readonly onclick={(event) => event.currentTarget.select()} /><button type="button" class="secondary" onclick={copyEndpoint} disabled={!endpoint}>Copy</button></div>{#if copied}<p class="muted" role="status">{copied}</p>{/if}</div>
  </section>
  <section class="group" aria-labelledby="appearance-heading">
    <div class="group-intro"><h2 id="appearance-heading">Appearance</h2><p>Saved in this browser.</p></div>
    <fieldset class="theme"><legend>Theme</legend>{#each ['system', 'light', 'dark'] as option}<label><input type="radio" name="theme" value={option} checked={theme === option} onchange={() => applyTheme(option as 'system' | 'light' | 'dark')} />{option === 'system' ? 'System' : option === 'light' ? 'Light' : 'Dark'}</label>{/each}</fieldset>
  </section>
</section>

<style>
  .settings, form { display: grid; gap: 1.25rem; min-width: 0; } h1, h2, p { margin: 0; } h1 { font-size: clamp(1.6rem, 3vw, 2.2rem); letter-spacing: -.03em; } h2 { font-size: 1.05rem; }
  header p, .group-intro p { color: var(--lu-ink-2); line-height: 1.5; margin-top: .4rem; }
  .group { display: grid; grid-template-columns: minmax(11rem, .8fr) minmax(0, 1.6fr); gap: 1.5rem; padding: 1.3rem; background: var(--lu-sheet); border: 1px solid var(--lu-edge); border-radius: var(--lu-radius-card); }
  .fields { display: grid; gap: 1rem; min-width: 0; } .fields > label, .columns label { display: grid; gap: .4rem; font-size: .9rem; font-weight: 700; min-width: 0; } label span { color: var(--lu-ink-2); font-size: .82rem; font-weight: 400; line-height: 1.45; }
  input:not([type=radio]):not([type=checkbox]), select { width: 100%; box-sizing: border-box; min-height: var(--lu-target); border: 1px solid var(--lu-edge); border-radius: var(--lu-radius-control); background: var(--lu-glass); color: var(--lu-ink); padding: .6rem .75rem; font: inherit; }
  input:focus-visible, select:focus-visible, button:focus-visible { outline: 2px solid var(--lu-accent); outline-offset: 3px; }
  .fields > label.toggle { display: flex; align-items: flex-start; gap: .7rem; } .toggle input { flex: none; width: 1.25rem; height: 1.25rem; margin: .15rem 0 0; accent-color: var(--lu-accent); } .toggle span { display: grid; gap: .25rem; min-width: 0; } .toggle strong { color: var(--lu-ink); font-size: .9rem; }
  .columns { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; } .test-row, .copy-row { display: flex; flex-wrap: wrap; align-items: center; gap: .7rem; } .copy-row input { flex: 1 1 14rem; min-width: 0; }
  button { min-height: var(--lu-target); border: 1px solid var(--lu-edge); border-radius: var(--lu-radius-control); padding: .55rem .9rem; font: inherit; font-weight: 700; cursor: pointer; } button:disabled { opacity: .55; cursor: not-allowed; }
  .primary { background: var(--lu-accent); color: var(--lu-accent-ink); } .secondary { background: var(--lu-glass); color: var(--lu-ink); } .save { display: flex; justify-content: flex-end; }
  .error { color: var(--lu-danger); overflow-wrap: anywhere; } .notice, .probe { color: var(--lu-positive); } .probe.error { color: var(--lu-danger); } .muted { color: var(--lu-ink-2); font-size: .88rem; line-height: 1.5; }
  .theme { border: 0; padding: 0; margin: 0; display: flex; gap: .75rem 1.25rem; flex-wrap: wrap; } .theme legend { font-size: .9rem; font-weight: 700; margin-bottom: .7rem; } .theme label { display: flex; align-items: center; gap: .45rem; min-height: var(--lu-target); } .theme input { accent-color: var(--lu-accent); }
  @media (max-width: 650px) { .group { grid-template-columns: 1fr; gap: 1rem; } } @media (max-width: 400px) { .columns { grid-template-columns: 1fr; } }
  @media (max-width: 400px) { .group { padding: 12px; } .copy-row input { flex-basis: 100%; } .fields, .group-intro, .theme { min-width: 0; overflow-wrap: anywhere; } button { max-width: 100%; overflow-wrap: anywhere; } }
</style>
