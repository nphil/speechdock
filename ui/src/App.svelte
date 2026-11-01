<script lang="ts">
  import { onMount } from 'svelte';
  import { Activity as ActivityIcon, Boxes, Settings as SettingsIcon, Cpu, Plus } from 'lucide-svelte';
  import Activity from './components/Activity.svelte';
  import Upload from './components/Upload.svelte';
  import Models from './components/Models.svelte';
  import Settings from './components/Settings.svelte';
  import { api, type Overview } from './api';

  type Page = 'activity' | 'models' | 'settings';
  let page: Page = 'activity';
  let overview: Overview | null = null;
  let overviewError = '';
  let activityVersion = 0;
  let disposed = false;
  let checking = false;

  async function refreshOverview() {
    if (checking) return;
    checking = true;
    try {
      const data = await api.overview();
      if (!disposed) { overview = data; overviewError = ''; }
    } catch (error) {
      if (!disposed) { overview = null; overviewError = error instanceof Error ? error.message : 'Could not reach SpeechDock.'; }
    } finally { checking = false; }
  }
  function jobSubmitted() { activityVersion += 1; void refreshOverview(); }
  function navigate(next: Page) { page = next; window.scrollTo({ top: 0, behavior: 'instant' }); }
  onMount(() => {
    void refreshOverview();
    const timer = window.setInterval(() => void refreshOverview(), 30000);
    return () => { disposed = true; window.clearInterval(timer); };
  });
</script>

<div class="shell" class:workspace={page === 'activity'}>
  <aside class="rail" aria-label="SpeechDock navigation">
    <a href="#activity" class="brand" onclick={(event) => { event.preventDefault(); navigate('activity'); }}>
      <img src="/icon.svg" alt="" width="40" height="40" />
      <span><strong>SpeechDock</strong><small>Local speech workspace</small></span>
    </a>
    <nav class="nav" aria-label="Primary navigation">
      <button class:current={page === 'activity'} aria-current={page === 'activity' ? 'page' : undefined} onclick={() => navigate('activity')}><ActivityIcon size={20} aria-hidden="true" />Activity</button>
      <button class:current={page === 'models'} aria-current={page === 'models' ? 'page' : undefined} onclick={() => navigate('models')}><Boxes size={20} aria-hidden="true" />Models</button>
      <button class:current={page === 'settings'} aria-current={page === 'settings' ? 'page' : undefined} onclick={() => navigate('settings')}><SettingsIcon size={20} aria-hidden="true" />Settings</button>
    </nav>
    <div class="engine" aria-live="polite">
      <span class="engine-label">Shared engine</span>
      <div class="engine-row"><Cpu size={20} aria-hidden="true" /><div><strong>llama-swap</strong><span>{overviewError ? 'SpeechDock unavailable' : !overview ? 'Checking connection…' : overview.engine.reachable ? 'Connected' : 'Not reachable'}</span></div></div>
      {#if overviewError}<span class="engine-error">{overviewError}</span>{:else if overview && !overview.engine.reachable && overview.engine.error}<span class="engine-error">{overview.engine.error}</span>{/if}
    </div>
  </aside>
  <main id="main-content" tabindex="-1">
    {#if page === 'activity'}
      <div class="mobile-upload-link"><a href="#new-job"><Plus size={18} aria-hidden="true" /> New job</a></div>
      {#if overview?.warnings.length}<div class="warnings" role="status">{#each overview.warnings as warning}<p>{warning}</p>{/each}</div>{/if}
      {#key activityVersion}<Activity />{/key}
    {:else if page === 'models'}<Models />
    {:else}<Settings />{/if}
  </main>
  {#if page === 'activity'}<aside class="composer" id="new-job" aria-label="New job"><Upload onChanged={jobSubmitted} /></aside>{/if}
  <nav class="mobile-nav" aria-label="Mobile navigation">
    <button class:current={page === 'activity'} aria-current={page === 'activity' ? 'page' : undefined} onclick={() => navigate('activity')}><ActivityIcon size={20} aria-hidden="true" />Activity</button>
    <button class:current={page === 'models'} aria-current={page === 'models' ? 'page' : undefined} onclick={() => navigate('models')}><Boxes size={20} aria-hidden="true" />Models</button>
    <button class:current={page === 'settings'} aria-current={page === 'settings' ? 'page' : undefined} onclick={() => navigate('settings')}><SettingsIcon size={20} aria-hidden="true" />Settings</button>
  </nav>
</div>

<style>
  .shell { min-height: 100vh; display: grid; grid-template-columns: 14rem minmax(0, 1fr); background: var(--lu-canvas); }
  .shell.workspace { grid-template-columns: 14rem minmax(0, 1fr) minmax(19rem, 25rem); }
  .rail { display: flex; flex-direction: column; gap: 2.5rem; padding: 2rem 1rem; background: var(--lu-sheet); border-right: 1px solid var(--lu-material-edge); min-width: 0; }
  .brand { display: flex; align-items: center; gap: .7rem; padding: 0 .5rem; text-decoration: none; min-width: 0; }
  .brand img { border-radius: var(--lu-radius-control); flex: none; }
  .brand span, .engine-row div { display: grid; min-width: 0; }
  .brand strong { font-size: 1.05rem; letter-spacing: -.03em; }
  .brand small, .engine-label, .engine-row span, .engine-error { color: var(--lu-ink-2); font-size: .75rem; }
  .nav { display: grid; gap: .35rem; }
  .nav button, .mobile-nav button { font: inherit; color: var(--lu-ink-2); background: transparent; border: 0; display: flex; align-items: center; gap: .8rem; text-align: left; border-radius: var(--lu-radius-control); padding: .75rem .85rem; font-weight: 650; }
  .nav button:hover, .mobile-nav button:hover { background: var(--lu-material-hover-wash); }
  .nav button.current, .mobile-nav button.current { color: var(--lu-ink); background: var(--lu-material-selected-wash); }
  .current :global(svg) { color: var(--lu-accent); }
  .engine { margin-top: auto; padding: 1.25rem .5rem 0; border-top: 1px solid var(--lu-material-edge); display: grid; gap: .7rem; overflow-wrap: anywhere; }
  .engine-row { display: flex; align-items: center; gap: .65rem; }
  .engine-row strong { font-size: .85rem; }
  main { padding: 3rem; min-width: 0; }
  .composer { background: var(--lu-sheet); border-left: 1px solid var(--lu-material-edge); padding: 3rem 1.5rem; min-width: 0; }
  .warnings { margin-bottom: 1.5rem; padding: 1rem; background: var(--lu-sheet); border-radius: var(--lu-radius-control); overflow-wrap: anywhere; }
  .warnings p { margin: 0; } .warnings p + p { margin-top: .5rem; }
  .mobile-nav, .mobile-upload-link { display: none; }
  @media (max-width: 1150px) { .shell.workspace { grid-template-columns: 12rem minmax(0,1fr) minmax(18rem, 21rem); } main { padding: 2rem; } .composer { padding: 2rem 1rem; } }
  @media (max-width: 850px) { .shell, .shell.workspace { grid-template-columns: 1fr; padding-bottom: 5.5rem; } .rail { display: none; } main { padding: 1.5rem; } .composer { border-left: 0; border-top: 1px solid var(--lu-material-edge); padding: 1.5rem; } .mobile-nav { display: flex; position: fixed; z-index: 10; bottom: 0; left: 0; right: 0; justify-content: space-around; background: var(--lu-material-overlay); border-top: 1px solid var(--lu-material-edge); padding: .35rem .5rem max(.35rem, env(safe-area-inset-bottom)); } .mobile-nav button { flex-direction: column; gap: .15rem; font-size: .75rem; padding: .4rem 1rem; } .mobile-upload-link { display: flex; justify-content: flex-end; margin-bottom: 1rem; } .mobile-upload-link a { display: inline-flex; align-items: center; gap: .4rem; font-weight: 700; } }
  @media (max-width: 400px) { main, .composer { padding: 1rem; } .mobile-nav button { padding-inline: .7rem; } }
  @media (max-width: 400px) { .shell, .shell.workspace { grid-template-columns: minmax(0,1fr); padding-bottom: 6.5rem; } main, .composer { padding: 12px; } .mobile-nav { padding-inline: 0; } .mobile-nav button { flex: 1 1 0; min-width: 0; padding: .3rem 0; font-size: .625rem; text-align: center; overflow-wrap: anywhere; } }
</style>
