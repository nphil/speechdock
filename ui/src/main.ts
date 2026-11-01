import { mount } from 'svelte';
import App from './App.svelte';
import './app.css';

try {
  const theme = localStorage.getItem('speechdock-theme');
  if (theme === 'dark' || theme === 'light') document.documentElement.dataset.theme = theme;
} catch { /* The system theme remains available when storage is blocked. */ }

mount(App, { target: document.getElementById('app')! });
