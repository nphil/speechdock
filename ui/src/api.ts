export type Task = 'transcribe' | 'translate' | 'detect';
export type JobStatus = 'queued' | 'running' | 'cancelling' | 'completed' | 'failed' | 'cancelled';
export interface Job {
  id: string; source: 'bazarr' | 'upload' | 'api'; filename: string; task: Task; language: string; model: string; status: JobStatus;
  created_at: string; started_at: string | null; finished_at: string | null; progress: number | null; duration_seconds: number | null;
  error: string | null; result_available: boolean; result_text?: string | null;
  detection?: { language_code: string; detected_language: string } | null;
}
export interface Settings { engine_url: string; transcription_model: string; translation_model: string; detection_model: string; inference_timeout_seconds: number; connect_timeout_seconds: number; max_upload_mb: number; retention_days: number; default_language: string }
export interface Download { id: string; status: string; received_bytes: number; total_bytes: number | null; progress: number | null; error: string | null }
export interface Model { id: string; name: string; description: string; model_id: string; filename: string | null; size_bytes: number | null; sha256: string | null; multilingual: boolean; translation_capable: boolean; quality: string | null; speed: string | null; installed: boolean; verified: boolean; registered: boolean; read_only: boolean; download: Download | null; can_register: boolean; warning: string; runtime: string | null; architecture: string | null; format: string | null; source: string; source_host: string | null; source_filename: string | null; language_detection_capable: boolean; requires_language: boolean; checksum_status: string; runtime_ready: boolean; runtime_error: string | null }
export interface ImportRequest { name: string; source: 'upload' | 'url'; url?: string; expected_sha256?: string; expected_size_bytes?: number; source_filename?: string }
export interface EngineProbe { reachable: boolean; models: string[]; error: string | null }
export interface Overview { version: string; engine: EngineProbe & { url: string }; queue: { waiting: number; running: number }; counts: { completed: number; failed: number }; storage: { free_bytes: number }; warnings: string[] }
export class HttpError extends Error { constructor(message: string, public status: number) { super(message); } }
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(path, init); } catch { throw new Error('Cannot reach SpeechDock. Check your connection and try again.'); }
  if (!response.ok) {
    let message = response.statusText || 'Request failed';
    try { const body: unknown = await response.json(); if (body && typeof body === 'object' && 'detail' in body) { const detail = body.detail; message = typeof detail === 'string' ? detail : JSON.stringify(detail); } } catch { /* HTTP status is still useful */ }
    throw new HttpError(message, response.status);
  }
  return response.json() as Promise<T>;
}
const json = (method: string, data: unknown): RequestInit => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
const modelPath = (id: string) => '/api/models/' + encodeURIComponent(id);
const jobPath = (id: string) => '/api/jobs/' + encodeURIComponent(id);
export const api = {
  overview: () => request<Overview>('/api/overview'),
  settings: () => request<Settings>('/api/settings'),
  saveSettings: (patch: Partial<Settings>) => request<Settings>('/api/settings', json('PATCH', patch)),
  testEngine: () => request<EngineProbe>('/api/engine/test', {method:'POST'}),
  models: () => request<{items: Model[]; downloads: Download[]; runtimes: Record<string, {available: boolean; version: string | null; error: string | null}>}>('/api/models'),
  createImport: (data: ImportRequest) => request<{model: Model; download: Download}>('/api/models/imports', json('POST', data)),
  removeImport: (id: string) => request<unknown>(modelPath(id) + '/import?confirm=true', {method:'DELETE'}),
  uploadModel: (id: string, file: File, signal: AbortSignal, onProgress: (progress: number | null) => void) => new Promise<void>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const abort = () => xhr.abort();
    if (signal.aborted) { reject(new DOMException('Upload cancelled', 'AbortError')); return; }
    signal.addEventListener('abort', abort, { once: true });
    const finish = () => signal.removeEventListener('abort', abort);
    xhr.open('PUT', modelPath(id) + '/upload');
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.upload.onprogress = event => onProgress(event.lengthComputable ? event.loaded / event.total : null);
    xhr.onload = () => { finish(); if (xhr.status >= 200 && xhr.status < 300) resolve(); else { let detail = xhr.statusText || 'Upload failed'; try { const body = JSON.parse(xhr.responseText); if (body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail); } catch { /* Keep HTTP status text. */ } reject(new Error(detail)); } };
    xhr.onerror = () => { finish(); reject(new Error('Cannot reach SpeechDock. Check your connection and try again.')); };
    xhr.onabort = () => { finish(); reject(new DOMException('Upload cancelled', 'AbortError')); };
    xhr.send(file);
  }),
  downloadModel: (id: string) => request<Download>(modelPath(id) + '/download', {method:'POST'}),
  cancelDownload: (id: string) => request<Download>(modelPath(id) + '/cancel-download', {method:'POST'}),
  registerModel: (id: string) => request<unknown>(modelPath(id) + '/register', json('POST', {confirm_reload:true})),
  unregisterModel: (id: string) => request<unknown>(modelPath(id) + '/registration?confirm_reload=true', {method:'DELETE'}),
  deleteModel: (id: string) => request<unknown>(modelPath(id) + '/files?confirm=true', {method:'DELETE'}),
  jobs: (limit=50, offset=0, status?: string) => { const q = new URLSearchParams({limit:String(limit), offset:String(offset)}); if(status) q.set('status', status); return request<{items: Job[]; total: number}>('/api/jobs?' + q); },
  job: (id: string) => request<Job>(jobPath(id)),
  cancelJob: (id: string) => request<Job>(jobPath(id) + '/cancel', {method:'POST'}),
  deleteJob: (id: string) => request<{deleted: boolean}>(jobPath(id) + '?confirm=true', {method:'DELETE'}),
  resultUrl: (id: string, format: 'srt' | 'vtt' | 'txt') => jobPath(id) + '/result?format=' + format,
  submitJob: (file: File, task: Task, language: string, model?: string) => { const form = new FormData(); form.set('file', file); form.set('task', task); form.set('language', language); if(model) form.set('model', model); return request<Job>('/api/jobs', {method:'POST',body:form}); }
};
