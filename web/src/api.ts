export class ApiError extends Error {
  constructor(message: string, public readonly status: number) { super(message); this.name = 'ApiError'; }
}

export async function api<T>(path: string, options: { method?: string; body?: unknown; signal?: AbortSignal } = {}): Promise<T> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  let timedOut = false;
  if (options.signal?.aborted) abort();
  else options.signal?.addEventListener('abort', abort, { once: true });
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, (options.method ?? 'GET') === 'GET' ? 8000 : 45000);
  try {
    const response = await fetch(`/api${path}`, {
      method: options.method ?? 'GET',
      headers: options.body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      signal: controller.signal,
      cache: 'no-store',
    });
    let payload: { detail?: unknown };
    try { payload = await response.json(); }
    catch (error) {
      if (controller.signal.aborted) throw error;
      throw new ApiError('The server returned an unreadable response. Check that the API server is running on port 8000.', response.status);
    }
    if (!response.ok) {
      const detail = typeof payload.detail === 'string' ? payload.detail : 'The request could not be completed. Check your input and try again.';
      throw new ApiError(detail, response.status);
    }
    return payload as T;
  } catch (error) {
    if (options.signal?.aborted) throw new DOMException('The request was cancelled.', 'AbortError');
    if (timedOut) throw new ApiError('The request timed out. Reconnect before retrying a list change; the server may have received it.', 0);
    if (error instanceof ApiError) throw error;
    if (error instanceof Error && error.name === 'AbortError') throw error;
    throw new ApiError('The local server is unreachable. Check that the backend is running, then reconnect.', 0);
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener('abort', abort);
  }
}
