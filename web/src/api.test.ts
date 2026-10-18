import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from './api';

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

describe('API recovery boundaries', () => {
  it('preserves server validation errors', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ detail: 'Review the label first.' }), { status: 409 })));
    await expect(api('/sessions/one/confirm', { method: 'POST' })).rejects.toMatchObject({ status: 409, message: 'Review the label first.' });
  });
  it('does not accept HTML from a missing API proxy as valid application state', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('<html>Frontend index</html>', { status: 200 })));
    await expect(api('/health')).rejects.toThrow('unreadable response');
  });
  it('bounds a hung observer poll so a later poll can reconnect', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('fetch', vi.fn((_url: string, options: RequestInit) => new Promise((_resolve, reject) => {
      options.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
    })));
    const request = api('/sessions/one/latest');
    const assertion = expect(request).rejects.toThrow('timed out');
    await vi.advanceTimersByTimeAsync(8000);
    await assertion;
  });
  it('keeps an explicit cancellation distinct from a network failure', async () => {
    const controller = new AbortController();
    vi.stubGlobal('fetch', vi.fn((_url: string, options: RequestInit) => new Promise((_resolve, reject) => {
      options.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
    })));
    const request = api('/sessions/one/turn', { method: 'POST', signal: controller.signal });
    const assertion = expect(request).rejects.toMatchObject({ name: 'AbortError' });
    controller.abort();
    await assertion;
  });
});
