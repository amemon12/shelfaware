import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  // Read by Vite's local server only. Never expose this as a VITE_ browser variable.
  const env = loadEnv(mode, '..', '');
  const headers = env.SHELF_API_TOKEN ? { Authorization: `Bearer ${env.SHELF_API_TOKEN}` } : undefined;
  return {
    plugins: [react()],
    server: { port: 5173, proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true, headers } } },
    preview: { port: 4173 },
  };
});
