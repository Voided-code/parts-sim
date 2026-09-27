import { defineConfig } from 'vite';

// cross-origin isolation, so the solvers can share memory between threads (SharedArrayBuffer);
// a static host without these headers runs them on one thread
const isolation = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' };

export default defineConfig({
  server: { headers: isolation },
  preview: { headers: isolation },
  // relative asset paths so the static build works from any sub-folder (e.g. GitHub Pages)
  base: './',
  worker: { format: 'es' },
  optimizeDeps: { include: ['occt-import-js'] },
  build: { target: 'es2022', chunkSizeWarningLimit: 1500 },
});
