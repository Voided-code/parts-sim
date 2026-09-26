import { defineConfig } from 'vite';

export default defineConfig({
  // relative asset paths so the static build works from any sub-folder (e.g. GitHub Pages)
  base: './',
  worker: { format: 'es' },
  optimizeDeps: { include: ['occt-import-js'] },
  build: { target: 'es2022', chunkSizeWarningLimit: 1500 },
});
