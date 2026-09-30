import { execSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { defineConfig } from 'vite';

// cross-origin isolation, so the solvers can share memory between threads (SharedArrayBuffer);
// a static host without these headers runs them on one thread
const isolation = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' };

// version and commit of this build, so benchmark reports from different builds can be told apart
const version = JSON.parse(readFileSync(new URL('./package.json', import.meta.url))).version;
let commit = 'unknown';
try {
  commit = execSync('git rev-parse --short HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
  if (execSync('git status --porcelain --untracked-files=no', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim()) commit += '-modified';
} catch { /* not a git checkout */ }

export default defineConfig({
  server: { headers: isolation },
  preview: { headers: isolation },
  // relative asset paths so the static build works from any sub-folder (e.g. GitHub Pages)
  base: './',
  worker: { format: 'es' },
  optimizeDeps: { include: ['occt-import-js'] },
  define: { __APP_VERSION__: JSON.stringify(version), __GIT_COMMIT__: JSON.stringify(commit) },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 1500,
    rollupOptions: { input: { main: 'index.html', bench: 'bench.html' } },
  },
});
