// Gives Node a browser-like module Worker (on worker_threads) and cross-origin isolation, so the
// solvers' shared-memory helper threads (core/threads.js) run in the tests. Import it first.
import { Worker as NodeWorker } from 'node:worker_threads';

const boot = new URL('./webworker-boot.mjs', import.meta.url);
globalThis.crossOriginIsolated = true;
globalThis.Worker = class {
  constructor(url) {
    // (not the test runner's flags, which would make each helper a test runner of its own)
    this.w = new NodeWorker(boot, { workerData: { url: String(url) }, execArgv: [] });
    this.w.unref(); // helpers must not keep the test process alive
    this.w.on('message', (data) => this.onmessage?.({ data }));
    this.w.on('error', (e) => this.onerror?.({ message: e.message, preventDefault() {} }));
  }
  postMessage(m, transfer) { this.w.postMessage(m, transfer); }
  terminate() { this.w.terminate(); }
};
