// A worker_threads worker that runs a web worker module (webworker.mjs): self, postMessage and
// onmessage as in a browser.
import { parentPort, workerData } from 'node:worker_threads';

globalThis.self = globalThis;
globalThis.crossOriginIsolated = true;
globalThis.postMessage = (m, transfer) => parentPort.postMessage(m, transfer);
const queue = [];
parentPort.on('message', (data) => (globalThis.onmessage ? globalThis.onmessage({ data }) : queue.push(data)));
await import(workerData.url);
for (const data of queue.splice(0)) globalThis.onmessage({ data });
