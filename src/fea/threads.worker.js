// A helper of Threads (threads.js): takes chunks of each call's nodes, waiting on Atomics in between.
import { work } from './threads.js';

self.onmessage = (ev) => {
  const s = ev.data, ctrl = s.ctrl;
  let gen = Atomics.load(ctrl, 0);
  self.postMessage('ready');
  for (;;) {
    Atomics.wait(ctrl, 0, gen);
    gen = Atomics.load(ctrl, 0);
    work(s);
    Atomics.add(ctrl, 2, 1);
    Atomics.notify(ctrl, 2);
  }
};
