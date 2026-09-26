// Tessellates STEP / IGES / BREP files with OpenCascade (occt-import-js, WebAssembly) off the main thread.
import occtimportjs from 'occt-import-js';
import wasmUrl from 'occt-import-js/dist/occt-import-js.wasm?url';
import { readCADMesh } from './cad.js';

let occtReady = null;

self.onmessage = async (ev) => {
  const { format, buffer, quality } = ev.data;
  try {
    occtReady ??= occtimportjs({ locateFile: () => wasmUrl });
    const occt = await occtReady;
    const { positions, index, faceIds, bodies } = readCADMesh(occt, format, buffer, quality);
    self.postMessage({ ok: true, positions, index, faceIds, bodies }, [
      positions.buffer,
      index.buffer,
      faceIds.buffer,
    ]);
  } catch (err) {
    self.postMessage({ ok: false, error: err?.message || String(err) });
  }
};
