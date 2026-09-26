// Interactive selection of surface patches: click CAD faces, or paint a circular area.
import { trianglesInSphere } from '../core/mesh.js';
import { $, $$ } from './dom.js';

function faceIndex(part) {
  if (part._faceIndex && part._faceIndex.count === part.faceCount && part._faceIndex.faceOf === part.faceOf) return part._faceIndex;
  const start = new Int32Array(part.faceCount + 1);
  for (let t = 0; t < part.nTri; t++) start[part.faceOf[t] + 1]++;
  for (let f = 0; f < part.faceCount; f++) start[f + 1] += start[f];
  const fill = start.slice(0, part.faceCount);
  const list = new Int32Array(part.nTri);
  for (let t = 0; t < part.nTri; t++) list[fill[part.faceOf[t]]++] = t;
  part._faceIndex = { start, list, count: part.faceCount, faceOf: part.faceOf };
  return part._faceIndex;
}

export function faceTriangles(part, face) {
  if (!Number.isInteger(face) || face < 0 || face >= part.faceCount) return new Int32Array(0);
  const idx = faceIndex(part);
  return idx.list.subarray(idx.start[face], idx.start[face + 1]);
}

export class Picker {
  constructor(app) {
    this.app = app;
    this.session = null;
    this.mode = 'face';
    this.el = $('#picker');
    this.radiusInput = $('#picker-radius');
    for (const b of $$('#picker-mode button')) {
      b.addEventListener('click', () => this.setMode(b.dataset.mode));
    }
    $('#picker-done').addEventListener('click', () => this.finish(true));
    $('#picker-cancel').addEventListener('click', () => this.finish(false));
    window.addEventListener('keydown', (e) => {
      if (!this.session) return;
      if (e.target instanceof HTMLElement && (e.target.matches('input, textarea, select') || e.target.isContentEditable)) return;
      if (e.key === 'Escape') this.finish(false);
      if (e.key === 'Enter') this.finish(true);
    });
  }

  get active() {
    return !!this.session;
  }

  get radius() {
    return (this.app.part.bbox.diag * Number(this.radiusInput.value)) / 100;
  }

  setMode(mode) {
    this.mode = mode;
    for (const b of $$('#picker-mode button')) b.classList.toggle('active', b.dataset.mode === mode);
    $('#picker-radius-wrap').hidden = mode !== 'brush';
    this.app.viewer.clearHover();
    this.updateHint();
  }

  /**
   * @param {{title: string, color: number, onPatch: (patch) => void, onDone: (commit: boolean) => void}} s
   */
  start(s) {
    if (this.session) this.finish(false);
    this.session = { ...s, count: 0 };
    $('#picker-title').textContent = s.title;
    this.el.hidden = false;
    this.app.viewport.classList.add('picking');
    this.setMode(this.mode);
  }

  updateHint() {
    if (!this.session) return;
    const what = this.mode === 'face' ? 'Click faces to add them' : 'Click to paint circular areas';
    this.app.hint(`${what}. Selected: ${this.session.count}. Press Enter or Done to finish, Esc to cancel.`);
  }

  move(ev) {
    if (!this.session) return;
    const v = this.app.viewer;
    const hit = v.pickPart(ev);
    if (!hit) { v.clearHover(); return; }
    if (this.mode === 'face') v.showHoverPatch(`f${hit.face}`, faceTriangles(this.app.part, hit.face), this.session.color);
    else v.showBrush(hit.rest, hit.normal, this.radius, this.session.color);
  }

  click(ev) {
    if (!this.session) return;
    const hit = this.app.viewer.pickPart(ev);
    if (!hit) return;
    const part = this.app.part;
    let patch;
    if (this.mode === 'face') {
      patch = { tris: Int32Array.from(faceTriangles(part, hit.face)), clip: null, label: `Face ${hit.face + 1}` };
    } else {
      const center = [hit.rest.x, hit.rest.y, hit.rest.z];
      const r = this.radius;
      patch = { tris: trianglesInSphere(part, hit.tri, center, r), clip: { center, radius: r }, label: 'Area' };
    }
    if (!patch.tris.length) return;
    this.session.count++;
    this.session.onPatch(patch);
    this.app.viewer.hoverKey = null;
    this.updateHint();
  }

  finish(commit) {
    if (!this.session) return;
    const s = this.session;
    this.session = null;
    this.el.hidden = true;
    this.app.viewport.classList.remove('picking');
    this.app.viewer.clearHover();
    this.app.hint(null);
    s.onDone(commit);
  }
}
