// Thermal tab: fixed temperatures, heat inputs and convection picked on faces, steady-state or
// transient heat conduction on the voxel model, and temperature / heat-flux results.
import { patchToJson, patchFromJson } from '../core/psim-patches.js';
import * as THREE from 'three';
import { FEAJob } from '../fea/structural.js';
import { completeMaterial } from '../core/materials.js';
import { $, $$, h, num, nextFrame } from './dom.js';
import { renderLegend } from './legend.js';
import { LineChart } from './chart.js';

const COLORS = { temp: 0xe0582b, heat: 0xc026d3, conv: 0x1f9bd6 };
const LABELS = { temp: 'Temperature', heat: 'Heat power', conv: 'Convection' };
let uid = 1;

export class ThermalPanel {
  constructor(app) {
    this.app = app;
    this.items = [];
    this.selected = null;
    this.result = null;
    this.mode = 'steady';
    this.view = { plot: 'T', frame: 0, playing: false, t: 0, bands: false, bcs: false };
    this.runVersion = 0;
    this.job = null;
    this.resolution = null;
    this.wire();
    this.chart = new LineChart($('#th-chart'), $('#th-tip'), {
      xLabel: 'time (s)', integerX: false, formatX: (x) => num(x), formatY: (y) => `${num(y)} °C`, yMin: undefined,
      tipText: (p) => `${num(p.x)} s · hottest ${num(p.y)} °C`,
      onPick: (i) => this.setFrame(i),
    });
    app.viewer.onFrame((dt) => this.tick(dt));
  }

  get part() { return this.app.part; }
  get material() { return completeMaterial(this.app.structural.material); }

  reset() {
    this.cancel();
    this.items = [];
    this.selected = null;
    this.result = null;
    this.resolution = null;
    $('#th-results').hidden = true;
    if (this.part) {
      const n = Number($('#res-range').value);
      $('#th-res').value = n;
      $('#th-res-out').textContent = n;
    }
    this.renderList();
    this.renderMaterial();
    this.updateMeshInfo();
  }

  onPartScaled(s, pivot) {
    this.cancel();
    const move = (c) => c.map((v, d) => pivot[d] + (v - pivot[d]) * s);
    for (const it of this.items) for (const p of it.patches) if (p.clip) p.clip = { center: move(p.clip.center), radius: p.clip.radius * s };
    this.result = null;
    $('#th-results').hidden = true;
    this.renderList();
    this.updateMeshInfo();
  }

  markStale() {
    this.cancel();
    if (this.result) {
      this.result.stale = true;
      this.app.status('Thermal setup changed - run it again to update the results.', 'warn');
    }
    this.drawOverlays();
  }

  wire() {
    $('#btn-th-temp').addEventListener('click', () => this.addItem('temp'));
    $('#btn-th-heat').addEventListener('click', () => this.addItem('heat'));
    $('#btn-th-conv').addEventListener('click', () => this.addItem('conv'));
    $('#th-ambient').addEventListener('change', (e) => { $('#th-ambient-props').hidden = !e.target.checked; this.markStale(); });
    for (const id of ['#th-ambient-h', '#th-ambient-t', '#th-duration', '#th-steps', '#th-initial']) $(id).addEventListener('change', () => this.markStale());
    for (const b of $$('#th-mode button')) {
      b.addEventListener('click', () => {
        this.mode = b.dataset.mode;
        for (const x of $$('#th-mode button')) x.classList.toggle('active', x === b);
        $('#th-transient').hidden = this.mode !== 'transient';
        this.markStale();
      });
    }
    const res = $('#th-res');
    res.addEventListener('input', () => { $('#th-res-out').textContent = res.value; });
    res.addEventListener('change', () => { this.markStale(); this.updateMeshInfo(); });
    $('#btn-th-run').addEventListener('click', () => this.run());
    $('#th-plot').addEventListener('change', (e) => { this.view.plot = e.target.value; this.show(); });
    $('#th-frame').addEventListener('input', (e) => { this.view.playing = false; this.setFrame(Number(e.target.value)); });
    $('#btn-th-play').addEventListener('click', () => this.togglePlay());
    $('#th-bands').addEventListener('change', (e) => { this.view.bands = e.target.checked; this.show(); });
    $('#th-bcs').addEventListener('change', (e) => { this.view.bcs = e.target.checked; this.drawOverlays(); });
  }

  renderMaterial() {
    const m = this.material;
    $('#th-mat-name').textContent = m.name;
    $('#th-mat-props').textContent = `k ${num(m.k)} W/mK · cp ${num(m.cp)} J/kgK`;
  }

  updateMeshInfo() {
    const el = $('#th-mesh-info');
    if (!this.part) { el.textContent = ''; return; }
    const res = Number($('#th-res').value);
    el.textContent = `Voxel size ≈ ${num(Math.max(...this.part.bbox.size) / res)} ${this.app.units}.`;
  }

  // ---------- heat inputs ----------

  addItem(type) {
    if (!this.part) return this.app.status('Import a part or open a sample first.', 'error');
    const n = this.items.filter((i) => i.type === type).length + 1;
    const defaults = { temp: { value: 80 }, heat: { value: 10 }, conv: { value: 25, ambient: 20 } };
    const it = { id: uid++, type, name: `${LABELS[type]} ${n}`, patches: [], ...defaults[type] };
    this.items.push(it);
    this.selected = it;
    this.renderList();
    this.pick(it, true);
  }

  pick(it, isNew = false) {
    this.showSetup();
    const before = it.patches.length;
    this.app.picker.start({
      title: it.name,
      color: COLORS[it.type],
      onPatch: (p) => {
        if (!p.tris.length) return;
        it.patches.push(p);
        this.markStale();
        this.renderList();
      },
      onDone: (commit) => {
        if (!commit) it.patches.length = before;
        if (!it.patches.length && isNew) {
          this.items = this.items.filter((x) => x !== it);
          if (this.selected === it) this.selected = this.items[0] || null;
        }
        this.renderList();
        this.drawOverlays();
      },
    });
  }

  renderList() {
    const ul = $('#th-list');
    const unit = { temp: '°C', heat: 'W', conv: 'W/m²K' };
    ul.replaceChildren(...this.items.map((it) => h('li', { className: it === this.selected ? 'selected' : '', onclick: () => { this.selected = it; this.renderList(); this.drawOverlays(); } },
      h('span.swatch', { style: { background: `#${COLORS[it.type].toString(16).padStart(6, '0')}` } }),
      h('span.name', {}, it.name),
      h('span.meta', {}, `${num(it.value)} ${unit[it.type]}${it.type === 'conv' ? ` → ${num(it.ambient)} °C` : ''}`),
      h('button.x', { title: 'Add more areas', onclick: (e) => { e.stopPropagation(); this.pick(it); } }, '+'),
      h('button.x', { title: 'Delete', onclick: (e) => { e.stopPropagation(); this.items = this.items.filter((x) => x !== it); if (this.selected === it) this.selected = this.items[0] || null; this.markStale(); this.renderList(); } }, '×'))));
    this.renderEditor();
  }

  renderEditor() {
    const ed = $('#th-editor');
    if (ed.contains(document.activeElement)) document.activeElement.blur();
    const it = this.selected;
    if (!it || !this.items.includes(it)) { ed.hidden = true; this.drawOverlays(); return; }
    ed.hidden = false;
    const input = (value, onchange) => {
      const el = h('input', { type: 'number', value, step: 'any', onchange: (e) => { onchange(Number(e.target.value)); this.markStale(); this.renderList(); } });
      el.style.cssText = 'height:28px;padding:0 8px;border:1px solid var(--border-strong);border-radius:7px;background:var(--panel);width:100%';
      return el;
    };
    const rows = [];
    if (it.type === 'temp') rows.push(h('label', {}, 'Temperature ', h('span.unit', {}, '°C'), input(it.value, (v) => { it.value = v; })));
    if (it.type === 'heat') rows.push(h('label', {}, 'Heat power (total) ', h('span.unit', {}, 'W'), input(it.value, (v) => { it.value = Math.max(0, v); })));
    if (it.type === 'conv') {
      rows.push(h('label', {}, 'Film coefficient ', h('span.unit', {}, 'W/m²K'), input(it.value, (v) => { it.value = Math.max(0, v); })));
      rows.push(h('label', {}, 'Fluid temperature ', h('span.unit', {}, '°C'), input(it.ambient, (v) => { it.ambient = v; })));
    }
    ed.replaceChildren(h('div.prop-grid', {}, ...rows),
      h('button.btn.small', { style: { marginTop: '6px' }, onclick: () => this.pick(it) }, '+ Area'));
    ed.style.marginTop = '8px';
    this.drawOverlays();
  }

  drawOverlays() {
    const v = this.app.viewer;
    if (this.app.tab !== 'thermal') return;
    v.clearOverlays();
    if (!this.part || !(this.display !== 'results' || this.view.bcs || this.app.picker.active)) return;
    for (const it of this.items) {
      const tris = new Set();
      for (const p of it.patches) for (const t of p.tris) tris.add(t);
      if (tris.size) v.overlayGroup.add(v.patchMesh(Int32Array.from(tris), COLORS[it.type], it === this.selected ? 0.6 : 0.42));
    }
  }

  // ---------- run ----------

  cancel() {
    this.runVersion++;
    if (this.job) {
      this.job.cancel();
      this.job = null;
      this.app.busy.hide();
      this.app.status('Heat transfer cancelled.', 'warn');
    }
  }

  assemble(model) {
    const nN = model.nNodes, toM = this.app.toMeters, A = toM * toM;
    const fixedNode = new Uint8Array(nN), fixedValue = new Float64Array(nN), source = new Float64Array(nN);
    const convH = new Float64Array(nN), convHT = new Float64Array(nN);
    const used = new Set();
    let heatIn = 0;
    const warnings = [];
    for (const it of this.items) {
      for (const p of it.patches) if (!p.clip) for (const t of p.tris) used.add(t);
      const s = model.samplePatches(it.patches);
      if (it.type === 'temp') {
        let hit = 0;
        const hold = (n) => { fixedNode[n] = 1; fixedValue[n] = it.value; };
        for (let i = 0; i < s.weights.length; i++) {
          const n = model.nearestSurfaceNode(s.points[3 * i], s.points[3 * i + 1], s.points[3 * i + 2]);
          if (n < 0) continue;
          hit++;
          hold(n);
          model.forSurfaceNodesNear(s.points[3 * i], s.points[3 * i + 1], s.points[3 * i + 2], 0.75, hold);
        }
        if (!hit) warnings.push(`${it.name}: no voxels under the selection.`);
        continue;
      }
      let W = 0;
      const nodes = [];
      for (let i = 0; i < s.weights.length; i++) {
        const n = model.nearestSurfaceNode(s.points[3 * i], s.points[3 * i + 1], s.points[3 * i + 2]);
        if (n < 0) continue;
        nodes.push([n, s.weights[i]]);
        W += s.weights[i];
      }
      if (!nodes.length) { warnings.push(`${it.name}: no voxels under the selection.`); continue; }
      for (const [n, w] of nodes) {
        if (it.type === 'heat') source[n] += (it.value * w) / W;
        else { convH[n] += it.value * w * A; convHT[n] += it.value * w * A * it.ambient; }
      }
      if (it.type === 'heat') heatIn += it.value;
    }
    if ($('#th-ambient').checked) {
      const hc = Number($('#th-ambient-h').value), Ta = Number($('#th-ambient-t').value);
      if (hc > 0) {
        const others = [];
        for (let t = 0; t < this.part.nTri; t++) if (!used.has(t)) others.push(t);
        const s = model.samplePatches([{ tris: Int32Array.from(others), clip: null }]);
        for (let i = 0; i < s.weights.length; i++) {
          const n = model.nearestSurfaceNode(s.points[3 * i], s.points[3 * i + 1], s.points[3 * i + 2]);
          if (n < 0) continue;
          convH[n] += hc * s.weights[i] * A;
          convHT[n] += hc * s.weights[i] * A * Ta;
        }
      }
    }
    const convT = new Float64Array(nN);
    for (let n = 0; n < nN; n++) convT[n] = convH[n] > 0 ? convHT[n] / convH[n] : 0;
    return { fixedNode, fixedValue, source, convH, convT, heatIn, warnings };
  }

  async run() {
    if (!this.part) return this.app.status('Import a part or open a sample first.', 'error');
    this.cancel();
    this.app.picker.finish(true);
    const version = this.runVersion;
    const mat = this.material;
    const transient = this.mode === 'transient';
    const duration = Number($('#th-duration').value), steps = Math.round(Number($('#th-steps').value)), initial = Number($('#th-initial').value);
    if (transient && (!(duration > 0) || !(steps >= 2))) return this.app.status('Set a positive duration and at least 2 time steps.', 'error');
    this.app.busy.show('Voxelizing part…', () => this.cancel());
    await nextFrame();
    if (version !== this.runVersion) return;
    let model, bc;
    try {
      model = this.app.structural.modelFor(Number($('#th-res').value));
      if (!model.voxelCount) throw new Error('The part produced no voxels. Increase the mesh resolution.');
      bc = this.assemble(model);
      if (!transient && !bc.fixedNode.some((x) => x) && !bc.convH.some((x) => x > 0)) {
        throw new Error('Steady state needs somewhere for the heat to go: add a fixed temperature or convection (or run it over time).');
      }
    } catch (err) {
      this.app.busy.hide();
      return this.app.status(err.message, 'error');
    }
    for (const w of bc.warnings) this.app.status(w, 'warn');
    const frames = [];
    const r = { frames, material: mat, transient, heatIn: bc.heatIn, units: this.app.units };
    try {
      this.job = new FEAJob({
        type: 'thermal', dims: model.dims, density: model.density, k: mat.k, h: model.h * this.app.toMeters,
        rhoCp: mat.density * mat.cp, duration: transient ? duration : 0, steps: transient ? steps : 1, initial,
        fixedNode: bc.fixedNode, fixedValue: bc.fixedValue, source: bc.source, convH: bc.convH, convT: bc.convT,
        map: model.mapping(),
      }, {
        onProgress: (d) => { if (version === this.runVersion) this.app.busy.progress(d.frac ?? 0.5, d.stage); },
        onStep: (d) => { if (version === this.runVersion) frames.push(d); },
      });
      this.app.busy.show(transient ? 'Heat transfer over time…' : 'Heat transfer…', () => this.cancel());
      const res = await this.job.promise;
      if (version !== this.runVersion) return;
      Object.assign(r, res);
    } catch (err) {
      if (version !== this.runVersion) return;
      this.app.busy.hide();
      this.job = null;
      return this.app.status(err.cancelled ? 'Cancelled.' : `Solver error: ${err.message}`, err.cancelled ? '' : 'error');
    }
    this.job = null;
    this.app.busy.hide();
    this.result = r;
    this.display = 'results';
    this.view.frame = Math.max(0, frames.length - 1);
    $('#th-results').hidden = false;
    $('#th-time').hidden = !transient;
    this.show();
    this.app.status(`Heat transfer on ${model.voxelCount.toLocaleString()} voxels: ${num(r.min)}–${num(r.max)} °C${transient ? ` after ${num(duration)} s` : ''}.${res0(r)}`);
    $('#th-results').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    this.app.runDone?.('thermal', r);
  }

  // ---------- display ----------

  showSetup() {
    this.display = 'setup';
    const v = this.app.viewer;
    v.setScalars(null);
    v.setDeformation(null);
    v.clearMarker();
    renderLegend($('#legend'), []);
    this.drawOverlays();
  }

  activate() {
    this.renderMaterial();
    this.updateMeshInfo();
    if (this.display === 'results' && this.result) this.show();
    else this.showSetup();
  }

  deactivate() {
    this.view.playing = false;
    const v = this.app.viewer;
    v.clearOverlays();
    v.clearMarker();
    v.setScalars(null);
  }

  setFrame(i) {
    const r = this.result;
    if (!r?.frames.length) return;
    this.view.frame = Math.max(0, Math.min(r.frames.length - 1, i));
    this.show();
  }

  current() {
    const r = this.result;
    if (!r) return null;
    if (r.transient && r.frames.length) {
      const fr = r.frames[Math.min(this.view.frame, r.frames.length - 1)];
      return { T: fr.T, t: fr.t, last: this.view.frame === r.frames.length - 1 };
    }
    return { T: r.T, t: 0, last: true };
  }

  show() {
    const r = this.result;
    if (!r || this.app.tab !== 'thermal') return;
    const cur = this.current();
    const v = this.app.viewer;
    const bands = this.view.bands ? 12 : 0;
    const flux = this.view.plot === 'flux' && cur.last;
    let values, min, max, title, fmt;
    if (flux) {
      values = r.flux;
      min = 0;
      max = 0;
      for (const x of values) if (!Number.isNaN(x)) max = Math.max(max, x);
      max ||= 1;
      title = 'Heat flux';
      fmt = (x) => (max >= 1e4 ? `${num(x / 1e3)} kW/m²` : `${num(x)} W/m²`);
    } else {
      values = cur.T;
      // fixed colour scale over the whole run so the animation compares like with like
      min = r.transient ? Math.min(...r.history.map((p) => p.min)) : r.min;
      max = r.transient ? Math.max(...r.history.map((p) => p.max)) : r.max;
      if (!(max > min)) max = min + 1;
      title = 'Temperature';
      fmt = (x) => `${num(x)} °C`;
    }
    v.setScalars(values, { min, max, bands, colormap: 'heat' });
    renderLegend($('#legend'), { title, sub: `${r.material.name}${r.transient ? ` · t = ${num(cur.t)} s` : ' · steady state'}${this.view.plot === 'flux' && !cur.last ? ' · flux shown at the end only' : ''}`, min, max, format: fmt, bands, colormap: 'heat' });
    this.shown = { values, fmt };
    // hottest point
    let hi = -Infinity, hot = -1;
    values.forEach((x, i) => { if (!Number.isNaN(x) && x > hi) { hi = x; hot = i; } });
    const V = this.part.vertices;
    if (hot >= 0) v.setMarker(new THREE.Vector3(V[3 * hot], V[3 * hot + 1], V[3 * hot + 2]), flux ? 'Highest heat flux' : 'Hottest point');
    this.drawOverlays();
    const kpi = (label, value, sub) => h('div.kpi', {}, h('div.k-label', {}, label), h('div.k-value', {}, value), sub ? h('div.k-sub', {}, sub) : null);
    let lo = Infinity, top = -Infinity;
    for (const x of cur.T) if (!Number.isNaN(x)) { lo = Math.min(lo, x); top = Math.max(top, x); }
    $('#th-kpis').replaceChildren(
      kpi('Hottest', `${num(top)} °C`, r.transient ? `at ${num(cur.t)} s` : 'steady state'),
      kpi('Coolest', `${num(lo)} °C`, `spread ${num(top - lo)} °C`),
      kpi('Heat input', `${num(r.heatIn)} W`, 'from heat-power faces'),
    );
    $('#th-alert').hidden = !r.stale;
    if (r.stale) $('#th-alert').textContent = 'The setup changed since this run - run it again to update.';
    if (r.transient) {
      const slider = $('#th-frame');
      slider.max = Math.max(0, r.frames.length - 1);
      slider.value = this.view.frame;
      $('#th-frame-out').textContent = `${num(cur.t)} s`;
      this.chart.set(r.history.map((p) => ({ x: p.t, y: p.max })), this.view.frame);
    }
  }

  probe(hit) {
    if (this.display !== 'results' || !this.shown) return null;
    const T = this.part.tris, t = hit.tri;
    const w = [hit.bary.x, hit.bary.y, hit.bary.z];
    let s = 0;
    for (let k = 0; k < 3; k++) s += w[k] * this.shown.values[T[3 * t + k]];
    return Number.isNaN(s) ? 'no data' : this.shown.fmt(s);
  }

  togglePlay() {
    const r = this.result;
    if (!r?.frames.length) return;
    this.view.playing = !this.view.playing;
    if (this.view.playing && this.view.frame >= r.frames.length - 1) this.view.frame = 0;
    $('#btn-th-play').textContent = this.view.playing ? '❚❚ Pause' : '▶ Play';
    this.view.t = 0;
  }

  tick(dt) {
    if (!this.view.playing || this.app.tab !== 'thermal' || !this.result) return;
    this.view.t += dt;
    if (this.view.t < 0.12) return;
    this.view.t = 0;
    if (this.view.frame >= this.result.frames.length - 1) {
      this.view.playing = false;
      $('#btn-th-play').textContent = '▶ Play';
      return;
    }
    this.setFrame(this.view.frame + 1);
  }

  // ---------- .psim file ----------

  exportState() {
    return {
      mode: this.mode,
      items: this.items.map((it) => ({ name: it.name, type: it.type, value: it.value, ambient: it.ambient, patches: it.patches.map(patchToJson) })),
      ambient: { enabled: $('#th-ambient').checked, h: Number($('#th-ambient-h').value), t: Number($('#th-ambient-t').value) },
      duration: Number($('#th-duration').value),
      steps: Number($('#th-steps').value),
      initial: Number($('#th-initial').value),
      resolution: Number($('#th-res').value),
    };
  }

  importState(s) {
    const nTri = this.part.nTri;
    this.items = (s.items || []).map((it) => {
      if (!['temp', 'heat', 'conv'].includes(it.type)) throw new Error(`The file has a thermal condition of an unknown type (${it.type}).`);
      return { id: uid++, name: String(it.name), type: it.type, value: Number(it.value), ambient: Number(it.ambient ?? 20), patches: (it.patches || []).map((p) => patchFromJson(p, nTri)).filter((p) => p.tris.length) };
    });
    this.selected = this.items[0] || null;
    this.mode = s.mode === 'transient' ? 'transient' : 'steady';
    for (const b of $$('#th-mode button')) b.classList.toggle('active', b.dataset.mode === this.mode);
    $('#th-transient').hidden = this.mode !== 'transient';
    const a = s.ambient || {};
    $('#th-ambient').checked = a.enabled !== false;
    $('#th-ambient-props').hidden = a.enabled === false;
    const put = (sel, v) => { if (Number.isFinite(v)) $(sel).value = v; };
    put('#th-ambient-h', a.h); put('#th-ambient-t', a.t); put('#th-duration', s.duration); put('#th-steps', s.steps); put('#th-initial', s.initial);
    if (Number.isFinite(s.resolution)) { $('#th-res').value = s.resolution; $('#th-res-out').textContent = s.resolution; }
    this.renderList();
  }

  /** The thermal result as { meta, arrays }, or null. */
  exportResult() {
    const r = this.result;
    if (!r) return null;
    const meta = {};
    for (const [k, v] of Object.entries(r)) if (!ArrayBuffer.isView(v) && k !== 'frames' && k !== 'stale') meta[k] = v;
    meta.frameTimes = r.frames.map((f) => ({ t: f.t, min: f.min, max: f.max }));
    const arrays = [{ name: 'thermal.T', data: r.T, enc: 'q16' }, { name: 'thermal.flux', data: r.flux, enc: 'q16' }];
    r.frames.forEach((f, i) => arrays.push({ name: `thermal.frame.${i}`, data: f.T, enc: 'q16' }));
    return { meta, arrays };
  }

  importResult(meta, arrays) {
    const nV = this.part.nVert;
    const get = (name) => {
      const a = arrays.get(name);
      if (!a || a.data.length !== nV) throw new Error('The thermal result in the file does not fit this part.');
      return a.data;
    };
    const { frameTimes = [], ...rest } = meta;
    // files from the native app carry the frame times but no separate history list
    if (!rest.history && frameTimes.length) rest.history = frameTimes.map((f) => ({ t: f.t, min: f.min, max: f.max }));
    const frames = frameTimes.map((f, i) => ({ i, t: f.t, min: f.min, max: f.max, T: get(`thermal.frame.${i}`) }));
    this.result = { ...rest, T: get('thermal.T'), flux: get('thermal.flux'), frames };
    this.display = 'results';
    this.view.frame = Math.max(0, frames.length - 1);
    $('#th-results').hidden = false;
    $('#th-time').hidden = !this.result.transient;
  }

  exportView() {
    return { plot: this.view.plot, frame: this.view.frame, bands: this.view.bands, bcs: this.view.bcs };
  }

  importView(v = {}) {
    if (['T', 'flux'].includes(v.plot)) this.view.plot = v.plot;
    if (Number.isInteger(v.frame) && this.result) this.view.frame = Math.max(0, Math.min(Math.max(0, this.result.frames.length - 1), v.frame));
    if (typeof v.bands === 'boolean') this.view.bands = v.bands;
    if (typeof v.bcs === 'boolean') this.view.bcs = v.bcs;
    $('#th-plot').value = this.view.plot;
    $('#th-bands').checked = this.view.bands;
    $('#th-bcs').checked = this.view.bcs;
  }

}

function res0(r) {
  return r.converged === false ? ' The solver did not fully converge.' : '';
}
