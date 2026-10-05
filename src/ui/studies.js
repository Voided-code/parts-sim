// Structural studies beyond linear static (the Structural tab's study picker): nonlinear static,
// frequency, buckling, fatigue, drop test, linear dynamic and optimization. Each study renders its
// options, runs its worker job and draws its results; StructuralPanel owns the shared setup
// (material, fixtures, loads, mesh) and the linear static study.
import { seriesSpecs } from '../core/psim-series.js';
import * as THREE from 'three';
import { FEAJob } from '../fea/structural.js';
import { MATERIALS, completeMaterial } from '../core/materials.js';
import { fatigueField, snCurve, strengthAt, cyclesToFailure, FINISHES, cycleStress, goodman } from '../fea/fatigue.js';
import { harmonicField, harmonicShape, excitation, timeHistory, shapeAt, stressAt } from '../fea/response.js';
import { surfaceNets, toSTL } from '../core/isosurface.js';
import { $, h, num, stress, stressFormatter, force } from './dom.js';
import { renderLegend } from './legend.js';
import { LineChart } from './chart.js';

export const STUDY_INFO = {
  static: { desc: 'Stress, deflection and safety factor under steady loads; the break test shows where cracks start and run.', fixtures: true, loads: true, button: 'Run bend test', budget: 8e4 },
  nonlinear: { desc: 'Large bending and metal yielding that a linear study misses. Can raise the load until the part collapses or tears, then shows the permanent bend.', fixtures: true, loads: true, button: 'Run nonlinear', budget: 1e4 },
  modal: { desc: 'Natural vibration frequencies and mode shapes. Uses the fixtures; loads are ignored. With no fixtures the part is free-floating.', fixtures: true, fixturesOptional: true, loads: false, button: 'Find frequencies', budget: 4e4 },
  buckling: { desc: 'How many times the loads can grow before a slender or compressed part suddenly buckles sideways.', fixtures: true, loads: true, button: 'Run buckling', budget: 4e4 },
  fatigue: { desc: 'How many load cycles the part survives when the loads are applied over and over (S-N curve).', fixtures: true, loads: true, button: 'Run fatigue', budget: 8e4 },
  drop: { desc: 'The part falls onto a rigid floor, as it is oriented now (gravity −Y). Rotate it in the Part tab to change which side hits. Fixtures and loads are not used.', fixtures: false, loads: false, button: 'Run drop test', budget: 4e4 },
  dynamic: { desc: 'Response to vibration, shock or earthquake shaking over time or frequency, from the natural modes (modal superposition).', fixtures: true, loads: true, loadsOptional: true, button: 'Run dynamic', budget: 3e4 },
  optimize: { desc: 'Remove material that does little (topology), or find the lightest / cheapest material and size that keeps the safety factor.', fixtures: true, loads: true, button: 'Optimize', budget: 2.4e4 },
};

const pct = (x) => (x < 5e-4 ? '0%' : `${num(x * 100, 2)}%`);

const kpi = (label, value, sub, status, wide = false) => h(wide ? 'div.kpi.wide' : 'div.kpi', {},
  h('div.k-label', {}, label), h('div.k-value', {}, value),
  sub || status ? h('div.k-sub', {}, status ? h('span', { className: `status ${status[0]}` }, status[1]) : null, status && sub ? ' ' : null, sub || null) : null);

function chartBlock(title) {
  const canvas = h('canvas', { height: 150 });
  canvas.style.cssText = 'width:100%;height:150px;display:block;cursor:crosshair';
  const tip = h('div.chart-tip', { hidden: true });
  return { el: h('div.chart-wrap', {}, h('div.chart-title', {}, title), canvas, tip), canvas, tip };
}

function field(label, input) {
  const wrap = h('div.field', {}, h('label', {}, label), input);
  return wrap;
}

function numberInput(value, onchange, { min, max, step = 'any', width = null } = {}) {
  const el = h('input', { type: 'number', value, min, max, step, onchange: (e) => onchange(Number(e.target.value)) });
  el.style.cssText = `height:28px;padding:0 8px;border:1px solid var(--border-strong);border-radius:7px;background:var(--panel);${width ? `width:${width}` : 'width:100%'}`;
  return el;
}

function seg(options, value, onpick) {
  const el = h('div.seg.wide');
  for (const [v, label, title] of options) {
    el.append(h('button', {
      className: v === value ? 'active' : '', title: title || '',
      onclick: (e) => {
        for (const b of el.children) b.classList.toggle('active', b === e.currentTarget);
        onpick(v);
      },
    }, label));
  }
  return el;
}

function select(options, value, onchange) {
  const el = h('select', { onchange: (e) => onchange(e.target.value) }, ...options.map(([v, label]) => h('option', { value: v, selected: v === value }, label)));
  return el;
}

function maxAbs3(u) {
  let m = 0;
  for (let i = 0; i < u.length; i += 3) if (!Number.isNaN(u[i])) m = Math.max(m, Math.hypot(u[i], u[i + 1], u[i + 2]));
  return m;
}

function magnitudes(u) {
  const out = new Float32Array(u.length / 3);
  for (let v = 0; v < out.length; v++) out[v] = Number.isNaN(u[3 * v]) ? NaN : Math.hypot(u[3 * v], u[3 * v + 1], u[3 * v + 2]);
  return out;
}

function range(values) {
  let lo = Infinity, hi = -Infinity, arg = -1;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isNaN(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) { hi = v; arg = i; }
  }
  return { lo, hi, arg };
}

// ---- saving and loading results (.psim) ----

const plain = (x) => (x === undefined ? undefined : structuredClone(x));
const isObj = (x) => !!x && typeof x === 'object' && !Array.isArray(x);

/** Copies the keys of `target` that `src` holds with the same type (and pass `rules[key]`, a list or a test). */
function adopt(target, src, keys, rules = {}) {
  if (!isObj(src)) return;
  for (const k of keys) {
    const v = src[k];
    if (typeof v !== typeof target[k]) continue;
    if (typeof v === 'number' && !Number.isFinite(v)) continue;
    const rule = rules[k];
    if (Array.isArray(rule) ? !rule.includes(v) : rule && !rule(v)) continue;
    target[k] = v;
  }
}

/** One stored array of a result, checked against the length this part needs. */
function arr(arrays, name, n) {
  const a = arrays?.get(name);
  if (!a || !a.data) throw new Error(`The file's study result has no "${name}".`);
  if (a.data.length !== n) throw new Error(`The file's study result does not fit this part: "${name}" has ${a.data.length} values, expected ${n}.`);
  return a.data;
}

function checkMeta(meta, nVert, id) {
  if (!isObj(meta)) throw new Error(`The file's ${id} result is damaged (no header).`);
  if (meta.nVert !== nVert) throw new Error(`The file's ${id} result is for a part with ${meta.nVert} vertices, this part has ${nVert}.`);
}

const finiteList = (l) => Array.isArray(l) && l.every((x) => typeof x === 'number');

const freqText = (f) => (f >= 1000 ? `${num(f / 1000)} kHz` : `${num(f)} Hz`);
const timeText = (t) => (t >= 1 ? `${num(t)} s` : t >= 1e-3 ? `${num(t * 1e3)} ms` : `${num(t * 1e6)} µs`);
const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
const sup = (n) => String(n).split('').map((c) => (c === '-' ? '⁻' : SUP[Number(c)] ?? c)).join('');
const pow10 = (e) => `10${sup(e)}`;
const cyclesText = (n) => {
  if (!Number.isFinite(n)) return 'unlimited';
  if (n >= 1e9) return `> ${pow10(9)}`;
  if (n < 1) return '< 1';
  if (n < 1e4) return String(Math.round(n));
  const e = Math.floor(Math.log10(n)), m = n / 10 ** e;
  return m < 1.05 ? pow10(e) : `${num(m, 2)}×${pow10(e)}`;
};

/** Base class: shared plumbing for running a job and drawing a field. */
class Study {
  constructor(panel) {
    this.panel = panel;
    this.app = panel.app;
    this.result = null;
    this.stale = false;
  }

  get part() { return this.panel.part; }
  get info() { return STUDY_INFO[this.id]; }

  options() { return null; }
  clear() { this.result = null; }

  // ---- .psim: view keys that are plain data (not the animation state) and checks for them
  viewKeys = [];
  optRules = {};
  viewRules = {};

  exportOptions() {
    const view = {};
    for (const k of this.viewKeys) view[k] = this.view[k];
    return { opts: plain(this.opts) ?? {}, view };
  }

  importOptions(o) {
    if (!isObj(o)) return;
    adopt(this.opts, o.opts, Object.keys(this.opts ?? {}), this.optRules);
    adopt(this.view, o.view, this.viewKeys, this.viewRules);
    this.fitView();
  }

  /** Keeps the view settings inside the result after they were loaded. */
  fitView() {}

  /** The result as { meta, arrays } for a .psim file, or null when there is none to store. */
  exportResult() { return null; }

  /** Sets the result from decoded data without solving or drawing; the caller then calls show(). */
  importResult() {}

  /** One short line about the result for the file info panel. */
  fileNote() { return ''; }

  async prepareRun(opts = {}) {
    const p = this.panel;
    if (!p.checkSetup(this.id, opts)) return null;
    p.cancelJob();
    p.stopBreakPlay();
    const version = p.runVersion;
    try {
      const prep = await p.prepare(version, opts);
      if (version !== p.runVersion) return null;
      return { ...prep, version };
    } catch (err) {
      if (version !== p.runVersion) return null;
      p.preparing = false;
      this.app.busy.hide();
      this.app.status(err.message, 'error');
      return null;
    }
  }

  /** Run a worker job; returns the final message, or null if cancelled or failed (status shown). */
  async job(msg, busyText, onStep = null) {
    const p = this.panel, version = p.runVersion;
    try {
      p.job = new FEAJob(msg, {
        onProgress: (d) => {
          if (version !== p.runVersion) return;
          const frac = d.frac ?? Math.min(1, Math.log10(Math.max(d.res, 1e-7)) / -6);
          this.app.busy.progress(frac, d.frac != null ? d.stage : `${d.stage}${d.it ? ` · iteration ${d.it}` : ''}`);
        },
        onStep: (d) => { if (version === p.runVersion) onStep?.(d); },
      });
      this.app.busy.show(busyText, () => p.cancelJob());
      const res = await p.job.promise;
      return version === p.runVersion ? res : null;
    } catch (err) {
      if (version !== p.runVersion) return null;
      this.app.status(err.cancelled ? 'Cancelled.' : `Solver error: ${err.message}`, err.cancelled ? '' : 'error');
      return null;
    } finally {
      if (version === p.runVersion) { p.job = null; this.app.busy.hide(); }
    }
  }

  engineNote(res) {
    return `on the ${res.engine || 'CPU'}${res.gpuNote ? ` (GPU not used: ${res.gpuNote})` : ''}`;
  }

  /** Colour the part with per-vertex values and draw the legend. */
  paint(values, { min, max, title, sub, format, reverse = false, markers = [], cmap = null }) {
    const v = this.panel.view;
    const colormap = cmap || (v.heat ? 'heat' : 'rainbow');
    const bands = v.bands ? 12 : 0;
    this.app.viewer.setScalars(values, { min, max, bands, reverse, colormap });
    renderLegend($('#legend'), { title, sub, min, max, format, bands, colormap, reverse, markers });
  }

  card(title) {
    $('#study-card').hidden = false;
    $('#study-title').textContent = title;
    $('#study-alert').hidden = true;
    return { kpis: $('#study-kpis'), body: $('#study-body') };
  }

  alert(text, error = true) {
    const el = $('#study-alert');
    el.hidden = !text;
    if (!text) return;
    el.replaceChildren(h('div', {}, text));
    el.style.borderColor = error ? '' : 'var(--border)';
    el.style.background = error ? '' : 'var(--panel)';
  }

  displayToggles() {
    const v = this.panel.view;
    const chk = (label, key) => h('label.check', {}, h('input', { type: 'checkbox', checked: v[key], onchange: (e) => { v[key] = e.target.checked; this.show(); } }), ` ${label}`);
    return h('div.row.gap.wrap', {}, chk('Contour bands', 'bands'), chk('Heat colours', 'heat'), chk('Show loads', 'bcs'));
  }

  marker(vertex, u, scale, label) {
    const V = this.part.vertices, viewer = this.app.viewer;
    if (vertex < 0 || !this.panel.view.marker) { viewer.clearMarker(); return; }
    const p = new THREE.Vector3(V[3 * vertex], V[3 * vertex + 1], V[3 * vertex + 2]);
    if (u) p.add(new THREE.Vector3(u[3 * vertex], u[3 * vertex + 1], u[3 * vertex + 2]).multiplyScalar(scale));
    if (viewer.marker && viewer.marker.label === label) viewer.moveMarker(p);
    else viewer.setMarker(p, label);
  }

  probeValue(hit, values, fmt) {
    if (!values) return null;
    const T = this.part.tris, t = hit.tri;
    const w = [hit.bary.x, hit.bary.y, hit.bary.z];
    let s = 0;
    for (let k = 0; k < 3; k++) s += w[k] * values[T[3 * t + k]];
    return Number.isNaN(s) ? 'no data' : fmt(s);
  }

  probe(hit) {
    return this.shown ? this.probeValue(hit, this.shown.values, this.shown.fmt) : null;
  }

  tick() {}
}

// ---------------------------------------------------------------------------------------------
// Nonlinear static

class NonlinearStudy extends Study {
  id = 'nonlinear';
  opts = { large: true, plastic: true, mode: 'applied', steps: 10 };
  view = { plot: 'vm', step: -1, exaggerate: 1, playing: false, t: 0, unloaded: false };

  options() {
    const o = this.opts;
    const mat = this.panel.material;
    return [
      h('h3', {}, 'Nonlinear options'),
      h('label.check', {}, h('input', { type: 'checkbox', checked: o.large, onchange: (e) => { o.large = e.target.checked; this.panel.markStale(); } }), ' Large deflection (geometry updates as it bends)'),
      h('label.check', {}, h('input', { type: 'checkbox', checked: o.plastic, disabled: !!mat.brittle, onchange: (e) => { o.plastic = e.target.checked; this.panel.markStale(); } }),
        mat.brittle ? ' Plasticity (off: the material is brittle)' : ` Plasticity (yields at ${num(mat.yield)} MPa, tears at ${num(mat.elongation * 100)}% strain)`),
      field('Load', seg([['applied', 'Applied loads'], ['failure', 'Until it fails', 'Keep raising the loads until the part collapses, tears or cracks']], o.mode, (v) => { o.mode = v; this.panel.markStale(); })),
      field('Load steps', numberInput(o.steps, (v) => { o.steps = Math.max(2, Math.min(60, Math.round(v) || 10)); this.panel.markStale(); }, { min: 2, max: 60, step: 1 })),
      h('p.muted.small', { style: { margin: '6px 0 0' } }, 'Steps adapt automatically near yielding and collapse. Nonlinear runs take longer than linear ones: start with a coarse mesh.'),
    ];
  }

  async run() {
    const prep = await this.prepareRun();
    if (!prep) return;
    const mat = completeMaterial(prep.material);
    const r = this.result = { steps: [], unloaded: null, material: mat, totalF: Math.hypot(...prep.asm.total), units: prep.units, done: false, opts: { ...this.opts } };
    this.view.step = -1;
    this.view.unloaded = false;
    this.panel.display = 'study';
    const res = await this.job({
      type: 'nonlinear', ...prep.msg, material: mat, largeDisplacement: this.opts.large, plasticity: this.opts.plastic,
      untilFailure: this.opts.mode === 'failure', steps: this.opts.steps,
    }, 'Nonlinear static: ramping the load…', (d) => {
      if (d.kind === 'unloaded') { r.unloaded = d; return; }
      r.steps.push(d);
      this.view.step = r.steps.length - 1;
      this.show();
    });
    if (!res) { r.done = true; if (r.steps.length) this.show(); return; }
    Object.assign(r, { done: true, reason: res.reason, engine: res.engine, gpuNote: res.gpuNote, plastic: res.plastic });
    this.stale = false;
    this.view.step = r.steps.length - 1;
    this.show();
    const last = r.steps[r.steps.length - 1];
    const load = last && r.totalF > 0 ? force(last.lam * r.totalF) : '–';
    const why = {
      reached: 'Reached the applied loads',
      collapse: `Collapsed at about ${load}: it keeps bending without taking more load`,
      rupture: `Tore: plastic strain reached the elongation at break at ${load}`,
      crack: `Cracked: tensile stress reached the strength at ${load}`,
      'large deformation': `Failed by gross bending at ${load} (it moved more than 15% of its size)`,
      'time limit': `Stopped after 4 minutes at ${load} without failing; try a coarser mesh`,
      'step limit': `Stopped after the maximum number of load steps at ${load}`,
    }[res.reason] || res.reason;
    this.app.status(`${why} (${r.steps.length} steps ${this.engineNote(res)}).`, res.reason === 'reached' ? '' : 'warn');
  }

  viewKeys = ['plot', 'step', 'exaggerate', 'unloaded'];
  optRules = { mode: ['applied', 'failure'], steps: (v) => v >= 2 && v <= 60 };
  viewRules = { plot: ['vm', 'disp', 'pe'], exaggerate: (v) => v >= 1 && v <= 50 };

  fitView() {
    const r = this.result, v = this.view;
    v.playing = false;
    if (!r) return;
    v.step = Math.max(-1, Math.min(r.steps.length - 1, Math.round(v.step)));
    if (v.unloaded && !r.unloaded) v.unloaded = false;
  }

  exportResult() {
    const r = this.result;
    if (!r?.done || !r.reason || !r.steps.length) return null;
    const nV = this.part.nVert, arrays = [];
    const one = (s) => ({ lam: s.lam, D: s.D, maxVM: s.maxVM, maxPE: s.maxPE, maxDisp: s.maxDisp, iterations: s.iterations });
    const steps = r.steps.map(one);
    const unloaded = r.unloaded ? one(r.unloaded) : null;
    // each field of the steps is one series (shared range, predicted from the step before); the spring-back state stands alone
    for (const [field, stride] of [['u', 3], ['vm', 1], ['pe', 1]]) {
      arrays.push(...seriesSpecs(r.steps.map((s) => s[field]), (i) => `nonlinear.step.${i}.${field}`, { stride }));
    }
    if (r.unloaded) for (const [field, stride] of [['u', 3], ['vm', 1], ['pe', 1]]) arrays.push({ name: `nonlinear.unloaded.${field}`, data: r.unloaded[field], enc: 'q16', ...(stride > 1 ? { stride } : {}) });
    const meta = {
      nVert: nV, units: r.units, material: plain(r.material), totalF: r.totalF, reason: r.reason, engine: r.engine ?? null, gpuNote: r.gpuNote ?? null,
      plastic: !!r.plastic, opts: plain(r.opts), steps, unloaded,
    };
    return { meta, arrays };
  }

  importResult(meta, arrays) {
    const nV = this.part.nVert;
    checkMeta(meta, nV, 'nonlinear');
    if (!Array.isArray(meta.steps) || !meta.steps.length || !isObj(meta.material)) throw new Error('The file\'s nonlinear result is damaged (no steps).');
    const read = (m, id, kind) => ({ ...m, kind, u: arr(arrays, `nonlinear.${id}.u`, 3 * nV), vm: arr(arrays, `nonlinear.${id}.vm`, nV), pe: arr(arrays, `nonlinear.${id}.pe`, nV) });
    const steps = meta.steps.map((m, i) => read(m, `step.${i}`, 'step'));
    const unloaded = isObj(meta.unloaded) ? read(meta.unloaded, 'unloaded', 'unloaded') : null;
    this.result = {
      steps, unloaded, material: meta.material, totalF: meta.totalF, units: meta.units, done: true, opts: { ...this.opts, ...(isObj(meta.opts) ? meta.opts : {}) },
      reason: meta.reason, engine: meta.engine ?? undefined, gpuNote: meta.gpuNote ?? undefined, plastic: !!meta.plastic,
    };
    this.stale = false;
    this.view.step = steps.length - 1;
    this.view.unloaded = false;
    this.view.playing = false;
  }

  fileNote(r = this.result) {
    const last = r?.steps?.[r.steps.length - 1];
    return last ? `${r.steps.length} load steps to ×${num(last.lam)}, ${r.reason}` : '';
  }

  current() {
    const r = this.result;
    if (!r || !r.steps.length) return null;
    if (this.view.unloaded && r.unloaded) return r.unloaded;
    return r.steps[Math.max(0, Math.min(r.steps.length - 1, this.view.step))];
  }

  show() {
    const r = this.result;
    if (!r || this.app.tab !== 'structural' || this.panel.study !== this.id) return;
    const s = this.current();
    if (!s) return;
    const { kpis, body } = this.card('Nonlinear static results');
    const mat = r.material;
    const loadAt = (lam) => (r.totalF > 1e-9 ? force(lam * r.totalF) : `${num(lam)} × loads`);
    const last = r.steps[r.steps.length - 1];
    let maxPE = 0;
    for (const st of r.steps) maxPE = Math.max(maxPE, st.maxPE);
    const firstYield = r.steps.find((st) => st.maxPE > 0);
    const status = !r.done ? ['warn', 'running'] : r.reason === 'reached' ? (maxPE > 0 ? ['warn', 'yielded'] : ['good', 'elastic']) : ['bad', r.reason];
    kpis.replaceChildren(
      kpi(r.opts.mode === 'failure' ? 'Failure load' : 'Load reached', loadAt(last.lam), null, status),
      kpi('Max von Mises', stress(s.maxVM), `yield ${num(mat.yield)} MPa`),
      kpi('Max displacement', `${num(s.maxDisp)} ${r.units}`, s.kind === 'unloaded' ? 'after unloading' : `at ${loadAt(s.lam)}`),
      kpi('Max plastic strain', `${num(maxPE * 100)} %`, firstYield ? `yielding from ${loadAt(firstYield.lam)}` : 'no yielding'),
    );
    if (r.unloaded) kpis.append(kpi('Permanent bend', `${num(r.unloaded.maxDisp)} ${r.units}`, 'after the load is removed'));
    // load-displacement curve
    const ch = chartBlock(`Load vs displacement (along the loads, ${r.units})`);
    const pts = [{ x: 0, y: 0 }, ...r.steps.map((st) => ({ x: st.D, y: st.lam * (r.totalF > 1e-9 ? r.totalF : 1) }))];
    const chart = new LineChart(ch.canvas, ch.tip, {
      xLabel: `displacement (${r.units})`, integerX: false, formatX: (x) => num(x), formatY: (y) => (r.totalF > 1e-9 ? force(y) : num(y)),
      tipText: (p, i) => `${i ? `step ${i}` : 'start'} · ${r.totalF > 1e-9 ? force(p.y) : num(p.y)} · ${num(p.x)} ${r.units}`,
      onPick: (i) => { this.view.step = Math.max(0, i - 1); this.view.unloaded = false; this.show(); },
    });
    const slider = h('input', { type: 'range', min: 0, max: r.steps.length - 1, value: this.view.step, oninput: (e) => { this.view.step = Number(e.target.value); this.view.unloaded = false; this.stopPlay(); this.show(); } });
    const playBtn = h('button.btn.small', { onclick: () => this.togglePlay() }, this.view.playing ? '❚❚ Pause' : '▶ Play');
    const plots = [['vm', 'von Mises stress'], ['disp', 'Displacement'], ['pe', 'Plastic (permanent) strain']];
    body.replaceChildren(
      ch.el,
      field(`Step (${this.view.unloaded ? 'unloaded' : `${this.view.step + 1}/${r.steps.length}`})`, slider),
      h('div.row.gap.wrap', {}, playBtn,
        r.unloaded ? h('label.check', {}, h('input', { type: 'checkbox', checked: this.view.unloaded, onchange: (e) => { this.view.unloaded = e.target.checked; this.stopPlay(); this.show(); } }), ' After unloading (spring-back)') : null),
      field('Plot', select(plots, this.view.plot, (v) => { this.view.plot = v; this.show(); })),
      field('Exaggerate shape', h('input', { type: 'range', min: 0, max: 100, value: Math.round(Math.log(this.view.exaggerate) / Math.log(50) * 100), oninput: (e) => { this.view.exaggerate = Math.pow(50, Number(e.target.value) / 100); this.draw(); } })),
      this.displayToggles(),
    );
    chart.set(pts, this.view.unloaded ? -1 : this.view.step + 1);
    this.draw();
  }

  draw() {
    const r = this.result, s = this.current();
    if (!r || !s) return;
    const mat = r.material;
    let values, spec;
    if (this.view.plot === 'disp') {
      values = magnitudes(s.u);
      const mx = Math.max(...r.steps.map((st) => st.maxDisp), 1e-12);
      spec = { min: 0, max: mx, title: 'Displacement', format: (x) => `${num(x)} ${r.units}` };
    } else if (this.view.plot === 'pe') {
      values = Float32Array.from(s.pe, (x) => x * 100);
      const mx = Math.max(mat.elongation * 100, ...r.steps.map((st) => st.maxPE * 100), 1e-6);
      spec = { min: 0, max: mx, title: 'Plastic strain', format: (x) => `${num(x)} %`, markers: [{ value: mat.elongation * 100, label: `breaks ${num(mat.elongation * 100)}%` }] };
    } else {
      values = s.vm;
      const hi = Math.max(mat.uts * 1e6, ...r.steps.map((st) => st.maxVM));
      spec = { min: 0, max: hi, title: 'von Mises stress', format: stressFormatter(hi), markers: [{ value: mat.yield * 1e6, label: `Yield ${num(mat.yield)}` }, { value: mat.uts * 1e6, label: `UTS ${num(mat.uts)}` }] };
    }
    const sub = `${mat.name} · ${s.kind === 'unloaded' ? 'after unloading' : `load ×${num(s.lam)}`} · ${this.view.exaggerate < 1.05 ? 'true-scale shape' : `shape ×${num(this.view.exaggerate)}`}`;
    this.paint(values, { ...spec, sub });
    this.app.viewer.setDeformation(s.u, this.view.exaggerate);
    this.shown = { values, fmt: spec.format };
    const { arg } = range(values);
    this.marker(arg, s.u, this.view.exaggerate, this.view.plot === 'pe' ? 'Most permanent strain' : 'Highest stress');
    this.panel.drawOverlays();
  }

  togglePlay() {
    if (this.view.playing) return this.stopPlay();
    const r = this.result;
    if (!r?.steps.length) return;
    this.view.unloaded = false;
    if (this.view.step >= r.steps.length - 1) this.view.step = 0;
    this.view.playing = true;
    this.view.t = 0;
    this.show();
  }

  stopPlay() { this.view.playing = false; }

  tick(dt) {
    if (!this.view.playing || !this.result) return;
    this.view.t += dt;
    if (this.view.t < 0.3) return;
    this.view.t = 0;
    if (this.view.step >= this.result.steps.length - 1) { this.view.playing = false; this.show(); return; }
    this.view.step++;
    this.show();
  }
}

// ---------------------------------------------------------------------------------------------
// Frequency (modal)

class ModalStudy extends Study {
  id = 'modal';
  opts = { nev: 5 };
  view = { mode: 0, amp: 0.08, phase: 0, animate: true };

  options() {
    return [
      h('h3', {}, 'Frequency options'),
      field('Number of modes', numberInput(this.opts.nev, (v) => { this.opts.nev = Math.max(1, Math.min(20, Math.round(v) || 5)); this.panel.markStale(); }, { min: 1, max: 20, step: 1 })),
    ];
  }

  async run() {
    const prep = await this.prepareRun({ requireLoads: false, requireFixtures: false });
    if (!prep) return;
    const res = await this.job({ type: 'modal', ...prep.msg, nev: this.opts.nev }, 'Finding natural frequencies…');
    if (!res) return;
    this.result = { ...res, units: prep.units, material: prep.material, diag: this.part.bbox.diag };
    this.stale = false;
    this.view.mode = 0;
    this.panel.display = 'study';
    this.show();
    const f = res.modes.map((m) => freqText(m.freq));
    this.app.status(`${res.modes.length} natural frequencies ${this.engineNote(res)}${res.free ? ' (free-floating: rigid-body modes skipped)' : ''}: ${f.join(', ')}.${res.converged ? '' : ' Some modes did not fully converge.'}`, res.converged ? '' : 'warn');
  }

  viewKeys = ['mode', 'amp', 'animate'];
  optRules = { nev: (v) => v >= 1 && v <= 20 };
  viewRules = { amp: (v) => v >= 0.01 && v <= 0.3 };

  fitView() {
    const n = this.result?.modes.length;
    if (n) this.view.mode = Math.max(0, Math.min(n - 1, Math.round(this.view.mode)));
  }

  exportResult() {
    const r = this.result;
    if (!r?.modes?.length) return null;
    const meta = {
      nVert: this.part.nVert, units: r.units, material: plain(r.material), diag: r.diag, free: !!r.free, converged: !!r.converged, iterations: r.iterations ?? null,
      totalMass: r.totalMass ?? null, engine: r.engine ?? null, gpuNote: r.gpuNote ?? null, modes: r.modes.map((m) => ({ freq: m.freq, eff: Array.from(m.eff) })),
    };
    return { meta, arrays: r.modes.map((m, i) => ({ name: `modal.shape.${i}`, data: m.shape, enc: 'q16', stride: 3 })) };
  }

  importResult(meta, arrays) {
    const nV = this.part.nVert;
    checkMeta(meta, nV, 'frequency');
    if (!Array.isArray(meta.modes) || !meta.modes.length || !isObj(meta.material) || !(meta.diag > 0)) throw new Error('The file\'s frequency result is damaged (no modes).');
    const modes = meta.modes.map((m, i) => {
      if (!finiteList(m.eff) || m.eff.length !== 3 || typeof m.freq !== 'number') throw new Error('The file\'s frequency result is damaged (bad mode).');
      return { freq: m.freq, eff: m.eff, shape: arr(arrays, `modal.shape.${i}`, 3 * nV) };
    });
    this.result = {
      modes, free: !!meta.free, converged: !!meta.converged, iterations: meta.iterations ?? undefined, totalMass: meta.totalMass ?? undefined,
      engine: meta.engine ?? undefined, gpuNote: meta.gpuNote ?? undefined, units: meta.units, material: meta.material, diag: meta.diag,
    };
    this.stale = false;
    this.view.mode = 0;
  }

  fileNote(r = this.result) {
    const m = r?.modes;
    return m?.length ? `${m.length} modes, ${freqText(m[0].freq)} to ${freqText(m[m.length - 1].freq)}` : '';
  }

  show() {
    const r = this.result;
    if (!r || this.app.tab !== 'structural' || this.panel.study !== this.id) return;
    const { kpis, body } = this.card('Natural frequencies');
    const sum = [0, 1, 2].map((d) => r.modes.reduce((s, m) => s + m.eff[d], 0));
    kpis.replaceChildren(
      kpi('Lowest frequency', r.modes.length ? freqText(r.modes[0].freq) : '–', r.free ? 'free-floating part' : 'with the fixtures'),
      kpi('Modes found', String(r.modes.length), `highest ${freqText(r.modes[r.modes.length - 1]?.freq ?? 0)}`),
      kpi('Mass captured by these modes', `X ${pct(sum[0])} · Y ${pct(sum[1])} · Z ${pct(sum[2])}`, 'shaking along an axis with little captured mass excites higher modes', null, true),
    );
    const list = h('ul.items');
    r.modes.forEach((m, i) => {
      const dom = ['X', 'Y', 'Z'][m.eff.indexOf(Math.max(...m.eff))];
      list.append(h('li', { className: i === this.view.mode ? 'selected' : '', onclick: () => { this.view.mode = i; this.show(); } },
        h('span.name', {}, `Mode ${i + 1}`),
        h('span.meta', {}, `${freqText(m.freq)} · ${Math.max(...m.eff) > 0.01 ? `${pct(Math.max(...m.eff))} mass ${dom}` : 'little mass'}`)));
    });
    body.replaceChildren(
      list,
      h('div.row.gap.wrap', { style: { marginTop: '8px' } },
        h('label.check', {}, h('input', { type: 'checkbox', checked: this.view.animate, onchange: (e) => { this.view.animate = e.target.checked; this.draw(); } }), ' Animate')),
      field('Amplitude (display only)', h('input', { type: 'range', min: 1, max: 30, value: Math.round(this.view.amp * 100), oninput: (e) => { this.view.amp = Number(e.target.value) / 100; this.draw(); } })),
      h('p.muted.small', {}, 'Mode shapes show the pattern of vibration; their size is arbitrary. Colours show where the part moves most in that mode. Effective mass tells how strongly shaking along X, Y or Z excites it.'),
      this.displayToggles(),
    );
    this.draw();
  }

  draw() {
    const r = this.result, m = r?.modes[this.view.mode];
    if (!m) return;
    const values = magnitudes(m.shape);
    this.paint(values, { min: 0, max: 1, title: 'Relative displacement', sub: `Mode ${this.view.mode + 1} · ${freqText(m.freq)}`, format: (x) => num(x, 2) });
    this.shown = { values, fmt: (x) => `${num(x * 100)} % of max` };
    const s = this.view.animate ? Math.sin(this.view.phase) : 1;
    this.app.viewer.setDeformation(m.shape, s * this.view.amp * r.diag);
    this.app.viewer.clearMarker();
    this.panel.drawOverlays();
  }

  tick(dt) {
    if (!this.result || !this.view.animate || this.panel.display !== 'study') return;
    this.view.phase += dt * Math.PI * 1.4;
    const m = this.result.modes[this.view.mode];
    if (m) this.app.viewer.setDeformation(m.shape, Math.sin(this.view.phase) * this.view.amp * this.result.diag);
  }
}

// ---------------------------------------------------------------------------------------------
// Buckling

class BucklingStudy extends Study {
  id = 'buckling';
  opts = { nev: 3 };
  view = { mode: 0, amp: 0.08, phase: 0, animate: false };

  options() {
    return [
      h('h3', {}, 'Buckling options'),
      field('Number of modes', numberInput(this.opts.nev, (v) => { this.opts.nev = Math.max(1, Math.min(8, Math.round(v) || 3)); this.panel.markStale(); }, { min: 1, max: 8, step: 1 })),
    ];
  }

  async run() {
    const prep = await this.prepareRun();
    if (!prep) return;
    const mat = completeMaterial(prep.material);
    const strength = (mat.brittle ? mat.uts : mat.yield) * 1e6;
    const res = await this.job({ type: 'buckling', ...prep.msg, nev: this.opts.nev, strength }, 'Buckling analysis…');
    if (!res) return;
    let maxVM = 0;
    for (const v of res.vm) if (!Number.isNaN(v)) maxVM = Math.max(maxVM, v);
    this.result = { ...res, units: prep.units, material: prep.material, totalF: Math.hypot(...prep.asm.total), diag: this.part.bbox.diag, maxVM };
    this.stale = false;
    this.view.mode = 0;
    this.panel.display = 'study';
    this.show();
    const bl = res.modes[0]?.factor;
    this.app.status(Number.isFinite(bl)
      ? `Lowest buckling load factor ${num(bl)} ${this.engineNote(res)}: it buckles at about ${force(bl * this.result.totalF)}.`
      : `No buckling below ${this.beyond()} ${this.engineNote(res)}.`, Number.isFinite(bl) && bl < 1 ? 'error' : '');
  }

  viewKeys = ['mode', 'amp', 'animate'];
  optRules = { nev: (v) => v >= 1 && v <= 8 };
  viewRules = { amp: (v) => v >= 0.01 && v <= 0.3 };

  fitView() {
    const n = this.result?.modes.length;
    if (n) this.view.mode = Math.max(0, Math.min(n - 1, Math.round(this.view.mode)));
  }

  exportResult() {
    const r = this.result;
    if (!r?.modes?.length) return null;
    const meta = {
      nVert: this.part.nVert, units: r.units, material: plain(r.material), diag: r.diag, totalF: r.totalF, maxVM: r.maxVM, maxFactor: r.maxFactor ?? null,
      converged: !!r.converged, iterations: r.iterations ?? null, engine: r.engine ?? null, gpuNote: r.gpuNote ?? null, factors: r.modes.map((m) => m.factor),
    };
    return { meta, arrays: r.modes.map((m, i) => ({ name: `buckling.shape.${i}`, data: m.shape, enc: 'q16', stride: 3 })) };
  }

  importResult(meta, arrays) {
    const nV = this.part.nVert;
    checkMeta(meta, nV, 'buckling');
    if (!finiteList(meta.factors) || !meta.factors.length || !isObj(meta.material) || !(meta.diag > 0)) throw new Error('The file\'s buckling result is damaged (no modes).');
    this.result = {
      modes: meta.factors.map((factor, i) => ({ factor, shape: arr(arrays, `buckling.shape.${i}`, 3 * nV) })),
      maxFactor: meta.maxFactor ?? undefined, converged: !!meta.converged, iterations: meta.iterations ?? undefined, engine: meta.engine ?? undefined, gpuNote: meta.gpuNote ?? undefined,
      units: meta.units, material: meta.material, totalF: meta.totalF, diag: meta.diag, maxVM: meta.maxVM,
    };
    this.stale = false;
    this.view.mode = 0;
  }

  fileNote(r = this.result) {
    const f = r?.modes?.[0]?.factor;
    return r?.modes?.length ? (Number.isFinite(f) ? `${r.modes.length} modes, lowest load factor ${num(f)}` : 'no buckling') : '';
  }

  // "no buckling" means none below the searched range (it yields long before)
  beyond() {
    const max = this.result?.maxFactor;
    return Number.isFinite(max) ? `${num(max)} × the loads (it yields long before that)` : 'any load (nothing is compressed)';
  }

  factorText(f) {
    const max = this.result?.maxFactor;
    return Number.isFinite(f) ? num(f) : Number.isFinite(max) ? `> ${num(max)}` : '∞';
  }

  show() {
    const r = this.result;
    if (!r || this.app.tab !== 'structural' || this.panel.study !== this.id) return;
    const { kpis, body } = this.card('Buckling');
    const bl = r.modes[0]?.factor;
    const mat = r.material;
    const yieldLam = r.maxVM > 0 ? (mat.brittle ? mat.uts : mat.yield) * 1e6 / r.maxVM : Infinity;
    const status = !Number.isFinite(bl) ? ['good', 'no buckling'] : bl < 1 ? ['bad', 'buckles under the applied loads'] : bl < 3 ? ['warn', 'low margin'] : ['good', 'safe margin'];
    kpis.replaceChildren(
      kpi('Buckling load factor', this.factorText(bl), null, status),
      kpi('Buckling load', Number.isFinite(bl) && r.totalF > 0 ? force(bl * r.totalF) : '–', 'loads × factor'),
      kpi('Yields first at', Number.isFinite(yieldLam) ? `× ${num(yieldLam)}` : '–', yieldLam < bl ? 'before it buckles' : 'after buckling'),
    );
    const list = h('ul.items');
    r.modes.forEach((m, i) => list.append(h('li', { className: i === this.view.mode ? 'selected' : '', onclick: () => { this.view.mode = i; this.show(); } },
      h('span.name', {}, `Mode ${i + 1}`), h('span.meta', {}, Number.isFinite(m.factor) || Number.isFinite(r.maxFactor) ? `factor ${this.factorText(m.factor)}` : 'no buckling'))));
    const notes = [];
    if (Number.isFinite(bl) && yieldLam < bl) notes.push(`The material yields at ${num(yieldLam)} × the loads, before the buckling load: expect it to fail by yielding (see the nonlinear study).`);
    notes.push('Linear buckling assumes a perfect part. Real parts with small imperfections often buckle at 50–80% of this load, so aim for a factor of 3 or more.');
    body.replaceChildren(
      list,
      h('div.row.gap.wrap', { style: { marginTop: '8px' } },
        h('label.check', {}, h('input', { type: 'checkbox', checked: this.view.animate, onchange: (e) => { this.view.animate = e.target.checked; this.draw(); } }), ' Animate')),
      field('Amplitude (display only)', h('input', { type: 'range', min: 1, max: 30, value: Math.round(this.view.amp * 100), oninput: (e) => { this.view.amp = Number(e.target.value) / 100; this.draw(); } })),
      ...notes.map((t) => h('p.muted.small', {}, t)),
      this.displayToggles(),
    );
    this.draw();
  }

  draw() {
    const r = this.result, m = r?.modes[this.view.mode];
    if (!m) return;
    const values = magnitudes(m.shape);
    this.paint(values, { min: 0, max: 1, title: 'Relative displacement', sub: `Buckling mode ${this.view.mode + 1} · factor ${this.factorText(m.factor)}`, format: (x) => num(x, 2) });
    this.shown = { values, fmt: (x) => `${num(x * 100)} % of max` };
    const s = this.view.animate ? Math.sin(this.view.phase) : 1;
    this.app.viewer.setDeformation(Number.isFinite(m.factor) ? m.shape : null, s * this.view.amp * r.diag);
    this.app.viewer.clearMarker();
    this.panel.drawOverlays();
  }

  tick(dt) {
    if (!this.result || !this.view.animate || this.panel.display !== 'study') return;
    this.view.phase += dt * Math.PI;
    const m = this.result.modes[this.view.mode];
    if (m && Number.isFinite(m.factor)) this.app.viewer.setDeformation(m.shape, Math.sin(this.view.phase) * this.view.amp * this.result.diag);
  }
}

// ---------------------------------------------------------------------------------------------
// Fatigue

class FatigueStudy extends Study {
  id = 'fatigue';
  opts = { loading: 'zero', R: 0, cycles: 1e6, finish: 'machined', scale: 1 };
  view = { plot: 'life' };

  options() {
    const o = this.opts;
    const mat = completeMaterial(this.panel.material);
    const Rinput = numberInput(o.R, (v) => { o.R = Math.max(-5, Math.min(0.99, v)); this.recompute(); }, { min: -5, max: 0.99, step: 0.1 });
    Rinput.disabled = o.loading !== 'custom';
    return [
      h('h3', {}, 'Fatigue options'),
      field('Each cycle', seg([['zero', '0 → load', 'The loads are applied and removed (R = 0)'], ['reversed', '± load', 'The loads reverse fully (R = −1)'], ['custom', 'Custom']], o.loading, (v) => {
        o.loading = v;
        if (v === 'zero') o.R = 0;
        if (v === 'reversed') o.R = -1;
        Rinput.value = o.R;
        Rinput.disabled = v !== 'custom';
        this.recompute();
      })),
      field('Load ratio R (min / max)', Rinput),
      field('Design life (cycles)', numberInput(o.cycles, (v) => { o.cycles = Math.max(1, v || 1e6); this.recompute(); }, { min: 1, step: 'any' })),
      field('Load scale', numberInput(o.scale, (v) => { o.scale = Math.max(0, v || 1); this.recompute(); }, { min: 0, step: 0.1 })),
      mat.fatigue.metal ? field('Surface finish', select(FINISHES, o.finish, (v) => { o.finish = v; this.recompute(); })) : null,
      h('p.muted.small', { style: { margin: '6px 0 0' } }, `${mat.name}: fatigue strength ≈ ${num(mat.fatigue.Se)} MPa at ${cyclesText(mat.fatigue.Ne)} cycles (polished)${mat.fatigue.endurance ? ', with an endurance limit' : ''}. Edit it under Material (Custom) if you have datasheet values.`),
    ];
  }

  async run() {
    const prep = await this.prepareRun();
    if (!prep) return;
    const res = await this.job({ type: 'solve', ...prep.msg }, 'Fatigue: solving the stresses…');
    if (!res) return;
    const mapped = this.panel.mapResult(res, prep.asm, prep);
    this.result = { mapped, material: completeMaterial(prep.material), units: prep.units, totalF: Math.hypot(...prep.asm.total), engine: res.engine, gpuNote: res.gpuNote };
    this.stale = false;
    this.panel.display = 'study';
    this.recompute();
    // an all-blue "unlimited life" plot says little: show the margin instead
    if (!Number.isFinite(this.result.fat.minLife) && this.view.plot === 'life') { this.view.plot = 'fos'; this.show(); }
    const f = this.result.fat;
    this.app.status(`Fatigue ${this.engineNote(res)}: shortest life ${cyclesText(f.minLife)} cycles; safety factor ${num(f.minFos)} for ${cyclesText(this.opts.cycles)} cycles.`, f.minLife < this.opts.cycles ? 'error' : '');
  }

  recompute() {
    const r = this.result;
    if (!r) return;
    this.computeFat();
    this.show();
  }

  computeFat() {
    const r = this.result, m = r.mapped;
    r.fat = fatigueField(m.vm, m.p1, m.p3, { material: r.material, finish: this.opts.finish, R: this.opts.R, cycles: this.opts.cycles, scale: this.opts.scale });
  }

  viewKeys = ['plot'];
  optRules = { loading: ['zero', 'reversed', 'custom'], finish: FINISHES.map((f) => f[0]), cycles: (v) => v >= 1, scale: (v) => v >= 0, R: (v) => v >= -5 && v <= 0.99 };
  viewRules = { plot: ['life', 'damage', 'fos'] };

  // the damage, life and safety factor follow from the stresses and the options, so they are not stored
  importOptions(o) {
    super.importOptions(o);
    if (this.result) this.computeFat();
  }

  exportResult() {
    const r = this.result;
    if (!r?.mapped) return null;
    const m = r.mapped;
    const meta = { nVert: this.part.nVert, units: r.units, material: plain(r.material), totalF: r.totalF, engine: r.engine ?? null, gpuNote: r.gpuNote ?? null };
    return { meta, arrays: [{ name: 'fatigue.vm', data: m.vm, enc: 'q16' }, { name: 'fatigue.p1', data: m.p1, enc: 'q16' }, { name: 'fatigue.p3', data: m.p3, enc: 'q16' }] };
  }

  importResult(meta, arrays) {
    const nV = this.part.nVert;
    checkMeta(meta, nV, 'fatigue');
    if (!isObj(meta.material)) throw new Error('The file\'s fatigue result is damaged (no material).');
    const mapped = { vm: arr(arrays, 'fatigue.vm', nV), p1: arr(arrays, 'fatigue.p1', nV), p3: arr(arrays, 'fatigue.p3', nV) };
    this.result = { mapped, material: meta.material, units: meta.units, totalF: meta.totalF, engine: meta.engine ?? undefined, gpuNote: meta.gpuNote ?? undefined };
    this.stale = false;
    this.computeFat();
  }

  fileNote(r = this.result) {
    const f = r?.fat;
    return f ? `shortest life ${cyclesText(f.minLife)} cycles, safety factor ${num(f.minFos)}` : '';
  }

  show() {
    const r = this.result;
    if (!r || this.app.tab !== 'structural' || this.panel.study !== this.id) return;
    const f = r.fat, o = this.opts;
    const { kpis, body } = this.card('Fatigue');
    const ok = f.minLife >= o.cycles;
    kpis.replaceChildren(
      kpi('Shortest life', Number.isFinite(f.minLife) ? cyclesText(f.minLife) : 'Unlimited', 'cycles', ok ? ['good', 'meets design life'] : ['bad', 'fails before design life']),
      kpi('Damage at design life', num(o.cycles / f.minLife), '≥ 1 means failure'),
      kpi('Safety factor', num(f.minFos), `for ${cyclesText(o.cycles)} cycles`),
      kpi('Fatigue strength', `${num(f.strengthAtLife)} MPa`, `at ${cyclesText(o.cycles)} cycles${f.curve.ka < 1 ? ` · finish ×${num(f.curve.ka)}` : ''}`),
    );
    // S-N curve (x = log10 cycles)
    const ch = chartBlock('S-N curve (stress amplitude vs cycles to failure)');
    const pts = [];
    for (let e = 0; e <= 9; e += 0.1) pts.push({ x: e, y: strengthAt(f.curve, 10 ** e) });
    // the worst point's equivalent amplitude
    const v = f.worst, m = r.mapped;
    let cur = -1;
    if (v >= 0) {
      const sign = Math.abs(m.p1[v]) >= Math.abs(m.p3[v]) ? Math.sign(m.p1[v]) || 1 : Math.sign(m.p3[v]) || 1;
      const { sa, sm } = cycleStress((sign * m.vm[v] * o.scale) / 1e6, o.R);
      const sar = goodman(sa, sm, f.curve.uts);
      const N = cyclesToFailure(f.curve, sar);
      if (Number.isFinite(N)) { pts.push({ x: Math.log10(Math.max(1, N)), y: sar }); pts.sort((a, b) => a.x - b.x); cur = pts.findIndex((p) => p.y === sar && p.x === Math.log10(Math.max(1, N))); }
    }
    const chart = new LineChart(ch.canvas, ch.tip, {
      xLabel: 'cycles to failure', integerX: true, formatX: (x) => pow10(Math.round(x)), formatY: (y) => `${num(y)}`,
      tipText: (p) => `${cyclesText(10 ** p.x)} cycles · ${num(p.y)} MPa`,
    });
    const plots = [['life', 'Life (cycles)'], ['damage', 'Damage at design life'], ['fos', 'Safety factor']];
    body.replaceChildren(
      ch.el,
      h('p.muted.small', {}, 'The marked point is the most damaged spot (equivalent fully-reversed amplitude, Goodman-corrected).'),
      field('Plot', select(plots, this.view.plot, (val) => { this.view.plot = val; this.show(); })),
      this.displayToggles(),
    );
    chart.set(pts, cur);
    this.draw();
  }

  draw() {
    const r = this.result, f = r.fat, o = this.opts;
    let values, spec;
    if (this.view.plot === 'damage') {
      values = f.damage;
      spec = { min: 0, max: Math.max(1, Math.min(10, range(values).hi)), title: 'Damage at design life', format: (x) => num(x, 2), markers: [{ value: 1, label: 'fails' }] };
    } else if (this.view.plot === 'fos') {
      const max = Math.min(10, Math.max(2, Math.ceil(f.minFos * 3)));
      values = Float32Array.from(f.fos, (x) => (Number.isNaN(x) ? NaN : Math.min(x, max)));
      spec = { min: 0, max, title: 'Fatigue safety factor', reverse: true, format: (x) => num(x, 2), markers: [{ value: 1, label: 'FOS = 1' }] };
    } else {
      // log10 life, capped at 1e9 (unlimited)
      values = Float32Array.from(f.life, (x) => (Number.isNaN(x) ? NaN : Math.min(9, Math.log10(Math.max(1, x)))));
      spec = { min: 0, max: 9, title: 'Life (cycles)', reverse: true, format: (x) => (Math.abs(x - Math.round(x)) < 0.01 ? pow10(Math.round(x)) : `10^${num(x, 2)}`), markers: [{ value: Math.log10(o.cycles), label: 'design life' }] };
    }
    this.paint(values, { ...spec, sub: `${r.material.name} · R = ${num(o.R)} · ×${num(o.scale)} loads` });
    this.shown = { values, fmt: this.view.plot === 'life' ? (x) => `${cyclesText(10 ** x)} cycles` : spec.format };
    this.app.viewer.setDeformation(null);
    this.marker(f.worst, null, 0, 'Fatigue crack starts here');
    this.panel.drawOverlays();
  }
}

// ---------------------------------------------------------------------------------------------
// Drop test

class DropStudy extends Study {
  id = 'drop';
  opts = { height: 1 };
  view = { plot: 'peak', frame: 0, playing: false, t: 0, exaggerate: 0 };

  options() {
    return [
      h('h3', {}, 'Drop options'),
      field('Drop height (m)', numberInput(this.opts.height, (v) => { this.opts.height = Math.max(0.001, v || 1); this.panel.markStale(); }, { min: 0.001, step: 0.1 })),
      h('p.muted.small', { style: { margin: '6px 0 0' } }, `Impact speed ${num(Math.sqrt(2 * 9.81 * this.opts.height))} m/s onto a rigid, frictionless floor.`),
    ];
  }

  async run() {
    const prep = await this.prepareRun({ requireLoads: false, requireFixtures: false });
    if (!prep) return;
    const r = { frames: [], material: prep.material, units: prep.units, done: false };
    this.result = r;
    this.panel.display = 'study';
    const res = await this.job({ type: 'drop', ...prep.msg, height: this.opts.height }, 'Drop test…', (d) => { r.frames.push(d); });
    if (!res) { this.result = null; return; }
    Object.assign(r, res, { done: true });
    const { hi, arg } = range(res.vmMax);
    r.peak = hi; r.peakAt = arg;
    let maxU = 0;
    for (const fr of r.frames) maxU = Math.max(maxU, maxAbs3(fr.u));
    r.autoScale = maxU > 0 ? Math.min(2000, Math.max(1, (0.05 * this.part.bbox.diag) / maxU)) : 1;
    this.view.frame = r.frames.findIndex((fr) => fr.t >= res.tPeak[arg]);
    if (this.view.frame < 0) this.view.frame = r.frames.length - 1;
    this.view.exaggerate = r.autoScale;
    this.stale = false;
    this.show();
    const mat = r.material;
    const verdict = hi >= mat.uts * 1e6 ? 'it breaks' : hi >= mat.yield * 1e6 ? (mat.brittle ? 'it cracks' : 'it bends permanently') : 'it survives';
    this.app.status(`Drop from ${num(this.opts.height)} m ${this.engineNote(res)}: peak stress ${stress(hi)} - ${verdict}. ${r.steps} time steps of ${timeText(r.dt)}.`, hi >= mat.yield * 1e6 ? 'error' : '');
  }

  viewKeys = ['plot', 'frame', 'exaggerate'];
  optRules = { height: (v) => v >= 0.001 };
  viewRules = { plot: ['peak', 'frame'], exaggerate: (v) => v >= 0 };

  fitView() {
    const r = this.result, v = this.view;
    v.playing = false;
    if (r?.done) v.frame = Math.max(0, Math.min(r.frames.length - 1, Math.round(v.frame)));
  }

  exportResult() {
    const r = this.result;
    if (!r?.done || !r.frames.length) return null;
    const arrays = [
      { name: 'drop.vmMax', data: r.vmMax, enc: 'q16' },
      { name: 'drop.times', data: r.times, enc: 'f32' },
      { name: 'drop.forces', data: r.forces, enc: 'f32' },
    ];
    arrays.push(...seriesSpecs(r.frames.map((fr) => fr.u), (i) => `drop.frame.${i}.u`, { stride: 3 }), ...seriesSpecs(r.frames.map((fr) => fr.vm), (i) => `drop.frame.${i}.vm`));
    const peakFrame = r.frames.findIndex((fr) => fr.t >= r.tPeak[r.peakAt]);
    const meta = {
      nVert: this.part.nVert, units: r.units, material: plain(r.material), height: this.opts.height, speed: r.speed, mass: r.mass,
      peak: r.peak, peakAt: r.peakAt, peakForce: r.peakForce, contactTime: r.contactTime, duration: r.duration, steps: r.steps, dt: r.dt,
      rebounded: !!r.rebounded, autoScale: r.autoScale, engine: r.engine ?? null, gpuNote: r.gpuNote ?? null,
      nTimes: r.times.length, peakFrame: peakFrame < 0 ? r.frames.length - 1 : peakFrame,
      frameTimes: r.frames.map((fr) => fr.t), frameForces: r.frames.map((fr) => fr.force),
    };
    return { meta, arrays };
  }

  importResult(meta, arrays) {
    const nV = this.part.nVert;
    checkMeta(meta, nV, 'drop test');
    if (!finiteList(meta.frameTimes) || !meta.frameTimes.length || !isObj(meta.material)) throw new Error('The file\'s drop test result is damaged (no frames).');
    const times = arr(arrays, 'drop.times', meta.nTimes), forces = arr(arrays, 'drop.forces', times.length);
    const frames = meta.frameTimes.map((t, i) => ({ t, force: meta.frameForces?.[i] ?? 0, u: arr(arrays, `drop.frame.${i}.u`, 3 * nV), vm: arr(arrays, `drop.frame.${i}.vm`, nV) }));
    const vmMax = arr(arrays, 'drop.vmMax', nV);
    const r = {
      frames, material: meta.material, units: meta.units, done: true, vmMax, times, forces, speed: meta.speed, mass: meta.mass,
      peakForce: meta.peakForce, contactTime: meta.contactTime, duration: meta.duration, steps: meta.steps, dt: meta.dt, rebounded: !!meta.rebounded,
      engine: meta.engine ?? undefined, gpuNote: meta.gpuNote ?? undefined, peak: meta.peak, peakAt: meta.peakAt, autoScale: meta.autoScale,
    };
    this.result = r;
    this.opts.height = meta.height > 0 ? meta.height : this.opts.height;
    this.stale = false;
    // the view a run ends on: the frame at the peak, the shape scaled to be visible
    this.view.frame = Number.isInteger(meta.peakFrame) ? Math.max(0, Math.min(frames.length - 1, meta.peakFrame)) : frames.length - 1;
    this.view.exaggerate = r.autoScale;
    this.view.plot = 'peak';
    this.view.playing = false;
  }

  fileNote(r = this.result) {
    return r?.done ? `peak stress ${stress(r.peak)}, ${r.frames.length} frames` : '';
  }

  show() {
    const r = this.result;
    if (!r?.done || this.app.tab !== 'structural' || this.panel.study !== this.id) return;
    const { kpis, body } = this.card('Drop test');
    const mat = r.material;
    const status = r.peak >= mat.uts * 1e6 ? ['bad', 'breaks'] : r.peak >= mat.yield * 1e6 ? ['bad', mat.brittle ? 'cracks' : 'dents / bends'] : ['good', 'survives'];
    kpis.replaceChildren(
      kpi('Peak stress', stress(r.peak), `yield ${num(mat.yield)} · UTS ${num(mat.uts)} MPa`, status),
      kpi('Peak impact force', force(r.peakForce), `${num(r.peakForce / (r.mass * 9.81))} g deceleration`),
      kpi('Contact time', timeText(r.contactTime), r.rebounded ? 'bounced off the floor' : 'still in contact at the end'),
      kpi('Impact speed', `${num(r.speed)} m/s`, `mass ${num(r.mass * 1000)} g`),
    );
    const ch = chartBlock('Floor force during the impact');
    const pts = Array.from(r.times, (t, i) => ({ x: t * 1e3, y: r.forces[i] }));
    const chart = new LineChart(ch.canvas, ch.tip, {
      xLabel: 'time (ms)', integerX: false, formatX: (x) => num(x), formatY: (y) => force(y),
      tipText: (p) => `${num(p.x)} ms · ${force(p.y)}`,
      onPick: (i) => { const t = pts[i].x / 1e3; this.view.frame = Math.max(0, r.frames.findIndex((f) => f.t >= t)); this.view.plot = 'frame'; this.show(); },
    });
    const fr = r.frames[this.view.frame];
    const plots = [['peak', 'Peak stress over the whole impact'], ['frame', 'Stress at this moment']];
    body.replaceChildren(
      ch.el,
      field('Plot', select(plots, this.view.plot, (v) => { this.view.plot = v; this.show(); })),
      field(`Moment (${fr ? timeText(fr.t) : '–'})`, h('input', { type: 'range', min: 0, max: r.frames.length - 1, value: this.view.frame, oninput: (e) => { this.view.frame = Number(e.target.value); this.view.plot = 'frame'; this.view.playing = false; this.show(); } })),
      h('div.row.gap.wrap', {}, h('button.btn.small', { onclick: () => this.togglePlay() }, this.view.playing ? '❚❚ Pause' : '▶ Play impact')),
      field('Exaggerate shape', h('input', { type: 'range', min: 0, max: 100, value: Math.round(Math.log(Math.max(1, this.view.exaggerate)) / Math.log(Math.max(2, 2 * r.autoScale)) * 100), oninput: (e) => { this.view.exaggerate = Math.pow(Math.max(2, 2 * r.autoScale), Number(e.target.value) / 100); this.draw(); } })),
      h('p.muted.small', {}, 'Stress is recovered at voxel corners like the static study; sharp corners that hit the floor show local peaks. Linear-elastic: beyond yield it shows where it would dent or crack, not how far.'),
      this.displayToggles(),
    );
    const t0 = r.frames[this.view.frame]?.t;
    chart.set(pts, t0 != null ? Math.max(0, pts.findIndex((p) => p.x >= t0 * 1e3)) : -1);
    this.draw();
  }

  draw() {
    const r = this.result;
    if (!r?.done) return;
    const mat = r.material;
    const fr = r.frames[this.view.frame];
    const hi = Math.max(r.peak, mat.yield * 1e6);
    const values = this.view.plot === 'frame' && fr ? fr.vm : r.vmMax;
    const fmt = stressFormatter(hi);
    this.paint(values, {
      min: 0, max: hi, title: this.view.plot === 'frame' ? 'von Mises stress' : 'Peak von Mises (whole impact)', format: fmt,
      sub: `${mat.name} · drop ${num(this.opts.height)} m${this.view.plot === 'frame' && fr ? ` · t = ${timeText(fr.t)}` : ''} · shape ×${num(this.view.exaggerate)}`,
      markers: [{ value: mat.yield * 1e6, label: `Yield ${num(mat.yield)}` }, { value: mat.uts * 1e6, label: `UTS ${num(mat.uts)}` }],
    });
    this.shown = { values, fmt };
    this.app.viewer.setDeformation(fr ? fr.u : null, this.view.exaggerate);
    this.marker(this.view.plot === 'frame' ? range(values).arg : r.peakAt, fr?.u, this.view.exaggerate, 'Highest impact stress');
    this.panel.drawOverlays();
  }

  togglePlay() {
    const r = this.result;
    if (!r?.frames.length) return;
    this.view.playing = !this.view.playing;
    if (this.view.playing) {
      if (this.view.frame >= r.frames.length - 1) this.view.frame = 0;
      this.view.plot = 'frame';
      this.view.t = 0;
    }
    this.show();
  }

  tick(dt) {
    if (!this.view.playing || !this.result) return;
    this.view.t += dt;
    if (this.view.t < 0.08) return;
    this.view.t = 0;
    if (this.view.frame >= this.result.frames.length - 1) { this.view.playing = false; this.show(); return; }
    this.view.frame++;
    this.show();
  }
}

// ---------------------------------------------------------------------------------------------
// Linear dynamic

class DynamicStudy extends Study {
  id = 'dynamic';
  opts = { source: 'loads', dir: 1, type: 'harmonic', amp: 1, zeta: 2, nev: 10, fOp: 0, pulse: 5, sineF: 50, duration: 0.2, quake: 10 };
  view = { freq: 0, index: 0, plot: 'vm', phase: 0, playing: false, animate: true };

  options() {
    const o = this.opts;
    const base = o.source === 'base';
    const els = [
      h('h3', {}, 'Dynamic options'),
      field('Excitation', seg([['loads', 'The loads', 'The forces and pressures you set vary in time'], ['base', 'Base shaking', 'The fixtures shake (vibration table, vehicle, earthquake)']], o.source, (v) => { o.source = v; if (v === 'base' && o.type === 'harmonic') o.amp = 1; this.panel.renderStudyOptions(); this.panel.markStale(); })),
    ];
    if (base) els.push(field('Shaking direction', seg([[0, 'X'], [1, 'Y'], [2, 'Z']], o.dir, (v) => { o.dir = v; this.panel.markStale(); })));
    els.push(field('Type', select([['harmonic', 'Frequency sweep (steady vibration)'], ['shock', 'Shock pulse (half-sine)'], ['sine', 'Sine burst'], ['quake', 'Earthquake (synthetic)']], o.type, (v) => { o.type = v; this.panel.renderStudyOptions(); this.recompute(); })));
    els.push(field(base ? 'Peak acceleration (g)' : 'Load multiplier', numberInput(o.amp, (v) => { o.amp = v || 1; this.recompute(); }, { step: 'any' })));
    if (o.type === 'shock') els.push(field('Pulse duration (ms)', numberInput(o.pulse, (v) => { o.pulse = Math.max(0.01, v || 5); this.recompute(); }, { min: 0.01, step: 'any' })));
    if (o.type === 'sine') {
      els.push(field('Frequency (Hz)', numberInput(o.sineF, (v) => { o.sineF = Math.max(0.01, v || 50); this.recompute(); }, { min: 0.01, step: 'any' })));
      els.push(field('Duration (s)', numberInput(o.duration, (v) => { o.duration = Math.max(1e-4, v || 0.2); this.recompute(); }, { min: 0.0001, step: 'any' })));
    }
    if (o.type === 'quake') els.push(field('Duration (s)', numberInput(o.quake, (v) => { o.quake = Math.max(0.5, v || 10); this.recompute(); }, { min: 0.5, step: 'any' })));
    els.push(field('Damping (% of critical)', numberInput(o.zeta, (v) => { o.zeta = Math.max(0.01, Math.min(50, v || 2)); this.recompute(); }, { min: 0.01, max: 50, step: 0.5 })));
    els.push(field('Modes used', numberInput(o.nev, (v) => { o.nev = Math.max(1, Math.min(20, Math.round(v) || 10)); this.panel.markStale(); }, { min: 1, max: 20, step: 1 })));
    els.push(h('p.muted.small', { style: { margin: '6px 0 0' } }, 'Metals ≈ 1–2 % damping, bolted assemblies 3–5 %, plastics and rubber mounts 5–10 %.'));
    return els;
  }

  async run() {
    const o = this.opts;
    const prep = await this.prepareRun({ requireLoads: o.source === 'loads' });
    if (!prep) return;
    const dir = [0, 0, 0];
    dir[o.dir] = 1;
    const res = await this.job({ type: 'modal', ...prep.msg, nev: o.nev, dynamic: { kind: o.source === 'base' ? 'base' : 'force', dir } }, 'Linear dynamic: natural modes…');
    if (!res) return;
    if (res.free) { this.app.status('Linear dynamic needs fixtures (what the part is mounted on).', 'error'); return; }
    this.result = { ...res, units: prep.units, material: prep.material, source: o.source, totalF: Math.hypot(...prep.asm.total), diag: this.part.bbox.diag };
    this.stale = false;
    this.panel.display = 'study';
    // sample vertices for curves: the most stressed / displaced in the static and modal fields
    const b = res.basis, nV = this.part.nVert;
    const pick = new Set();
    const top = (vals, k) => {
      const idx = Array.from({ length: nV }, (_, i) => i).filter((i) => !Number.isNaN(vals[i]));
      idx.sort((a, c) => vals[c] - vals[a]);
      for (let i = 0; i < Math.min(k, idx.length); i++) pick.add(idx[i]);
    };
    const vmOf = (S) => { const out = new Float32Array(nV); for (let v = 0; v < nV; v++) { const s = S.subarray(6 * v, 6 * v + 6); out[v] = Math.sqrt(0.5 * ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2) + 3 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)); } return out; };
    top(vmOf(b.staticS), 150);
    top(magnitudes(b.staticU), 30);
    b.modeS.forEach((S) => top(vmOf(S), 60));
    b.modeU.forEach((U) => top(magnitudes(U), 15));
    this.result.samples = Int32Array.from(pick);
    this.view.freq = 0;
    this.recompute();
    this.app.status(`Linear dynamic ${this.engineNote(res)} with ${res.basis.omegas.length} modes (${res.modes.map((m) => freqText(m.freq)).slice(0, 4).join(', ')}${res.modes.length > 4 ? ', …' : ''}).`);
  }

  // not stored in version 1 (the modal basis is large): exportResult() is null, Re-run solves again
  viewKeys = ['plot', 'animate'];
  optRules = { source: ['loads', 'base'], type: ['harmonic', 'shock', 'sine', 'quake'], dir: [0, 1, 2], nev: (v) => v >= 1 && v <= 20 };
  viewRules = { plot: ['vm', 'disp', 'envelope', 'now'] };

  scale() {
    // pattern is per unit load (loads) or per 1 m/s^2 of base acceleration
    return this.opts.source === 'base' ? this.opts.amp * 9.81 : this.opts.amp;
  }

  recompute() {
    const r = this.result;
    if (!r) return;
    const o = this.opts, b = r.basis, zeta = o.zeta / 100, k = this.scale();
    if (o.type === 'harmonic') {
      const fMaxMode = Math.max(...r.modes.map((m) => m.freq));
      const fMax = fMaxMode * 1.3;
      const freqs = new Set();
      for (let i = 1; i <= 300; i++) freqs.add((fMax * i) / 300);
      for (const m of r.modes) for (let d = -8; d <= 8; d++) freqs.add(Math.max(1e-3, m.freq * (1 + d * zeta * 0.5)));
      const list = Array.from(freqs).sort((a, c) => a - c);
      const curve = list.map((f) => {
        const fld = harmonicField(b, 2 * Math.PI * f, zeta, { phases: 8, vertices: r.samples });
        let s = 0, u = 0;
        for (let i = 0; i < r.samples.length; i++) { if (fld.vm[i] > s) s = fld.vm[i]; if (fld.disp[i] > u) u = fld.disp[i]; }
        return { f, vm: s * Math.abs(k), disp: u * Math.abs(k) };
      });
      r.curve = curve;
      const worst = curve.reduce((a, c) => (c.vm > a.vm ? c : a), curve[0]);
      r.worstFreq = worst.f;
      r.staticVM = curve[0].vm;
      if (!this.view.freq) this.view.freq = o.fOp > 0 ? o.fOp : worst.f;
      this.fieldAtFreq();
    } else {
      const g0 = excitation({ kind: o.type === 'shock' ? 'shock' : o.type, amplitude: 1, freq: o.sineF, duration: o.pulse / 1000, total: o.type === 'quake' ? o.quake : o.duration });
      const T1 = 1 / Math.min(...r.modes.map((m) => m.freq));
      const total = o.type === 'shock' ? Math.max(4 * o.pulse / 1000, 6 * T1) : o.type === 'quake' ? o.quake * 1.1 : o.duration + 3 * T1;
      const hist = timeHistory(b, g0, { zeta, total });
      r.hist = hist;
      const n = hist.times.length;
      const every = Math.max(1, Math.ceil(n / 1500));
      const pts = [];
      for (let i = 0; i < n; i += every) {
        const vm = stressAt(b, hist, i, r.samples);
        let s = 0;
        for (const v of vm) if (v > s) s = v;
        pts.push({ i, t: hist.times[i], vm: s * Math.abs(k), g: hist.g[i] });
      }
      r.tcurve = pts;
      const worst = pts.reduce((a, c) => (c.vm > a.vm ? c : a), pts[0]);
      r.worstIndex = worst.i;
      // envelope over the largest local peaks of the response
      const peaks = pts.filter((p, j) => j > 0 && j < pts.length - 1 && p.vm >= pts[j - 1].vm && p.vm >= pts[j + 1].vm).sort((a, c) => c.vm - a.vm).slice(0, 10);
      const env = new Float32Array(this.part.nVert);
      for (const p of [worst, ...peaks]) {
        const vm = stressAt(b, hist, p.i);
        for (let v = 0; v < env.length; v++) env[v] = Number.isNaN(vm[v]) ? NaN : Math.max(env[v], vm[v] * Math.abs(k));
      }
      r.envelope = env;
      this.view.index = worst.i;
      this.view.plot = 'envelope';
    }
    this.show();
  }

  fieldAtFreq() {
    const r = this.result, k = Math.abs(this.scale());
    const fld = harmonicField(r.basis, 2 * Math.PI * this.view.freq, this.opts.zeta / 100, { phases: 16 });
    r.field = { vm: fld.vm.map((x) => x * k), disp: fld.disp.map((x) => x * k), f: this.view.freq };
  }

  show() {
    const r = this.result;
    if (!r || this.app.tab !== 'structural' || this.panel.study !== this.id) return;
    const o = this.opts;
    const { kpis, body } = this.card(o.type === 'harmonic' ? 'Frequency response' : 'Time response');
    const mat = r.material;
    const unitIn = o.source === 'base' ? `${num(o.amp)} g base shaking` : `${num(o.amp)} × the loads`;
    if (o.type === 'harmonic') {
      const worst = r.curve.reduce((a, c) => (c.vm > a.vm ? c : a), r.curve[0]);
      const f = r.field;
      const mx = range(f.vm).hi, du = range(f.disp).hi;
      kpis.replaceChildren(
        kpi('First resonance', freqText(r.modes[0].freq), `${r.modes.length} modes`),
        kpi('Worst resonance', freqText(worst.f), `${stress(worst.vm)} peak stress`, worst.vm >= mat.yield * 1e6 ? ['bad', 'yields'] : null),
        kpi(`At ${freqText(f.f)}`, stress(mx), `${num(du)} ${r.units} vibration amplitude`),
        kpi('Amplification', r.staticVM > 0 ? `× ${num(worst.vm / r.staticVM)}` : '–', 'worst resonance vs steady load'),
      );
      const ch = chartBlock(`Peak stress vs frequency (${unitIn})`);
      const pts = r.curve.map((p) => ({ x: p.f, y: p.vm }));
      const cur = pts.reduce((best, p, i) => (Math.abs(p.x - f.f) < Math.abs(pts[best].x - f.f) ? i : best), 0);
      const chart = new LineChart(ch.canvas, ch.tip, {
        xLabel: 'frequency (Hz)', integerX: false, logY: true, formatX: (x) => num(x), formatY: (y) => stress(y),
        tipText: (p) => `${freqText(p.x)} · ${stress(p.y)}`,
        onPick: (i) => { this.view.freq = pts[i].x; this.fieldAtFreq(); this.show(); },
      });
      body.replaceChildren(
        ch.el,
        h('p.muted.small', {}, 'Click the curve to see the part vibrating at that frequency.'),
        field('Frequency (Hz)', numberInput(Number(f.f.toPrecision(4)), (v) => { if (v > 0) { this.view.freq = v; this.fieldAtFreq(); this.show(); } }, { min: 0, step: 'any' })),
        field('Plot', select([['vm', 'Stress amplitude (peak von Mises)'], ['disp', 'Vibration amplitude']], this.view.plot === 'disp' ? 'disp' : 'vm', (v) => { this.view.plot = v; this.show(); })),
        h('label.check', {}, h('input', { type: 'checkbox', checked: this.view.animate, onchange: (e) => { this.view.animate = e.target.checked; this.draw(); } }), ' Animate vibration'),
        this.displayToggles(),
      );
      chart.set(pts, cur);
    } else {
      const worst = r.tcurve.reduce((a, c) => (c.vm > a.vm ? c : a), r.tcurve[0]);
      const env = range(r.envelope).hi;
      const b = r.basis;
      const u = shapeAt(b, r.hist, this.view.index);
      const k = Math.abs(this.scale());
      kpis.replaceChildren(
        kpi('Peak stress', stress(env), `at ${timeText(worst.t)}`, env >= mat.uts * 1e6 ? ['bad', 'breaks'] : env >= mat.yield * 1e6 ? ['bad', 'yields'] : ['good', 'elastic']),
        kpi('Displacement now', `${num(maxAbs3(u) * k)} ${r.units}`, `t = ${timeText(r.hist.times[this.view.index])}`),
        kpi('Lowest mode', freqText(r.modes[0].freq), `${r.modes.length} modes, ${num(o.zeta)}% damping`),
      );
      const ch = chartBlock(`Peak stress over time (${unitIn})`);
      const pts = r.tcurve.map((p) => ({ x: p.t * 1e3, y: p.vm }));
      const cur = r.tcurve.reduce((best, p, i) => (Math.abs(p.i - this.view.index) < Math.abs(r.tcurve[best].i - this.view.index) ? i : best), 0);
      const chart = new LineChart(ch.canvas, ch.tip, {
        xLabel: 'time (ms)', integerX: false, formatX: (x) => num(x), formatY: (y) => stress(y),
        tipText: (p) => `${num(p.x)} ms · ${stress(p.y)}`,
        onPick: (i) => { this.view.index = r.tcurve[i].i; this.view.plot = 'now'; this.view.playing = false; this.show(); },
      });
      body.replaceChildren(
        ch.el,
        field('Time', h('input', { type: 'range', min: 0, max: r.hist.times.length - 1, value: this.view.index, oninput: (e) => { this.view.index = Number(e.target.value); this.view.plot = 'now'; this.view.playing = false; this.show(); } })),
        h('div.row.gap.wrap', {}, h('button.btn.small', { onclick: () => { this.view.playing = !this.view.playing; if (this.view.playing && this.view.index >= r.hist.times.length - 1) this.view.index = 0; this.view.plot = 'now'; this.show(); } }, this.view.playing ? '❚❚ Pause' : '▶ Play')),
        field('Plot', select([['envelope', 'Peak stress over the whole event'], ['now', 'Stress at this moment']], this.view.plot === 'now' ? 'now' : 'envelope', (v) => { this.view.plot = v; this.show(); })),
        this.displayToggles(),
      );
      chart.set(pts, cur);
    }
    this.draw();
  }

  draw() {
    const r = this.result;
    if (!r) return;
    const mat = r.material, o = this.opts, k = Math.abs(this.scale());
    let values, u = null, scale = 1, sub;
    if (o.type === 'harmonic') {
      const f = r.field;
      values = this.view.plot === 'disp' ? f.disp : f.vm;
      u = harmonicShape(r.basis, 2 * Math.PI * f.f, o.zeta / 100, this.view.animate ? this.view.phase : 0);
      // scale from the vibration amplitude (the shape at one phase can be near zero)
      const amp = (range(f.disp).hi || 0) / (k || 1);
      scale = amp > 0 ? Math.min(1e6, (0.05 * r.diag) / amp) : 1;
      sub = `${mat.name} · ${freqText(f.f)} · ${num(o.zeta)}% damping · shape exaggerated`;
    } else {
      const b = r.basis;
      values = this.view.plot === 'now' ? stressAt(b, r.hist, this.view.index).map((x) => x * k) : r.envelope;
      u = shapeAt(b, r.hist, this.view.index);
      const peakU = Math.max(maxAbs3(shapeAt(b, r.hist, r.worstIndex)), 1e-30);
      scale = Math.min(1e6, (0.05 * r.diag) / peakU);
      sub = `${mat.name} · ${this.view.plot === 'now' ? `t = ${timeText(r.hist.times[this.view.index])}` : 'peak over the event'} · shape exaggerated`;
    }
    const disp = o.type === 'harmonic' && this.view.plot === 'disp';
    // time plots keep one scale (the event's peak) so the animation compares like with like
    const peak = o.type === 'harmonic' || disp ? range(values).hi : Math.max(range(r.envelope).hi, range(values).hi);
    const hi = peak > 0 ? peak : 1;
    const fmt = disp ? (x) => `${num(x)} ${r.units}` : stressFormatter(hi);
    this.paint(values, {
      min: 0, max: hi, title: disp ? 'Vibration amplitude' : o.type === 'harmonic' ? 'Stress amplitude' : 'von Mises stress', format: fmt, sub,
      markers: disp || hi < mat.yield * 1e6 ? [] : [{ value: mat.yield * 1e6, label: `Yield ${num(mat.yield)}` }],
    });
    this.shown = { values, fmt };
    this.app.viewer.setDeformation(u, scale);
    this.drawScale = scale;
    this.marker(range(values).arg, u, scale, 'Highest dynamic stress');
    this.panel.drawOverlays();
  }

  tick(dt) {
    const r = this.result;
    if (!r || this.panel.display !== 'study') return;
    const o = this.opts;
    if (o.type === 'harmonic' && this.view.animate && r.field) {
      this.view.phase += dt * Math.PI * 1.2;
      const u = harmonicShape(r.basis, 2 * Math.PI * r.field.f, o.zeta / 100, this.view.phase);
      this.app.viewer.setDeformation(u, this.drawScale || 1);
    } else if (o.type !== 'harmonic' && this.view.playing && r.hist) {
      const n = r.hist.times.length;
      this.view.index = Math.min(n - 1, this.view.index + Math.max(1, Math.round((n * dt) / 6)));
      if (this.view.index >= n - 1) this.view.playing = false;
      this.show();
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Optimization

class OptimizeStudy extends Study {
  id = 'optimize';
  opts = { goal: 'topology', keep: 40, iters: 30, fos: 2, maxDisp: 0, objective: 'mass', family: 'all' };
  view = { level: 0.5 };

  options() {
    const o = this.opts;
    const els = [
      h('h3', {}, 'Optimization'),
      field('Goal', seg([['topology', 'Remove material'], ['sizing', 'Material & size']], o.goal, (v) => { o.goal = v; this.panel.renderStudyOptions(); this.panel.markStale(); })),
    ];
    if (o.goal === 'topology') {
      els.push(
        field('Keep (% of the mass)', numberInput(o.keep, (v) => { o.keep = Math.max(5, Math.min(95, v || 40)); this.panel.markStale(); }, { min: 5, max: 95, step: 5 })),
        field('Iterations', numberInput(o.iters, (v) => { o.iters = Math.max(5, Math.min(100, Math.round(v) || 30)); this.panel.markStale(); }, { min: 5, max: 100, step: 5 })),
        h('p.muted.small', { style: { margin: '6px 0 0' } }, 'Finds the stiffest shape that uses this much material for your fixtures and loads. Material at the fixtures and loads is always kept. Each iteration is one full solve, so use a coarse mesh first.'),
      );
    } else {
      els.push(
        field('Minimum safety factor', numberInput(o.fos, (v) => { o.fos = Math.max(0.1, v || 2); this.panel.markStale(); }, { min: 0.1, step: 0.1 })),
        field(`Max displacement (${this.app.units}, 0 = any)`, numberInput(o.maxDisp, (v) => { o.maxDisp = Math.max(0, v || 0); this.panel.markStale(); }, { min: 0, step: 'any' })),
        field('Minimize', seg([['mass', 'Weight'], ['cost', 'Material cost']], o.objective, (v) => { o.objective = v; this.show(); })),
        field('Materials', seg([['all', 'All'], ['metals', 'Metals'], ['plastics', 'Plastics']], o.family, (v) => { o.family = v; this.panel.markStale(); })),
        h('p.muted.small', { style: { margin: '6px 0 0' } }, 'Scales the whole part (0.25× to 4×) for each material to the smallest size that keeps the safety factor. Costs are rough raw-material prices.'),
      );
    }
    return els;
  }

  clear() {
    super.clear();
    this.app.viewer.setShape?.(null);
  }

  viewKeys = ['level'];
  optRules = {
    goal: ['topology', 'sizing'], objective: ['mass', 'cost'], family: ['all', 'metals', 'plastics'],
    keep: (v) => v >= 5 && v <= 95, iters: (v) => v >= 5 && v <= 100, fos: (v) => v >= 0.1, maxDisp: (v) => v >= 0,
  };
  viewRules = { level: (v) => v >= 0.2 && v <= 0.8 };

  importOptions(o) {
    const level = this.view.level;
    super.importOptions(o);
    if (this.result?.goal === 'topology' && this.result.done && this.view.level !== level) this.buildShape();
  }

  exportResult() {
    const r = this.result;
    if (!r) return null;
    if (r.goal === 'sizing') {
      if (!r.rows) return null;
      return { meta: { goal: 'sizing', units: r.units, material: plain(r.material), engine: r.engine ?? null, gpuNote: r.gpuNote ?? null, base: plain(r.base), rows: plain(r.rows) }, arrays: [] };
    }
    if (!r.done || !r.density) return null;
    const m = r.model;
    const meta = {
      goal: 'topology', units: r.units, material: plain(r.material), toMeters: r.toMeters, resolution: m.resolution, dims: Array.from(m.dims),
      history: plain(r.history), keptFraction: r.keptFraction ?? null, engine: r.engine ?? null,
    };
    return { meta, arrays: [{ name: 'optimize.density', data: r.density, enc: 'q8', dims: Array.from(m.dims) }] };
  }

  importResult(meta, arrays) {
    if (!isObj(meta) || !isObj(meta.material)) throw new Error('The file\'s optimization result is damaged (no material).');
    if (meta.goal === 'sizing') {
      if (!Array.isArray(meta.rows) || !isObj(meta.base)) throw new Error('The file\'s optimization result is damaged (no table).');
      this.result = { goal: 'sizing', rows: meta.rows, base: meta.base, units: meta.units, material: meta.material, engine: meta.engine ?? undefined, gpuNote: meta.gpuNote ?? undefined };
      this.opts.goal = 'sizing';
      this.stale = false;
      return;
    }
    if (meta.goal !== 'topology') throw new Error('The file\'s optimization result has an unknown goal.');
    if (!Number.isInteger(meta.resolution) || meta.resolution < 1 || !Array.isArray(meta.dims) || !Array.isArray(meta.history)) throw new Error('The file\'s topology result is damaged.');
    const model = this.panel.modelFor(meta.resolution);
    const [nx, ny, nz] = model.dims;
    if (meta.dims.join() !== `${nx},${ny},${nz}`) throw new Error(`The file's topology result was made on a ${meta.dims.join('x')} voxel grid; this part gives ${nx}x${ny}x${nz}.`);
    const density = arr(arrays, 'optimize.density', nx * ny * nz);
    this.result = {
      goal: 'topology', model, material: meta.material, units: meta.units, history: meta.history, density, toMeters: meta.toMeters,
      keptFraction: meta.keptFraction ?? undefined, engine: meta.engine ?? undefined, done: true,
    };
    this.opts.goal = 'topology';
    this.stale = false;
    this.buildShape();
    this.xrayPending = true;
  }

  fileNote(r = this.result) {
    if (!r) return '';
    if (r.goal === 'sizing') return `${r.rows.filter((x) => x.feasible).length} of ${r.rows.length} materials meet the safety factor`;
    const h1 = r.history?.[r.history.length - 1];
    return h1 ? `${r.history.length} iterations, ${num(h1.volume * 100)}% of the material kept` : '';
  }

  leave() {
    if (this.autoXRay) { this.app.setXRay(false); this.autoXRay = false; }
  }

  async run() {
    if (this.opts.goal === 'topology') return this.runTopology();
    return this.runSizing();
  }

  async runTopology() {
    const prep = await this.prepareRun();
    if (!prep) return;
    const { model, asm } = prep;
    // keep voxels touching a fixture or a loaded node
    const [nx, ny, nz] = model.dims;
    const keep = new Uint8Array(model.density.length);
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const e = i + nx * (j + ny * k);
      if (!(model.density[e] > 0)) continue;
      for (let c = 0; c < 8; c++) {
        const n = model.node(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
        if (asm.bc[3 * n] || asm.f[3 * n] || asm.f[3 * n + 1] || asm.f[3 * n + 2]) { keep[e] = 1; break; }
      }
    }
    const r = { goal: 'topology', model, material: prep.material, units: prep.units, history: [], density: null, toMeters: prep.toMeters };
    this.result = r;
    this.panel.display = 'study';
    if (!$('#chk-xray').checked) { this.app.setXRay(true); this.autoXRay = true; }
    let lastDraw = 0;
    const res = await this.job({ type: 'topology', ...prep.msg, keep, volFrac: this.opts.keep / 100, maxIter: this.opts.iters }, 'Topology optimization…', (d) => {
      r.history.push({ it: d.it, compliance: d.compliance, volume: d.volume });
      r.density = d.density;
      const now = performance.now();
      if (now - lastDraw > 400) { lastDraw = now; this.previewVoxels(d.density); }
    });
    if (!res) { if (!r.density) this.result = null; else this.show(); return; }
    Object.assign(r, { density: res.density, history: res.history, keptFraction: res.keptFraction, engine: res.engine, done: true });
    this.stale = false;
    this.buildShape();
    this.show();
    const h0 = res.history[0], h1 = res.history[res.history.length - 1];
    this.app.status(`Topology optimized ${this.engineNote(res)} in ${res.history.length} iterations: ${num(h1.volume * 100)}% of the material kept; compliance ${num(h1.compliance / h0.compliance * 100)}% of the uniform start.`);
  }

  previewVoxels(density) {
    const m = this.result.model;
    const pts = m.voxelCenters((e) => density[e] > this.view.level);
    this.app.viewer.setVoxels('mesh', pts, m.h * 0.95, 0x86a8d8);
  }

  buildShape() {
    const r = this.result, m = r.model;
    const s = surfaceNets(r.density, m.dims, m.origin, m.h, this.view.level, 3);
    r.shape = s;
    let vol = 0, total = 0;
    for (let e = 0; e < r.density.length; e++) { if (m.density[e] > 0) { total += m.density[e]; if (r.density[e] > this.view.level) vol += m.density[e]; } }
    r.shapeFraction = total > 0 ? vol / total : 0;
  }

  async runSizing() {
    const prep = await this.prepareRun();
    if (!prep) return;
    const p = this.panel, model = prep.model;
    const toM = prep.toMeters;
    const groups = [];
    const add = (name, kind, loads, gravity) => {
      if (!loads.length && !gravity) return;
      const a = model.assemble({ fixtures: p.fixtures, loads, gravity, material: prep.material, toMeters: toM });
      if (a.f.some((v, i) => v !== 0 && !a.bc[i])) groups.push({ name, kind, f: a.f });
    };
    add('forces', 'force', p.loads.filter((l) => l.type === 'force'), false);
    add('pressures & wind', 'pressure', p.loads.filter((l) => l.type !== 'force'), false);
    add('self-weight', 'gravity', [], $('#chk-gravity').checked);
    const fam = this.opts.family;
    const isPlastic = (m) => m.E < 10;
    const mats = MATERIALS.map(completeMaterial).filter((m) => fam === 'all' || (fam === 'metals' ? !isPlastic(m) && m.fatigue.metal : isPlastic(m)));
    const current = completeMaterial(prep.material);
    if (!mats.some((m) => m.id === current.id)) mats.unshift(current);
    const res = await this.job({
      type: 'sizing', ...prep.msg, f: undefined, groups, materials: mats, baseMaterial: current,
      fosTarget: this.opts.fos, maxDisp: this.opts.maxDisp, scaleRange: [0.25, 4], volume: this.part.volume,
    }, 'Comparing materials and sizes…');
    if (!res) return;
    this.result = { goal: 'sizing', ...res, units: prep.units, material: current };
    this.stale = false;
    this.panel.display = 'study';
    this.show();
    const best = this.ranked()[0];
    this.app.status(best ? `Best: ${best.name} at ${num(best.scale)}× size, ${num(best.mass * 1000)} g (${this.engineNote(res)}).` : 'No material meets the safety factor within 0.25×–4× size.', best ? '' : 'warn');
  }

  ranked() {
    const r = this.result;
    const key = this.opts.objective === 'cost' ? 'cost' : 'mass';
    return r.rows.filter((x) => x.feasible).sort((a, b) => a[key] - b[key]);
  }

  show() {
    const r = this.result;
    if (!r || this.app.tab !== 'structural' || this.panel.study !== this.id) return;
    if (r.goal === 'sizing') return this.showSizing();
    if (this.xrayPending) {
      this.xrayPending = false;
      if (!$('#chk-xray').checked) { this.app.setXRay(true); this.autoXRay = true; }
    }
    const { kpis, body } = this.card('Topology optimization');
    const h0 = r.history[0], h1 = r.history[r.history.length - 1];
    const mat = r.material;
    const massFull = this.part.volume * r.toMeters ** 3 * mat.density;
    kpis.replaceChildren(
      kpi('Material kept', `${num((r.shapeFraction ?? h1?.volume ?? 0) * 100)} %`, `≈ ${num(massFull * (r.shapeFraction ?? h1?.volume ?? 1) * 1000)} g of ${num(massFull * 1000)} g`),
      kpi('Iterations', String(r.history.length), r.done ? 'converged or limit' : 'running…'),
      kpi('Compliance', h0 && h1 ? `${num((h1.compliance / h0.compliance) * 100)} %` : '–', 'of the uniform start (lower = stiffer)'),
    );
    const ch = chartBlock('Compliance per iteration (lower is stiffer)');
    const chart = new LineChart(ch.canvas, ch.tip, { xLabel: 'iteration', formatX: (x) => String(Math.round(x)), formatY: (y) => num(y), tipText: (p) => `iteration ${p.x + 1} · ${num(p.y)}` });
    const slider = h('input', { type: 'range', min: 20, max: 80, value: Math.round(this.view.level * 100), onchange: (e) => { this.view.level = Number(e.target.value) / 100; if (r.done) { this.buildShape(); this.show(); } else this.previewVoxels(r.density); } });
    body.replaceChildren(
      ch.el,
      field('Keep voxels denser than', slider),
      r.done ? h('div.row.gap.wrap', { style: { marginTop: '8px' } },
        h('button.btn.small', { onclick: () => this.exportSTL() }, 'Export STL…'),
        h('button.btn.small', { onclick: () => this.useAsPart() }, 'Use as new part')) : null,
      h('p.muted.small', {}, 'The blue shape is the optimized design; the original part is shown see-through. Re-model it in CAD (or export the STL) and check it with a static study.'),
    );
    chart.set(r.history.map((x) => ({ x: x.it, y: x.compliance })), r.history.length - 1);
    this.draw();
  }

  draw() {
    const r = this.result;
    if (!r) return;
    const v = this.app.viewer;
    if (r.goal === 'sizing') { v.setShape?.(null); v.setVoxels('mesh', new Float32Array(0), 1, 0); return; }
    v.setScalars(null);
    v.setDeformation(null);
    v.clearMarker();
    renderLegend($('#legend'), []);
    if (r.done && r.shape) {
      v.setVoxels('mesh', new Float32Array(0), 1, 0);
      v.setShape(r.shape.positions, r.shape.index, 0x3f7fd9);
    } else if (r.density) this.previewVoxels(r.density);
    this.shown = null;
    this.panel.drawOverlays();
  }

  exportSTL() {
    const r = this.result;
    if (!r?.shape) return;
    const name = `${this.part.name || 'part'}-optimized`;
    const buf = toSTL(r.shape.positions, r.shape.index, name);
    if (this.app.desktop?.saveFile) this.app.desktop.saveFile(new Uint8Array(buf), `${name}.stl`).then((ok) => ok && this.app.status('STL saved.'));
    else {
      const url = URL.createObjectURL(new Blob([buf], { type: 'model/stl' }));
      h('a', { href: url, download: `${name}.stl` }).click();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    }
  }

  useAsPart() {
    const r = this.result;
    if (!r?.shape) return;
    this.app.loadGeneratedPart({ positions: r.shape.positions, index: r.shape.index, name: `${this.part.name || 'part'} (optimized)` });
  }

  showSizing() {
    const r = this.result;
    const { kpis, body } = this.card('Material & size');
    const list = this.ranked();
    const best = list[0];
    const cur = r.base;
    const massNow = this.part.volume * this.panel.app.toMeters ** 3 * r.material.density;
    kpis.replaceChildren(
      kpi('Current design', `FOS ${num(cur.fos)}`, `${r.material.name} · ${num(massNow * 1000)} g`),
      kpi(this.opts.objective === 'cost' ? 'Cheapest' : 'Lightest', best ? best.name.split(' (')[0] : 'none', best ? `${num(best.scale)}× size · ${num(best.mass * 1000)} g · $${num(best.cost)}` : 'nothing meets the target'),
    );
    const table = h('table.opt-table', {},
      h('thead', {}, h('tr', {}, h('th', {}, 'Material'), h('th', {}, 'Size'), h('th', {}, 'Mass'), h('th', {}, 'Cost'), h('th', {}, 'FOS'), h('th', {}))),
      h('tbody', {}, ...[...list, ...r.rows.filter((x) => !x.feasible)].map((row, i) => h('tr', { className: i === 0 && row.feasible ? 'best' : '' },
        h('td', { title: row.name }, row.name.replace(/ \(.*\)$/, '')),
        h('td', {}, row.feasible ? `${num(row.scale)}×` : '–'),
        h('td', {}, row.feasible ? `${num(row.mass * 1000)} g` : '–'),
        h('td', {}, row.feasible ? `$${num(row.cost)}` : '–'),
        h('td', {}, row.feasible ? num(row.fos) : 'no'),
        h('td', {}, row.feasible ? h('button.btn.small', { title: 'Use this material and size', onclick: () => this.apply(row) }, 'Use') : null)))),
    );
    body.replaceChildren(
      table,
      h('p.muted.small', {}, `Safety factor ≥ ${num(this.opts.fos)}${this.opts.maxDisp > 0 ? `, displacement ≤ ${num(this.opts.maxDisp)} ${r.units}` : ''}. Stress scales exactly with size and load type; differences in Poisson's ratio are ignored, so re-run the static study after applying.`),
    );
    this.draw();
  }

  apply(row) {
    this.panel.setMaterial(row.id);
    this.app.onMaterialChange?.();
    if (Math.abs(row.scale - 1) > 1e-3) this.app.applyScale(row.scale);
    this.app.status(`Applied ${row.name} at ${num(row.scale)}× size. Run the static study to check it.`);
  }
}

export function createStudies(panel) {
  const list = [NonlinearStudy, ModalStudy, BucklingStudy, FatigueStudy, DropStudy, DynamicStudy, OptimizeStudy].map((C) => new C(panel));
  return Object.fromEntries(list.map((s) => [s.id, s]));
}
