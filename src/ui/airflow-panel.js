// "Airflow" tab: wind setup, LBM run control, aerodynamic results and flow visualisation.
import * as THREE from 'three';
import { AirflowStudy, windDirection, U_LAT, flowCapacity, planTunnel, FLOW_CELLS, FLOW_BYTES, TURBULENT_RE } from '../cfd/airflow.js';
import { RAMP_STEPS } from '../cfd/flow.js';
import { webgpuAvailable } from '../cfd/lbm-gpu.js';
import { $, $$, h, num, force, stress } from './dom.js';
import { renderLegend } from './legend.js';

const WIND_COLOR = 0x2a78d6;

export class AirflowPanel {
  constructor(app) {
    this.app = app;
    this.yaw = 90;
    this.pitch = 0;
    this.cells = FLOW_CELLS.gpuDefault;
    // grid-size range per engine (filled in once the GPU is detected)
    this.caps = { gpu: null, cpu: { min: FLOW_CELLS.min, max: FLOW_CELLS.cpuMax, default: FLOW_CELLS.cpuDefault } };
    this.lastMlups = 0;
    this.lastEngine = null;
    this.dirty = true;
    this.cp = null;
    this.lastCp = 0;
    this.operation = 0;
    this.building = false;
    this.study = new AirflowStudy(app.viewer, {
      onStatus: (m) => app.status(m),
      onUpdate: () => this.update(),
    });
    this.study.setGroupVisible(false);
    this.wire();
    this.detectGPU();
  }

  async detectGPU() {
    const ok = await webgpuAvailable();
    this.gpu = ok;
    if (ok) this.caps.gpu = await flowCapacity(true);
    this.cells = this.capacity().default;
    this.syncCells();
    $('#gpu-info').textContent = ok
      ? 'WebGPU detected: the flow runs on your graphics card.'
      : 'WebGPU is not available in this browser, so the flow runs on the CPU using a coarser grid. Try a browser with WebGPU enabled for a larger grid.';
  }

  wire() {
    const yaw = $('#yaw-range'), pitch = $('#pitch-range');
    const sync = () => {
      $('#yaw-out').textContent = `${this.yaw}°`;
      $('#pitch-out').textContent = `${this.pitch}°`;
      yaw.value = this.yaw;
      pitch.value = this.pitch;
      for (const b of $$('#wind-presets button')) {
        b.classList.toggle('active', Number(b.dataset.yaw) === this.yaw && Number(b.dataset.pitch) === this.pitch);
      }
      this.updateCellsInfo(); // the tunnel's shape follows the wind direction
    };
    this.syncWind = sync;
    yaw.addEventListener('input', () => { this.yaw = Number(yaw.value); sync(); this.markDirty(); });
    pitch.addEventListener('input', () => { this.pitch = Number(pitch.value); sync(); this.markDirty(); });
    for (const b of $$('#wind-presets button')) {
      b.addEventListener('click', () => { this.yaw = Number(b.dataset.yaw); this.pitch = Number(b.dataset.pitch); sync(); this.markDirty(); });
    }
    sync();
    for (const id of ['#wind-speed', '#air-density', '#engine-select', '#ground-clearance', '#bl-select']) $(id).addEventListener('change', () => this.markDirty());
    $('#chk-ground').addEventListener('change', (e) => {
      $('#ground-controls').hidden = !e.target.checked;
      this.updateCellsInfo();
      this.markDirty();
    });
    $('#chk-autostop').addEventListener('change', (e) => { this.study.autoStop = e.target.checked; });
    // grid size on a log scale: every step of the slider is the same ratio of cells
    $('#flow-cells').addEventListener('input', (e) => {
      const cap = this.capacity();
      const raw = cap.min * Math.pow(cap.max / cap.min, Number(e.target.value) / 1000);
      const p = Math.pow(10, Math.floor(Math.log10(raw)) - 1); // two significant digits
      this.cells = Math.min(cap.max, Math.max(cap.min, Math.round(raw / p) * p));
      this.updateCellsInfo();
      this.markDirty();
    });
    $('#engine-select').addEventListener('change', () => {
      // the CPU worker takes smaller grids: keep the slider within what this engine can do
      const cap = this.capacity();
      this.cells = Math.min(cap.max, Math.max(cap.min, this.cells));
      this.syncCells();
    });
    $('#btn-flow-run').addEventListener('click', () => this.run());
    $('#btn-flow-reset').addEventListener('click', () => this.study.reset());
    const vis = { '#chk-particles': 'particles', '#chk-streamlines': 'streamlines', '#chk-domain': 'domain' };
    for (const [id, key] of Object.entries(vis)) {
      $(id).addEventListener('change', (e) => { this.study.setVisible(key, e.target.checked); this.renderLegends(); });
    }
    $('#chk-cp').addEventListener('change', () => this.applyColoring(true));
    $('#chk-slice').addEventListener('change', (e) => {
      $('#slice-controls').hidden = !e.target.checked;
      this.study.setVisible('slice', e.target.checked);
      this.renderLegends();
    });
    $('#slice-axis').addEventListener('change', (e) => { this.study.sliceAxis = e.target.value; this.study.updateSlice(); });
    $('#slice-qty').addEventListener('change', (e) => { this.study.sliceQuantity = e.target.value; this.study.updateSlice(); this.renderLegends(); });
    const sp = $('#slice-pos');
    sp.addEventListener('input', () => {
      $('#slice-pos-out').textContent = `${sp.value}%`;
      this.study.slicePos = Number(sp.value) / 100;
      this.study.updateSlice();
    });
    $('#btn-wind-load').addEventListener('click', () => this.useAsLoad());
  }

  settings() {
    const speed = Number($('#wind-speed').value);
    const airDensity = Number($('#air-density').value);
    if (!Number.isFinite(speed) || speed <= 0) throw new Error('Wind speed must be a finite positive number.');
    if (!Number.isFinite(airDensity) || airDensity <= 0) throw new Error('Air density must be a finite positive number.');
    const clearance = Number($('#ground-clearance').value);
    if ($('#chk-ground').checked && !(Number.isFinite(clearance) && clearance >= 0)) throw new Error('Ground clearance must be zero or more.');
    return {
      dir: windDirection(this.yaw, this.pitch),
      speed,
      airDensity,
      cells: this.cells,
      engine: $('#engine-select').value,
      toMeters: this.app.toMeters,
      ground: $('#chk-ground').checked ? clearance : null,
      boundaryLayer: $('#bl-select').value,
    };
  }

  usesGPU() {
    return !!this.gpu && !!this.caps.gpu && $('#engine-select').value !== 'cpu';
  }

  capacity() {
    return this.usesGPU() ? this.caps.gpu : this.caps.cpu;
  }

  syncCells() {
    const cap = this.capacity();
    $('#flow-cells').value = Math.round((1000 * Math.log(this.cells / cap.min)) / Math.log(cap.max / cap.min));
    this.updateCellsInfo();
  }

  updateCellsInfo() {
    const cap = this.capacity();
    const count = (n) => (n >= 1e6 ? `${num(n / 1e6, 2)} M` : `${num(n / 1e3, 2)} k`);
    $('#flow-cells-out').textContent = count(this.cells);
    const parts = [];
    let nx = Math.cbrt(this.cells);
    if (this.app.part) {
      try {
        const plan = planTunnel(this.app.part, windDirection(this.yaw, this.pitch), this.cells, { margins: 'app', ground: $('#chk-ground').checked ? Number($('#ground-clearance').value) || 0 : null });
        nx = plan.dims[0];
        parts.push(`Tunnel ${plan.dims.join(' × ')}, cells ${num(plan.h)} ${this.app.units} across.`);
      } catch { /* no finite part */ }
    }
    const gb = (bytes) => num((this.cells * bytes) / 1e9, 2);
    parts.push(this.usesGPU() ? `About ${gb(FLOW_BYTES.gpu)} GB of GPU memory and ${gb(FLOW_BYTES.page)} GB of RAM.` : `About ${gb(FLOW_BYTES.page + 2 * 76)} GB of RAM.`);
    // from the speed measured on the last run: how long until the flow has developed
    const engine = this.usesGPU() ? 'WebGPU' : 'CPU';
    if (this.lastMlups > 0 && this.lastEngine === engine) {
      const perSecond = (this.lastMlups * 1e6) / this.cells;
      const develop = (RAMP_STEPS + (1.5 * nx) / U_LAT) / perSecond;
      parts.push(`At the last run's ${num(this.lastMlups)} MLUPS: about ${num(perSecond)} steps/s, developed flow after about ${num(develop, 2)} s.`);
    }
    parts.push(`Range here: ${count(cap.min)} to ${count(cap.max)} cells.`);
    $('#flow-cells-info').textContent = parts.join(' ');
  }

  markDirty() {
    this.dirty = true;
    this.drawWindArrow();
    this.update();
  }

  /** The part was scaled: the tunnel and its results no longer fit. */
  onPartScaled() {
    this.reset();
  }

  reset(preset = null) {
    this.operation++;
    this.building = false;
    this.study.dispose(false);
    this.dirty = true;
    this.cp = null;
    this.cpRange = null;
    if (preset) {
      this.yaw = preset.yaw;
      this.pitch = preset.pitch;
      $('#wind-speed').value = preset.speed;
      $('#chk-ground').checked = Number.isFinite(preset.ground);
      $('#ground-controls').hidden = !$('#chk-ground').checked;
      if (Number.isFinite(preset.ground)) $('#ground-clearance').value = preset.ground;
      this.syncWind();
    }
    $('#ground-unit').textContent = this.app.units;
    $('#flow-card').hidden = true;
    $('#flow-display').hidden = true;
    this.updateCellsInfo(); // a new part means a new tunnel
    this.updateButtons();
    if (this.app.tab === 'airflow') this.activate();
  }

  updateButtons() {
    const run = $('#btn-flow-run');
    run.disabled = this.building || this.study.initializing;
    if (this.study.running) run.textContent = this.dirty ? 'Apply & restart' : 'Pause';
    else if (this.study.ready && !this.dirty) run.textContent = 'Resume';
    else run.textContent = this.study.ready ? 'Apply & run' : 'Run airflow';
    $('#btn-flow-reset').disabled = !this.study.ready || this.building;
    $('#btn-wind-load').disabled = this.dirty || !this.study.results || this.study.developing || !this.study.surface;
  }

  async run() {
    if (this.building || this.study.initializing) return;
    const part = this.app.part;
    if (!part) return this.app.status('Import a part or open a sample first.', 'error');
    if (this.study.running && !this.dirty) {
      this.study.pause();
      this.updateButtons();
      return;
    }
    if (this.dirty || !this.study.ready) {
      let settings;
      try { settings = this.settings(); }
      catch (err) { return this.app.status(err.message, 'error'); }
      const operation = ++this.operation;
      this.building = true;
      this.updateButtons();
      this.app.busy.show('Building wind tunnel…', () => {
        this.operation++;
        this.building = false;
        this.study.dispose(false);
        this.app.busy.hide();
        this.update();
        this.app.status('Airflow setup cancelled.');
      });
      try {
        await this.study.setup(part, settings);
      } catch (err) {
        if (err.name !== 'AbortError' && operation === this.operation) this.app.status(`Airflow setup failed: ${err.message}`, 'error');
        return;
      } finally {
        if (operation === this.operation) {
          this.building = false;
          this.app.busy.hide();
          this.update();
        }
      }
      if (operation !== this.operation || this.app.part !== part) return;
      this.dirty = false;
      this.cp = null;
      this.study.setVisible('particles', $('#chk-particles').checked);
      this.study.setVisible('domain', $('#chk-domain').checked);
      this.study.show.streamlines = $('#chk-streamlines').checked;
      this.study.show.slice = $('#chk-slice').checked;
    }
    this.study.start();
    $('#flow-card').hidden = false;
    $('#flow-display').hidden = false;
    this.updateButtons();
    this.renderLegends();
  }

  update() {
    this.updateButtons();
    const s = this.study;
    // remember the measured speed for the grid-size estimate
    if (s.mlups > 0 && !s.developing) {
      this.lastMlups = s.mlups;
      this.lastEngine = s.engine;
      if (performance.now() - (this.lastInfo || 0) > 2000) { this.lastInfo = performance.now(); this.updateCellsInfo(); }
    }
    const r = s.results;
    const kpi = (label, value, sub) => h('div.kpi', {}, h('div.k-label', {}, label), h('div.k-value', {}, value), sub ? h('div.k-sub', {}, sub) : null);
    const u = this.app.units;
    // "± x" for a 95% confidence interval, once the averages have one
    const pm = (v, ci, fmt) => (r?.averaged && Number.isFinite(ci) ? `${fmt(v)} ± ${fmt(ci).replace(/^-/, '')}` : fmt(v));
    const ci = (v) => (r?.averaged && Number.isFinite(v) ? `± ${force(Math.abs(v))} · ` : '');
    if (r) {
      const area = r.frontalArea / (this.app.toMeters * this.app.toMeters);
      // the part's own weight (from its material) against the aerodynamic force
      const mat = this.app.structural.material;
      const weight = (this.app.part?.volume ?? 0) * this.app.toMeters ** 3 * mat.density * 9.81;
      const up = r.force?.y ?? 0; // world +Y
      const ratio = weight > 0 ? up / weight : 0;
      const verdict = ratio >= 1 ? h('span.status.bad', {}, 'the wind lifts it') : ratio >= 0.5 ? h('span.status.warn', {}, `lift is ${Math.round(ratio * 100)}% of its weight`) : `lift is ${Math.max(0, Math.round(ratio * 100))}% of its weight`;
      $('#flow-kpis').replaceChildren(
        kpi('Drag force', force(r.drag), `${ci(r.dragCI)}Cd ${pm(r.cd, r.cdCI, (x) => num(x))}`),
        kpi('Lift force', force(r.lift), `${ci(r.liftCI)}Cl ${pm(r.cl, r.clCI, (x) => num(x))}`),
        h('div.kpi', {}, h('div.k-label', {}, `Weight (${mat.name.split(' (')[0]})`), h('div.k-value', {}, force(weight)), h('div.k-sub', {}, verdict)),
        kpi('Frontal area', `${num(area)} ${u}²`, `side force ${pm(r.side, r.sideCI, force)}`),
        kpi('Reynolds number', num(s.reynolds), s.simReynolds < 0.5 * s.reynolds ? `simulated ${num(s.simReynolds)}` : 'simulated at full value'),
      );
      const notes = [];
      notes.push(r.averaged ? 'Forces are time averages with their 95% confidence intervals, from the momentum the air exchanges with the part (pressure and skin friction).' : 'Forces now; their averages start once the flow has developed.');
      notes.push(s.wallModel
        ? `Boundary layer: turbulent (wall model, Re ${num(s.reynoldsLength)} along the part${s.reynoldsLength < TURBULENT_RE ? ', set by hand' : ''}).`
        : `Boundary layer: resolved by the grid (laminar${s.reynoldsLength >= TURBULENT_RE ? ', set by hand; the real one is turbulent' : ''}).`);
      if (s.simReynolds < 0.5 * s.reynolds) notes.push(`The grid holds the flow at a lower Reynolds number (${num(s.simReynolds)}) than real air (${num(s.reynolds)}): expect the drag of rounded shapes to differ.`);
      $('#flow-notes').textContent = notes.join(' ');
    } else {
      $('#flow-kpis').replaceChildren();
      $('#flow-notes').textContent = '';
    }
    const phase = !s.steps ? 'starting' : s.developing ? 'developing flow' : s.converged ? `converged (${s.samples} samples)` : `averaging (${s.samples} samples)`;
    $('#flow-state').textContent = s.ready
      ? `${this.dirty ? 'Settings changed; apply to update results. ' : ''}${s.engine} · ${s.dims.join('×')} cells · step ${s.steps.toLocaleString()} · ${num(s.mlups || 0)} MLUPS · ${phase}.`
      : '';
    if (!s.surface) {
      this.cp = null;
      this.cpRange = null;
      if (this.app.tab === 'airflow') this.app.viewer.setScalars(null);
    }
    if (this.app.tab === 'airflow' && performance.now() - this.lastCp > 900) this.applyColoring();
  }

  applyColoring(force = false) {
    if (this.app.tab !== 'airflow') return;
    const v = this.app.viewer;
    const s = this.study;
    if ($('#chk-cp').checked && s.surface) {
      if (force || performance.now() - this.lastCp > 900 || !this.cp) {
        this.cp = s.surfaceCp();
        this.lastCp = performance.now();
      }
      let min = Infinity, max = -Infinity;
      for (const c of this.cp) if (!Number.isNaN(c)) { min = Math.min(min, c); max = Math.max(max, c); }
      this.cpRange = [Math.max(-3, Math.min(min, -0.2)), Math.min(1.2, Math.max(max, 0.2))];
      v.setScalars(this.cp, { min: this.cpRange[0], max: this.cpRange[1] });
    } else {
      this.cp = null;
      this.cpRange = null;
      v.setScalars(null);
    }
    this.renderLegends();
  }

  renderLegends() {
    if (this.app.tab !== 'airflow') return;
    const s = this.study;
    const specs = [];
    const U = s.opts?.speed || Number($('#wind-speed').value);
    const q = s.q || 0.5 * 1.225 * U * U;
    if ($('#chk-cp').checked && this.cp && this.cpRange) {
      specs.push({
        title: 'Surface pressure (Cp)', sub: `1 Cp = ${stress(q)} at ${num(U)} m/s`,
        min: this.cpRange[0], max: this.cpRange[1], format: (x) => num(x, 2),
      });
    }
    const speedShown = (s.fields && ($('#chk-particles').checked || $('#chk-streamlines').checked)) || ($('#chk-slice').checked && s.sliceQuantity === 'speed');
    if (speedShown && s.ready) specs.push({ title: 'Air speed', sub: 'particles, streamlines, slice', min: 0, max: 1.6 * U, format: (x) => `${num(x)} m/s` });
    if ($('#chk-slice').checked && s.sliceQuantity === 'pressure' && s.ready) {
      specs.push({ title: 'Slice pressure (Cp)', min: -1.2, max: 1, format: (x) => num(x, 2) });
    }
    renderLegend($('#legend'), specs);
  }

  drawWindArrow() {
    const v = this.app.viewer;
    if (this.arrow) {
      v.flowGroup.remove(this.arrow);
      this.arrow.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); });
      this.arrow = null;
    }
    const part = this.app.part;
    if (!part || this.app.tab !== 'airflow') return;
    const dir = windDirection(this.yaw, this.pitch);
    const c = new THREE.Vector3(...part.bbox.min.map((m, i) => (m + part.bbox.max[i]) / 2));
    const diag = part.bbox.diag;
    const tip = c.clone().addScaledVector(dir, -0.6 * diag);
    this.arrow = v.arrow(tip, dir, 0.35 * diag, WIND_COLOR, 0.35 * diag * 0.04);
    v.flowGroup.add(this.arrow);
  }

  activate() {
    this.study.setGroupVisible(true);
    this.drawWindArrow();
    this.applyColoring(true);
    this.renderLegends();
  }

  deactivate() {
    this.study.setGroupVisible(false);
    const v = this.app.viewer;
    if (this.arrow) {
      v.flowGroup.remove(this.arrow);
      this.arrow.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); });
      this.arrow = null;
    }
    v.setScalars(null);
  }

  probe(hit) {
    if (!this.cp || !$('#chk-cp').checked) return null;
    const T = this.app.part.tris, t = hit.tri;
    const w = [hit.bary.x, hit.bary.y, hit.bary.z];
    let s = 0;
    for (let k = 0; k < 3; k++) s += w[k] * this.cp[T[3 * t + k]];
    if (Number.isNaN(s)) return 'no data';
    return `Cp ${num(s, 2)} · ${stress(s * this.study.q)}`;
  }

  useAsLoad() {
    const s = this.study;
    if (this.dirty) return this.app.status('Apply the changed airflow settings before transferring the wind load.', 'warn');
    if (!s.results || !s.surface) return this.app.status('Run the airflow first.', 'error');
    if (s.developing) return this.app.status('The flow is still developing - wait until it says “averaging”, then try again.', 'warn');
    const F = s.triangleForces();
    const net = new THREE.Vector3();
    for (let t = 0; t < F.length; t += 3) net.x += F[t], net.y += F[t + 1], net.z += F[t + 2];
    this.app.structural.addWindLoad(F, net);
    this.app.setTab('structural');
    this.app.status(`Wind load added (${force(net.length())} net). Add a fixture if needed, then run the bend test.`);
  }
}
