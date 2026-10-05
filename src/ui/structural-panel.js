// Structural tab: material, fixtures, loads and the study picker. The linear static study (bend
// test and progressive-damage break test) lives here; the other studies are in studies.js.
import * as THREE from 'three';
import { MATERIALS } from '../core/materials.js';
import { typicalWallThickness } from '../core/shell.js';
import { StructuralModel, FEAJob, validateMaterial } from '../fea/structural.js';
import { patchFrame, sampleTriangles } from '../core/mesh.js';
import { $, $$, h, num, stress, stressFormatter, force, nextFrame } from './dom.js';
import { renderLegend } from './legend.js';
import { LineChart } from './chart.js';
import { createStudies, STUDY_INFO } from './studies.js';
import { patchToJson, patchFromJson } from '../core/psim-patches.js';
import { seriesSpecs } from '../core/psim-series.js';

const FIX_COLOR = 0x1a9f55;
const LOAD_COLOR = 0x9b46d4;
const LOAD_SELECTED = 0xc026d3;
const WIND_COLOR = 0x2a78d6;
const CRACK_COLOR = 0xe0262b;
let uid = 1;

export class StructuralPanel {
  constructor(app) {
    this.app = app;
    this.material = { ...MATERIALS[0] };
    this.fixtures = [];
    this.loads = [];
    this.selected = null;
    this.model = null;
    this.result = null;
    this.brk = null;
    this.display = 'setup';
    this.view = { plot: 'vm', scalePct: 1, level: 'applied', animate: false, bands: false, heat: false, bcs: false, marker: true, breakStress: true };
    this.phase = 0;
    this.runVersion = 0;
    this.preparing = false;
    this.study = 'static';
    this.resFor = {};
    this.models = new Map();
    this.studies = createStudies(this);
    this.initMaterial();
    this.wire();
    this.chart = new LineChart($('#break-chart'), $('#break-tip'), {
      xLabel: 'Crack growth step',
      formatX: (v) => String(Math.round(v)),
      formatY: (v) => force(v),
      onPick: (i) => this.setBreakStep(i),
    });
    app.viewer.onFrame((dt) => this.tick(dt));
  }

  get part() {
    return this.app.part;
  }

  // ---------- setup ----------

  reset(setup = null) {
    this.cancelJob();
    this.stopBreakPlay();
    this.fixtures = [];
    this.loads = [];
    this.selected = null;
    this.model = null;
    this.result = null;
    this.brk = null;
    this.display = 'setup';
    this.stale = false;
    this.models.clear();
    this.resFor = {};
    for (const st of Object.values(this.studies)) st.clear();
    this.app.viewer.setShape?.(null);
    $('#results-card').hidden = true;
    $('#break-card').hidden = true;
    $('#study-card').hidden = true;
    $('#chk-voxels').checked = false;
    if (setup) {
      if (setup.material) this.setMaterial(setup.material);
      for (const f of setup.fixtures || []) this.fixtures.push({ id: uid++, name: f.name, patches: f.patches.filter((p) => p.tris.length) });
      for (const l of setup.loads || []) {
        this.loads.push({ id: uid++, name: l.name, type: l.type, magnitude: l.magnitude, dir: l.dir.slice(), patches: l.patches.filter((p) => p.tris.length), userDir: true });
      }
      this.selected = this.loads[0] || null;
    }
    if (this.part) {
      const n = suggestResolution(this.part, STUDY_INFO[this.study].budget);
      $('#res-range').value = n;
      $('#res-out').textContent = n;
    }
    this.renderLists();
    this.renderStudyOptions();
    this.updateMeshInfo();
    if (this.app.tab === 'structural') this.activate();
  }

  // ---------- study picker ----------

  setStudy(id) {
    if (!STUDY_INFO[id] || id === this.study) return;
    this.cancelJob();
    this.stopBreakPlay();
    this.resFor[this.study] = Number($('#res-range').value);
    this.studies[this.study]?.leave?.();
    this.study = id;
    $('#study-select').value = id;
    if (this.part) {
      const n = this.resFor[id] ?? suggestResolution(this.part, STUDY_INFO[id].budget);
      $('#res-range').value = n;
      $('#res-out').textContent = n;
      this.updateMeshInfo();
    }
    const st = this.studies[id];
    this.display = id === 'static' ? (this.result ? 'results' : this.brk?.steps.length ? 'break' : 'setup') : st?.result ? 'study' : 'setup';
    this.app.viewer.setShape?.(null);
    this.app.viewer.setVoxels('mesh', new Float32Array(0), 1, 0);
    if (this.app.viewer.mesh) this.app.viewer.mesh.visible = true;
    this.renderStudyOptions();
    this.applyDisplay();
  }

  renderStudyOptions() {
    const info = STUDY_INFO[this.study];
    $('#study-desc').textContent = info.desc;
    $('#btn-run').textContent = info.button;
    $('#btn-break').hidden = this.study !== 'static';
    $('#fixtures-card').hidden = !info.fixtures;
    $('#loads-card').hidden = !info.loads;
    $('#results-card').hidden = this.study !== 'static' || !this.result;
    $('#break-card').hidden = this.study !== 'static' || !this.brk;
    $('#study-card').hidden = this.study === 'static' || !this.studies[this.study]?.result;
    const card = $('#study-options');
    if (card.contains(document.activeElement)) document.activeElement.blur();
    const content = this.studies[this.study]?.options();
    card.hidden = !content;
    card.replaceChildren(...(content || []).filter(Boolean));
  }

  runStudy() {
    if (this.study === 'static') return this.run();
    return this.studies[this.study].run();
  }

  initMaterial() {
    const sel = $('#mat-select');
    for (const m of MATERIALS) sel.append(h('option', { value: m.id }, m.name));
    sel.append(h('option', { value: 'custom' }, 'Custom…'));
    sel.value = this.material.id;
    sel.addEventListener('change', () => {
      if (sel.value === 'custom') {
        this.material = { ...this.material, id: 'custom', name: 'Custom' };
      } else this.setMaterial(sel.value);
      this.renderMaterial();
      this.markStale();
    });
    this.renderMaterial();
  }

  setMaterial(id) {
    const m = MATERIALS.find((x) => x.id === id);
    if (m) this.material = { ...m };
    $('#mat-select').value = this.material.id;
    this.renderMaterial();
  }

  renderMaterial() {
    const grid = $('#mat-props');
    // Commit a focused field before replacing its ancestor. Chromium may fire
    // change during replaceChildren, recursively rebuilding a half-removed tree.
    if (grid.contains(document.activeElement)) document.activeElement.blur();
    const m = this.material;
    const field = (key, label, unit, step) =>
      h('label', {}, h('span', {}, label, ' ', h('span.unit', {}, unit)),
        h('input', {
          type: 'number', value: m[key], step, min: key === 'nu' ? -0.99 : 0, max: key === 'nu' ? 0.499 : undefined,
          onchange: (e) => {
            m[key] = Number(e.target.value);
            if (m.id !== 'custom') { m.id = 'custom'; m.name = `Custom (${m.name})`; $('#mat-select').value = 'custom'; }
            $('#mat-summary-name').textContent = m.name;
            this.app.onMaterialChange?.();
            this.markStale();
          },
        }));
    $('#mat-summary-name').textContent = m.name;
    this.app.onMaterialChange?.();
    if (this.studies && this.study !== 'static') this.renderStudyOptions();
    grid.replaceChildren(
      field('E', "Young's modulus", 'GPa', 'any'),
      field('nu', "Poisson's ratio", '', 0.01),
      field('yield', 'Yield strength', 'MPa', 'any'),
      field('uts', 'Tensile strength', 'MPa', 'any'),
      field('density', 'Density', 'kg/m³', 'any'),
      h('label', {}, h('span', {}, 'Failure mode'),
        h('select', {
          onchange: (e) => { m.brittle = e.target.value === 'brittle'; this.markStale(); },
        },
        h('option', { value: 'ductile', selected: !m.brittle }, 'Ductile (von Mises)'),
        h('option', { value: 'brittle', selected: !!m.brittle }, 'Brittle (principal)'))),
    );
  }

  wire() {
    $('#btn-add-fixture').addEventListener('click', () => this.addFixture());
    $('#btn-add-force').addEventListener('click', () => this.addLoad('force'));
    $('#btn-add-pressure').addEventListener('click', () => this.addLoad('pressure'));
    $('#chk-gravity').addEventListener('change', () => this.markStale());
    const res = $('#res-range');
    res.addEventListener('input', () => { $('#res-out').textContent = res.value; });
    res.addEventListener('change', () => {
      this.model = null;
      this.markStale();
      this.updateMeshInfo();
      if ($('#chk-voxels').checked) this.showVoxelPreview();
    });
    $('#chk-voxels').addEventListener('change', () => this.showVoxelPreview());
    $('#btn-run').addEventListener('click', () => this.runStudy());
    $('#study-select').addEventListener('change', (e) => this.setStudy(e.target.value));
    $('#btn-break').addEventListener('click', () => this.runBreak());

    $('#plot-select').addEventListener('change', (e) => { this.view.plot = e.target.value; this.showResults(); });
    const sc = $('#scale-range');
    sc.addEventListener('input', () => { this.view.scalePct = Number(sc.value); this.showResults(false); });
    for (const b of $$('#load-level button')) b.addEventListener('click', () => this.setLevel(b.dataset.level));
    $('#btn-bend-break').addEventListener('click', () => this.bendToBreak());
    $('#chk-animate').addEventListener('change', (e) => { this.view.animate = e.target.checked; this.phase = 0; this.showResults(false); });
    $('#chk-bands').addEventListener('change', (e) => { this.view.bands = e.target.checked; this.refreshDisplay(); });
    $('#chk-heat').addEventListener('change', (e) => { this.view.heat = e.target.checked; this.refreshDisplay(); });
    $('#chk-bcs').addEventListener('change', (e) => { this.view.bcs = e.target.checked; this.drawOverlays(); });
    $('#chk-marker').addEventListener('change', (e) => { this.view.marker = e.target.checked; this.refreshDisplay(); });

    const step = $('#break-step');
    step.addEventListener('input', () => this.setBreakStep(Number(step.value)));
    $('#btn-break-play').addEventListener('click', () => this.toggleBreakPlay());
    $('#chk-break-stress').addEventListener('change', (e) => { this.view.breakStress = e.target.checked; this.showBreakStep(); });
  }

  markStale() {
    if (this.job || this.preparing) this.cancelJob();
    const st = this.studies?.[this.study];
    if (this.result || this.brk || st?.result) {
      this.stale = true;
      if (st) st.stale = true;
      this.app.status('Setup changed - run the study again to update the results.', 'warn');
    }
    this.drawOverlays();
  }

  /** The part was scaled in place by `s` about `pivot`: keep the setup, drop stale results. */
  onPartScaled(s, pivot) {
    this.cancelJob();
    this.stopBreakPlay?.();
    const move = (c) => c.map((v, d) => pivot[d] + (v - pivot[d]) * s);
    for (const f of [...this.fixtures, ...this.loads]) {
      for (const p of f.patches || []) if (p.clip) p.clip = { center: move(p.clip.center), radius: p.clip.radius * s };
      if (f.type === 'wind' && f.forces) {
        // same pressure on s^2 the area
        for (let i = 0; i < f.forces.length; i++) f.forces[i] *= s * s;
        f.magnitude *= s * s;
        if (f.net) f.net = f.net.map((v) => v * s * s);
      }
    }
    this.model = null;
    this.models.clear();
    this.result = null;
    this.brk = null;
    this.anim = null;
    this.display = 'setup';
    for (const st of Object.values(this.studies)) st.clear();
    this.app.viewer.setShape?.(null);
    $('#results-card').hidden = true;
    $('#break-card').hidden = true;
    $('#study-card').hidden = true;
    this.renderLists();
    this.renderStudyOptions();
    this.updateMeshInfo();
  }

  // ---------- fixtures & loads ----------

  requirePart() {
    if (!this.part) {
      this.app.status('Import a part or open a sample first.', 'error');
      return false;
    }
    return true;
  }

  enterSetupView() {
    this.display = 'setup';
    this.stopBreakPlay();
    this.applyDisplay();
  }

  pick(feature, { title, color, isNew }) {
    this.enterSetupView();
    const before = feature.patches.length;
    const beforeDir = feature.dir?.slice();
    this.app.picker.start({
      title,
      color,
      onPatch: (p) => {
        if (!p.tris.length) return;
        if (!p.clip && feature.patches.some((old) => !old.clip && old.tris.length === p.tris.length && old.tris.every((t, i) => t === p.tris[i]))) return;
        feature.patches.push(p);
        if (feature.type === 'force' && !feature.userDir) {
          const fr = this.frame(feature);
          if (Math.hypot(...fr.normal) > 1e-9) feature.dir = fr.normal.map((v) => -v);
        }
        this.markStale();
        this.renderLists();
      },
      onDone: (commit) => {
        if (!commit) {
          feature.patches.length = before;
          if (beforeDir) feature.dir = beforeDir;
        }
        if (!feature.patches.length && isNew) {
          this.fixtures = this.fixtures.filter((f) => f !== feature);
          this.loads = this.loads.filter((l) => l !== feature);
          if (this.selected === feature) this.selected = this.loads[0] || null;
        }
        this.renderLists();
        this.drawOverlays();
      },
    });
  }

  addFixture() {
    if (!this.requirePart()) return;
    const fx = { id: uid++, name: `Fixed support ${this.fixtures.length + 1}`, patches: [] };
    this.fixtures.push(fx);
    this.renderLists();
    this.pick(fx, { title: 'Fixed support', color: FIX_COLOR, isNew: true });
  }

  addLoad(type) {
    if (!this.requirePart()) return;
    const n = this.loads.filter((l) => l.type === type).length + 1;
    const ld = type === 'force'
      ? { id: uid++, name: `Force ${n}`, type, magnitude: 500, dir: [0, -1, 0], patches: [], userDir: false }
      : { id: uid++, name: `Pressure ${n}`, type, magnitude: 1, dir: [0, -1, 0], patches: [] };
    this.loads.push(ld);
    this.selected = ld;
    this.renderLists();
    this.pick(ld, { title: type === 'force' ? 'Force' : 'Pressure', color: LOAD_SELECTED, isNew: true });
  }

  addWindLoad(forces, net) {
    this.loads = this.loads.filter((l) => l.type !== 'wind');
    const ld = { id: uid++, name: 'Wind load (airflow)', type: 'wind', forces, net: [net.x, net.y, net.z], magnitude: net.length(), patches: [], dir: net.clone().normalize().toArray() };
    this.loads.push(ld);
    this.selected = ld;
    this.markStale();
    this.renderLists();
  }

  frame(feature) {
    const tris = allTris(feature);
    const clip = feature.patches.length === 1 ? feature.patches[0].clip : null;
    return patchFrame(this.part, tris, clip);
  }

  renderLists() {
    const fl = $('#fixture-list');
    fl.replaceChildren(
      ...this.fixtures.map((f) =>
        h('li', {},
          h('span.swatch', { style: { background: 'var(--fixture)' } }),
          h('span.name', {}, f.name),
          h('span.meta', {}, `${f.patches.length} area${f.patches.length === 1 ? '' : 's'}`),
          h('button.x', { title: 'Add more areas', onclick: (e) => { e.stopPropagation(); this.pick(f, { title: f.name, color: FIX_COLOR }); } }, '+'),
          h('button.x', { title: 'Delete', onclick: (e) => { e.stopPropagation(); this.fixtures = this.fixtures.filter((x) => x !== f); this.markStale(); this.renderLists(); } }, '×'),
        )),
    );
    const ll = $('#load-list');
    ll.replaceChildren(
      ...this.loads.map((l) =>
        h('li', { className: l === this.selected ? 'selected' : '', onclick: () => { this.selected = l; this.renderLists(); this.drawOverlays(); } },
          h('span.swatch', { style: { background: l.type === 'wind' ? 'var(--accent)' : 'var(--load)' } }),
          h('span.name', {}, l.name),
          h('span.meta', {}, l.type === 'pressure' ? `${num(l.magnitude)} MPa` : force(l.magnitude)),
          h('button.x', { title: 'Delete', onclick: (e) => { e.stopPropagation(); this.loads = this.loads.filter((x) => x !== l); if (this.selected === l) this.selected = this.loads[0] || null; this.markStale(); this.renderLists(); } }, '×'),
        )),
    );
    this.renderLoadEditor();
  }

  renderLoadEditor() {
    const ed = $('#load-editor');
    if (ed.contains(document.activeElement)) document.activeElement.blur();
    const l = this.selected;
    if (!l || !this.loads.includes(l)) { ed.hidden = true; return; }
    ed.hidden = false;
    const setDir = (d) => {
      const len = Math.hypot(...d);
      if (!d.every(Number.isFinite) || !(len > 1e-12)) {
        this.app.status('Choose a nonzero force direction.', 'error');
        this.renderLoadEditor();
        return;
      }
      l.dir = d.map((v) => v / len);
      l.userDir = true;
      this.markStale();
      this.renderLoadEditor();
    };
    const nrm = () => this.frame(l).normal;
    const vecInputs = [0, 1, 2].map((i) =>
      h('input', {
        type: 'number', step: 0.1, value: Number(l.dir[i].toFixed(3)), title: 'XYZ'[i],
        onchange: () => setDir(vecInputs.map((x) => Number(x.value))),
      }));
    const nameInput = h('input', { type: 'text', value: l.name, onchange: (e) => { l.name = e.target.value || l.name; this.renderLists(); } });
    nameInput.style.cssText = 'width:100%;height:28px;padding:0 8px;border:1px solid var(--border-strong);border-radius:7px;background:var(--panel)';
    const parts = [h('div.lbl', {}, 'Name'), nameInput];
    if (l.type === 'wind') {
      parts.push(h('p.muted.small', { style: { marginTop: '8px' } },
        `Surface pressure from the airflow study. Net aerodynamic force ${force(l.magnitude)}.`));
    } else {
      parts.push(
        h('div.lbl', {}, l.type === 'force' ? 'Total force' : 'Pressure'),
        h('div.mag', {},
          h('input', {
            type: 'number', value: l.magnitude, step: 'any', min: 0,
            onchange: (e) => { l.magnitude = Number(e.target.value); this.markStale(); this.renderLists(); },
          }),
          h('span.muted', {}, l.type === 'force' ? 'N' : 'MPa')),
      );
    }
    if (l.type === 'force') {
      const b = (label, d, title) => h('button.btn.small', { title, onclick: () => setDir(typeof d === 'function' ? d() : d) }, label);
      parts.push(
        h('div.lbl', {}, 'Direction'),
        h('div.dir-grid', {},
          b('Push ⊥', () => nrm().map((v) => -v), 'Into the surface'),
          b('Pull ⊥', () => nrm(), 'Out of the surface'),
          b('Reverse', () => l.dir.map((v) => -v)),
          h('button.btn.small', { onclick: () => this.pick(l, { title: l.name, color: LOAD_SELECTED }) }, '+ Area'),
          b('+X', [1, 0, 0]), b('−X', [-1, 0, 0]), b('+Y', [0, 1, 0]), b('−Y', [0, -1, 0]),
          b('+Z', [0, 0, 1]), b('−Z', [0, 0, -1]),
        ),
        h('div.vec', {}, ...vecInputs),
        h('p.muted.small', { style: { margin: '6px 0 0' } }, 'Tip: drag the round handle on the arrow in the view to aim the force.'),
      );
    } else if (l.type === 'pressure') {
      parts.push(
        h('p.muted.small', { style: { margin: '6px 0 0' } }, 'Acts normal to the selected faces; positive pushes into the part.'),
        h('button.btn.small', { style: { marginTop: '6px' }, onclick: () => this.pick(l, { title: l.name, color: LOAD_SELECTED }) }, '+ Area'),
      );
    }
    ed.replaceChildren(...parts);
    this.drawOverlays();
  }

  showingOverlays() {
    return this.app.tab === 'structural' && (this.display === 'setup' || this.view.bcs || this.app.picker.active);
  }

  /** Setup items the current study uses (drop tests have neither fixtures nor loads). */
  get activeFixtures() {
    return STUDY_INFO[this.study].fixtures ? this.fixtures : [];
  }

  get activeLoads() {
    return STUDY_INFO[this.study].loads ? this.loads : [];
  }

  drawOverlays() {
    const v = this.app.viewer;
    v.clearOverlays();
    const part = this.part;
    if (!part || !this.showingOverlays()) return;
    const diag = part.bbox.diag;
    const spread = (feature, n) => {
      const tris = allTris(feature);
      if (!tris.length) return [];
      let area = 0;
      for (const t of tris) area += part.triArea[t];
      const pts = [];
      for (const p of feature.patches) {
        const s = sampleTriangles(part, p.tris, Math.sqrt(area / (n * 2)), p.clip);
        for (let i = 0; i < s.weights.length; i++) pts.push({ p: new THREE.Vector3().fromArray(s.points, 3 * i), tri: s.tris[i] });
      }
      if (pts.length <= n) return pts;
      const out = [];
      for (let i = 0; i < n; i++) out.push(pts[Math.floor(((i + 0.5) * pts.length) / n)]);
      return out;
    };
    for (const f of this.activeFixtures) {
      const tris = allTris(f);
      if (!tris.length) continue;
      v.overlayGroup.add(v.patchMesh(tris, FIX_COLOR, 0.5));
      for (const s of spread(f, 6)) {
        v.overlayGroup.add(v.anchor(s.p, Array.from(part.triNormal.subarray(3 * s.tri, 3 * s.tri + 3)), diag * 0.03, FIX_COLOR));
      }
    }
    for (const l of this.activeLoads) {
      const sel = l === this.selected;
      const color = sel ? LOAD_SELECTED : LOAD_COLOR;
      if (l.type === 'wind') {
        const c = new THREE.Vector3(...part.bbox.min.map((m, i) => (m + part.bbox.max[i]) / 2));
        const d = new THREE.Vector3(...l.dir);
        v.overlayGroup.add(v.arrow(c.clone().addScaledVector(d, -diag * 0.05), d, diag * 0.25, WIND_COLOR));
        continue;
      }
      const tris = allTris(l);
      if (!tris.length) continue;
      v.overlayGroup.add(v.patchMesh(tris, color, sel ? 0.55 : 0.4));
      let area = 0;
      for (const t of tris) area += part.triArea[t];
      // about one arrow per 6% of the part size across the patch, fewer when the main arrow is shown
      const n = Math.max(1, Math.min(7, Math.round(Math.sqrt(area) / (0.06 * diag))));
      const pts = sel && l.type === 'force' && n <= 2 ? [] : spread(l, n);
      for (const s of pts) {
        const d = l.type === 'pressure'
          ? new THREE.Vector3().fromArray(part.triNormal, 3 * s.tri).negate()
          : new THREE.Vector3(...l.dir);
        v.overlayGroup.add(v.arrow(s.p, d, diag * 0.07, color));
      }
      if (sel && l.type === 'force') {
        const fr = this.frame(l);
        const tip = new THREE.Vector3(...fr.center);
        const dir = new THREE.Vector3(...l.dir).normalize();
        const len = diag * 0.18;
        const main = v.arrow(tip, dir, len, color, len * 0.03);
        v.overlayGroup.add(main);
        const handle = new THREE.Mesh(
          new THREE.SphereGeometry(len * 0.075, 20, 14),
          new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: color, emissiveIntensity: 0.6 }),
        );
        handle.position.copy(tip).addScaledVector(dir, -len);
        v.overlayGroup.add(handle);
        v.addDraggable(handle, tip, (p) => {
          const d = tip.clone().sub(p);
          if (d.lengthSq() < 1e-12) return;
          d.normalize();
          l.dir = d.toArray();
          l.userDir = true;
          main.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d);
          handle.position.copy(tip).addScaledVector(d, -len);
        }, () => { this.markStale(); this.renderLoadEditor(); });
      }
    }
  }

  // ---------- model & runs ----------

  ensureModel() {
    this.model = this.modelFor(Number($('#res-range').value));
    return this.model;
  }

  /** Voxel model of the current part at a resolution (the last few are cached). */
  modelFor(res) {
    const key = res;
    let m = this.models.get(key);
    if (!m || m.part !== this.part) {
      m = new StructuralModel(this.part, res);
      this.models.set(key, m);
      while (this.models.size > 3) this.models.delete(this.models.keys().next().value);
    }
    return m;
  }

  updateMeshInfo() {
    const el = $('#mesh-info');
    if (!this.part) { el.textContent = ''; return; }
    const res = Number($('#res-range').value);
    const hv = Math.max(...this.part.bbox.size) / res;
    const wall = typicalWallThickness(this.part);
    el.textContent = `Voxel size ≈ ${num(hv)} ${this.app.units}.` + (Number.isFinite(wall)
      ? ` Walls ≈ ${num(wall)} ${this.app.units} thick (${num(wall / hv)} voxels across)${wall < 1.5 * hv ? '; thinner walls are kept as connected layers' : ''}.`
      : '');
    if (this.model && this.model.resolution === res && this.model.part === this.part) {
      el.textContent += ` ${this.model.voxelCount.toLocaleString()} voxels.`;
    }
  }

  async showVoxelPreview() {
    if (this.job || this.preparing) return;
    const v = this.app.viewer;
    v.setVoxels('mesh', new Float32Array(0), 1, 0);
    if (!$('#chk-voxels').checked || !this.part) {
      if (v.mesh) { v.mesh.visible = true; v.edges.visible = v.showEdges !== false; }
      return;
    }
    const part = this.part, version = this.runVersion;
    this.app.busy.show('Voxelizing…');
    await nextFrame();
    if (version !== this.runVersion || part !== this.part) return;
    try {
      if (!$('#chk-voxels').checked || this.app.tab !== 'structural') return;
      const m = this.ensureModel();
      v.setVoxels('mesh', m.voxelCenters((e) => m.isSurfaceVoxel(e)), m.h * 0.94, 0x86a8d8);
      v.mesh.visible = false;
      v.edges.visible = false;
      this.updateMeshInfo();
    } catch (err) {
      $('#chk-voxels').checked = false;
      this.app.status(`Mesh preview failed: ${err.message}`, 'error');
    } finally {
      this.app.busy.hide();
    }
  }

  checkSetup(study = 'static', { requireFixtures = true, requireLoads = true } = {}) {
    if (!this.requirePart()) return false;
    try { validateMaterial(this.material); }
    catch (err) { this.app.status(err.message, 'error'); return false; }
    const info = STUDY_INFO[study] || STUDY_INFO.static;
    if (requireFixtures && info.fixtures && !info.fixturesOptional && !this.fixtures.some((f) => f.patches.length)) {
      this.app.status('Add at least one fixed support (click “+ Fixed support”, then click faces).', 'error');
      return false;
    }
    if (requireLoads && info.loads && !this.loads.length && !$('#chk-gravity').checked) {
      this.app.status('Add a force or pressure load (or enable self-weight).', 'error');
      return false;
    }
    return true;
  }

  async prepare(version, { requireFixtures = true, requireLoads = true } = {}) {
    this.preparing = true;
    this.app.picker.finish(true);
    this.app.busy.show('Voxelizing part…', () => this.cancelJob());
    await nextFrame();
    if (version !== this.runVersion) throw Object.assign(new Error('Cancelled'), { cancelled: true });
    const model = this.ensureModel();
    this.updateMeshInfo();
    if (!model.voxelCount) throw new Error('The part produced no voxels. Increase the mesh resolution.');
    const info = STUDY_INFO[this.study] || STUDY_INFO.static;
    const asm = model.assemble({
      fixtures: info.fixtures ? this.fixtures : [], loads: info.loads ? this.loads : [], gravity: info.loads && $('#chk-gravity').checked,
      material: this.material, toMeters: this.app.toMeters,
    });
    if (requireFixtures && info.fixtures && !info.fixturesOptional && !asm.fixedNodes) throw new Error('No voxels found under the fixtures. Try a higher mesh resolution.');
    if (requireLoads && info.loads && !asm.f.some((value, i) => value !== 0 && !asm.bc[i])) {
      throw new Error('No nonzero load reaches a free node. Set a load magnitude and select an area away from the fixed support.');
    }
    for (const w of asm.warnings) this.app.status(w, 'warn');
    const m = this.material;
    const msg = {
      dims: model.dims, density: model.density, nu: m.nu, E: m.E * 1e9, h: model.h * this.app.toMeters,
      bc: asm.bc, f: asm.f, engine: $('#fea-engine').value,
      rho: m.density, toMeters: this.app.toMeters, map: model.mapping(),
    };
    this.preparing = false;
    return { model, asm, msg, material: { ...m }, toMeters: this.app.toMeters, units: this.app.units };
  }

  cancelJob() {
    this.runVersion++;
    const active = this.job || this.preparing;
    this.preparing = false;
    if (this.job) {
      this.job.cancel();
      this.job = null;
    }
    if (active) {
      this.app.busy.hide();
      if (this.brk && !this.brk.done) {
        this.brk.done = true;
        this.brk.reason = 'stopped';
        this.renderBreakKPIs();
      }
      this.app.status('Simulation cancelled.', 'warn');
    }
  }

  async run() {
    if (!this.checkSetup()) return;
    this.cancelJob();
    this.stopBreakPlay();
    const version = this.runVersion;
    let prep;
    try {
      prep = await this.prepare(version);
    } catch (err) {
      if (version !== this.runVersion) return;
      this.preparing = false;
      this.app.busy.hide();
      return this.app.status(err.message, 'error');
    }
    if (version !== this.runVersion) return;
    const { model, asm, msg } = prep;
    const t0 = performance.now();
    let res;
    try {
      this.job = new FEAJob({ type: 'solve', ...msg }, {
        onProgress: (d) => {
          if (version === this.runVersion) this.app.busy.progress(Math.min(1, Math.log10(Math.max(d.res, 1e-7)) / -6), `${d.stage} · iteration ${d.it}`);
        },
      });
      this.app.busy.show(`Solving ${model.voxelCount.toLocaleString()} voxels…`, () => this.cancelJob());
      res = await this.job.promise;
    } catch (err) {
      if (version !== this.runVersion) return;
      this.app.busy.hide();
      this.job = null;
      return this.app.status(err.cancelled ? 'Cancelled.' : `Solver error: ${err.message}`, err.cancelled ? '' : 'error');
    }
    if (version !== this.runVersion) return;
    this.job = null;
    this.app.busy.hide();
    this.result = this.mapResult(res, asm, prep);
    this.stale = false;
    this.display = 'results';
    $('#results-card').hidden = false;
    $('#scale-range').value = this.view.scalePct = 1;
    this.setLevel('applied', false);
    const notes = [];
    if (res.removed) notes.push(`${res.removed} voxels not connected to a fixture were ignored`);
    if (res.lostLoad > 0.01) notes.push(`${Math.round(res.lostLoad * 100)}% of the load was on unsupported bits`);
    if (!res.converged) notes.push('solver did not fully converge');
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    this.app.status(
      (this.result.unreliable ? 'Results not reliable - see the note in Results. ' : '') +
        `Solved ${res.voxels.toLocaleString()} voxels on the ${res.engine || 'CPU'} in ${secs} s (${res.iterations} multigrid-CG iterations).` +
          (res.gpuNote ? ` The GPU was not used: ${res.gpuNote}.` : '') + (notes.length ? ` Note: ${notes.join('; ')}.` : ''),
      this.result.unreliable ? 'error' : notes.length ? 'warn' : '',
    );
    this.applyDisplay();
    $('#results-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    this.app.runDone?.('static', this.result);
  }

  mapResult(res, asm, prep) {
    const m = prep.model, part = m.part, toM = prep.toMeters;
    const W = m.vertexWeights(res.activeNode);
    const vm = m.interpolate(W, res.nodeVM), p1 = m.interpolate(W, res.nodeP1), p3 = m.interpolate(W, res.nodeP3);
    const u = m.interpolate(W, res.u, 3);
    for (let i = 0; i < u.length; i++) u[i] /= toM;
    const nV = part.nVert;
    const dmag = new Float32Array(nV), fos = new Float32Array(nV);
    const mat = prep.material;
    const ys = mat.yield * 1e6, uts = mat.uts * 1e6;
    let maxVM = 0, maxDisp = 0, maxRatio = 0, weakest = 0, minP1 = Infinity, maxP1 = -Infinity, minP3 = Infinity, maxP3 = -Infinity;
    for (let v = 0; v < nV; v++) {
      dmag[v] = Math.hypot(u[3 * v], u[3 * v + 1], u[3 * v + 2]);
      if (Number.isNaN(vm[v])) { fos[v] = NaN; continue; }
      maxVM = Math.max(maxVM, vm[v]);
      maxDisp = Math.max(maxDisp, dmag[v]);
      minP1 = Math.min(minP1, p1[v]); maxP1 = Math.max(maxP1, p1[v]);
      minP3 = Math.min(minP3, p3[v]); maxP3 = Math.max(maxP3, p3[v]);
      const ratio = mat.brittle ? Math.max(p1[v], 0) / uts : vm[v] / ys;
      fos[v] = ratio > 0 ? 1 / ratio : 1e3;
      if (ratio > maxRatio) { maxRatio = ratio; weakest = v; }
    }
    const diag = part.bbox.diag;
    // exaggeration that makes the largest displacement ~5% of the part size (never above x1000)
    const autoScale = maxDisp > 0 ? Math.min(1000, Math.max(1, (0.05 * diag) / maxDisp)) : 1;
    const solved = res.voxels / Math.max(1, res.voxels + res.removed);
    return {
      vm, p1, p3, u, dmag, fos, maxVM, maxDisp, minP1, maxP1, minP3, maxP3,
      minFos: maxRatio > 0 ? 1 / maxRatio : Infinity, weakest, lambda: maxRatio > 0 ? 1 / maxRatio : Infinity,
      // load multiples (linear-elastic): first yield, and the tensile strength reached (breaking)
      lamYield: maxRatio > 0 ? 1 / maxRatio : Infinity,
      lamBreak: mat.brittle ? (maxRatio > 0 ? 1 / maxRatio : Infinity) : maxVM > 0 ? uts / maxVM : Infinity,
      totalF: Math.hypot(...asm.total), reaction: res.reaction, material: mat, autoScale,
      iterations: res.iterations, voxels: res.voxels,
      units: prep.units, incomplete: !res.converged || res.lostLoad > 0.01 || res.removed > 0,
      // most of the part missing from the solved model: do not present stresses as a result
      unreliable: !res.converged || res.lostLoad > 0.05 || solved < 0.85,
      solvedShare: solved, lostLoad: res.lostLoad, converged: res.converged,
      thinVoxels: prep.model.thinVoxels, voxelSize: prep.model.h, resolution: prep.model.resolution, wallThickness: prep.model.wallThickness,
    };
  }

  // ---------- display ----------

  activate() {
    this.applyDisplay();
  }

  deactivate() {
    const v = this.app.viewer;
    this.stopBreakPlay();
    v.clearOverlays();
    v.clearMarker();
    v.setVoxels('cracks', new Float32Array(0), 1, 0);
    v.setVoxels('detached', new Float32Array(0), 1, 0);
    v.setVoxels('mesh', new Float32Array(0), 1, 0);
    if (v.mesh) { v.mesh.visible = true; v.edges.visible = v.showEdges !== false; }
    v.setDeformation(null);
    v.setScalars(null);
    v.setShape?.(null);
  }

  refreshDisplay() {
    if (this.display === 'results') this.showResults();
    else if (this.display === 'break') this.showBreakStep();
    else if (this.display === 'study') this.studies[this.study]?.show();
  }

  applyDisplay() {
    if (this.app.tab !== 'structural') return;
    this.renderAlert();
    const v = this.app.viewer;
    const st = this.study === 'static' ? null : this.studies[this.study];
    $('#study-card').hidden = !st?.result;
    if (st && this.display === 'study' && st.result) {
      v.setVoxels('cracks', new Float32Array(0), 1, 0);
      v.setVoxels('detached', new Float32Array(0), 1, 0);
      st.show();
    } else if (!st && this.display === 'results' && this.result) this.showResults();
    else if (!st && this.display === 'break' && this.brk?.steps.length) this.showBreakStep();
    else {
      this.display = 'setup';
      v.setShape?.(null);
      v.setVoxels('cracks', new Float32Array(0), 1, 0);
      v.setVoxels('detached', new Float32Array(0), 1, 0);
      v.setDeformation(null);
      v.setScalars(null);
      v.clearMarker();
      renderLegend($('#legend'), []);
      this.drawOverlays();
      if ($('#chk-voxels').checked) this.showVoxelPreview();
    }
  }

  /**
   * What to colour and how. `lam` is the load multiple shown: stresses and displacements scale
   * with it (linear elasticity), so the legend shows true values while the stored per-vertex
   * values stay at the applied load (`scaled: true` tells the caller to rescale the range).
   */
  plotSpec(lam = 1) {
    const r = this.result, units = r.units, mat = r.material;
    const cmap = this.view.heat ? 'heat' : 'rainbow';
    const bands = this.view.bands ? 12 : 0;
    const beyond = lam !== 1 || !!this.anim; // yield/break views and the bend animation
    const uts = mat.uts * 1e6;
    switch (this.view.plot) {
      case 'disp': {
        const max = (r.maxDisp || 1) * (this.anim ? r.lamBreak : lam);
        return { values: r.dmag, scaled: true, legend: { title: 'Displacement', min: 0, max, format: (x) => `${num(x)} ${units}` }, fmt: (x) => `${num(x * lam)} ${units}`, cmap, bands };
      }
      case 'fos': {
        const max = Math.min(10, Math.max(2, Math.ceil((r.minFos / lam) * 3)));
        return {
          values: r.fos.map((f) => (Number.isNaN(f) ? NaN : Math.min(f / lam, max))),
          legend: { title: 'Factor of safety', min: 0, max, reverse: true, format: (x) => num(x, 2), markers: [{ value: 1, label: 'FOS = 1' }] },
          fmt: (x) => (x >= max ? `> ${max}` : num(x, 3)), cmap, bands, reverse: true,
        };
      }
      case 'p1': {
        const hi = beyond ? Math.max(r.maxP1 * lam, uts) : Math.max(r.maxP1, 1);
        const f = stressFormatter(Math.max(Math.abs(r.minP1 * lam), Math.abs(hi)));
        return { values: r.p1, scaled: true, legend: { title: 'Max principal stress', min: Math.min(0, r.minP1 * lam), max: hi, format: f, markers: [{ value: uts, label: `UTS ${num(mat.uts)}` }] }, fmt: (x) => f(x * lam), cmap, bands };
      }
      case 'p3': {
        const f = stressFormatter(Math.max(Math.abs(r.minP3), Math.abs(r.maxP3)) * lam);
        return { values: r.p3, scaled: true, legend: { title: 'Min principal stress', min: Math.min(r.minP3, -1) * lam, max: Math.max(0, r.maxP3) * lam, format: f }, fmt: (x) => f(x * lam), cmap, bands };
      }
      default: {
        // beyond the applied load, keep the scale fixed at the tensile strength so red means breaking
        const hi = beyond ? Math.max(r.maxVM * lam, uts) : r.maxVM || 1;
        const f = stressFormatter(hi);
        const markers = [{ value: mat.yield * 1e6, label: `Yield ${num(mat.yield)}` }];
        if (beyond && !mat.brittle) markers.push({ value: uts, label: `UTS ${num(mat.uts)}` });
        return { values: r.vm, scaled: true, legend: { title: 'von Mises stress', min: 0, max: hi, format: f, markers }, fmt: (x) => f(x * lam), cmap, bands };
      }
    }
  }

  /** Slider 0..100 -> deformation scale: 0 = undeformed, 1 = true scale, then up to 2x the auto exaggeration. */
  deformationScale(auto) {
    const v = this.view.scalePct;
    if (!v) return 0;
    if (v <= 1) return 1;
    const top = Math.max(1, Math.min(2000, 2 * auto));
    return Math.pow(top, (v - 1) / 99);
  }

  /** Load multiple being shown: 1 = the applied loads; first yield / breaking load (linear-elastic). */
  loadFactor() {
    const r = this.result;
    if (!r) return 1;
    if (this.anim) return this.anim.lambda;
    return this.view.level === 'yield' ? r.lamYield : this.view.level === 'break' ? r.lamBreak : 1;
  }

  setLevel(level, refresh = true) {
    this.view.level = level;
    this.anim = null;
    for (const b of $$('#load-level button')) b.classList.toggle('active', b.dataset.level === level);
    // showing the real bend: make sure the shape is drawn (at true scale if it was off)
    if (level !== 'applied' && !this.view.scalePct) $('#scale-range').value = this.view.scalePct = 1;
    if (refresh) this.showResults();
  }

  /** Ramp the load from zero to the breaking load, drawing the true bend as it goes. */
  bendToBreak() {
    const r = this.result;
    if (!r || r.unreliable || !Number.isFinite(r.lamBreak)) return;
    if (!this.view.scalePct) $('#scale-range').value = this.view.scalePct = 1;
    this.view.animate = false;
    $('#chk-animate').checked = false;
    this.anim = { t: 0, dur: 3.5, lambda: 0, lastLegend: 0 };
    this.app.viewer.clearMarker();
  }

  currentScale() {
    if (!this.result || this.result.unreliable) return 0;
    return this.deformationScale(this.result.autoScale);
  }

  showResults(full = true) {
    const r = this.result;
    if (!r || this.app.tab !== 'structural') return;
    const v = this.app.viewer;
    const lam = this.loadFactor();
    const spec = this.plotSpec(lam);
    const scale = this.currentScale();
    $('#scale-out').textContent = scale === 0 ? 'off' : scale < 1.05 ? 'true scale' : `×${num(scale)}`;
    this.renderAlert();
    this.renderLevelNote();
    if (!this.view.animate) v.setDeformation(r.u, scale * lam);
    if (full) {
      v.setVoxels('cracks', new Float32Array(0), 1, 0);
      v.setVoxels('detached', new Float32Array(0), 1, 0);
      if (v.mesh) { v.mesh.visible = true; v.edges.visible = v.showEdges !== false; }
      v.setVoxels('mesh', new Float32Array(0), 1, 0);
      // values are stored at the applied load; showing another load multiple rescales the range
      const k = spec.scaled ? lam : 1;
      v.setScalars(spec.values, { min: spec.legend.min / k, max: spec.legend.max / k, bands: spec.bands, reverse: spec.reverse, colormap: spec.cmap });
      renderLegend($('#legend'), {
        ...spec.legend, bands: spec.bands, colormap: spec.cmap,
        sub: `${r.material.name} · ${this.levelLabel(lam)} · ${scale ? (scale < 1.05 ? 'true-scale shape' : `deformation ×${num(scale)}`) : 'undeformed shape'}`,
      });
      this.drawOverlays();
      this.renderKPIs();
    }
    this.placeMarker(scale * lam);
  }

  levelLabel(lam) {
    const r = this.result;
    const load = r.totalF > 1e-9 ? force(lam * r.totalF) : `${num(lam)} × loads`;
    if (this.anim) return `load ${load}`;
    if (this.view.level === 'yield') return `at first yield (${load})`;
    if (this.view.level === 'break') return `at breaking load (${load})`;
    return 'at applied load';
  }

  renderLevelNote() {
    const el = $('#load-level-note');
    const r = this.result;
    if (!r || r.unreliable || !Number.isFinite(r.lamBreak)) { el.textContent = ''; return; }
    const load = r.totalF > 1e-9 ? force(r.lamBreak * r.totalF) : `${num(r.lamBreak)} × the loads`;
    const bend = `${num(r.maxDisp * r.lamBreak)} ${r.units}`;
    const subtle = r.maxDisp * r.lamBreak < 0.02 * this.part.bbox.diag
      ? ' At true scale that is hard to see; slide "Deformed shape" right to exaggerate it.'
      : '';
    el.textContent = (r.material.brittle
      ? `It breaks at about ${load}, after bending ${bend} (brittle: it cracks at the marker with little warning).`
      : `It reaches its tensile strength at about ${load}, after bending ${bend} elastically. Ductile metals bend further (permanently) before they tear, so treat this as the point where it is badly bent or failing.`) + subtle;
  }

  placeMarker(scale) {
    const v = this.app.viewer;
    const r = this.result;
    if (!r || r.unreliable || !this.view.marker || !Number.isFinite(r.lambda)) { v.clearMarker(); return; }
    const i = r.weakest, V = this.part.vertices;
    const p = new THREE.Vector3(V[3 * i] + scale * r.u[3 * i], V[3 * i + 1] + scale * r.u[3 * i + 1], V[3 * i + 2] + scale * r.u[3 * i + 2]);
    const level = this.anim ? 'anim' : this.view.level;
    const label = level === 'break' ? 'Breaks here' : level === 'yield' ? 'Yields here' : r.minFos < 1 ? 'Strength exceeded here' : 'Estimated weakest point';
    if (v.marker && v.marker.label === label) v.moveMarker(p);
    else v.setMarker(p, label);
  }

  /** Explain results that should not be trusted, with a one-click fix when a finer mesh helps. */
  renderAlert() {
    const el = $('#results-alert');
    const r = this.result;
    if (!r || this.display !== 'results' || !(r.unreliable || r.thinVoxels)) { el.hidden = true; return; }
    const units = r.units;
    const finer = Math.min(480, Math.max(r.resolution + 8, Math.round(r.resolution * 1.6 / 4) * 4));
    const parts = [];
    if (r.unreliable) {
      const why = !r.converged
        ? 'the solver could not balance the loads (usually pieces that are not connected to a fixture, or a fixture too small to stop the part pivoting)'
        : `only ${Math.round(r.solvedShare * 100)}% of the part was connected to the fixtures in the voxel model${r.lostLoad > 0.05 ? ` and ${Math.round(r.lostLoad * 100)}% of the load fell on the unconnected bits` : ''}`;
      const thinWalls = r.wallThickness < 2 * r.voxelSize;
      const cause = thinWalls
        ? `The walls (≈ ${num(r.wallThickness)} ${units}) are thin compared with the voxels (${num(r.voxelSize)} ${units}); a finer mesh usually fixes this.`
        : 'Check that every body touches the rest of the part or has its own fixed support, and that the load is on the supported part.';
      parts.push(h('div', {}, h('b', {}, 'These results are not reliable: '), `${why}. ${cause} The shape is shown undeformed.`));
      if (thinWalls && r.resolution < 480) parts.push(h('button.btn.small', { onclick: () => this.rerunWith(finer) }, `Use a finer mesh (${finer} voxels) and run again`));
    } else {
      parts.push(h('div', {}, `Some walls are thinner than a voxel (${num(r.voxelSize)} ${units}); they are modelled as connected layers with their true cross-section. Stiffness along the walls is accurate; bending of those walls themselves is approximate. For sheet-metal bending, raise the voxel count until this note disappears.`));
    }
    el.replaceChildren(...parts);
    el.style.borderColor = r.unreliable ? '' : 'var(--border)';
    el.style.background = r.unreliable ? '' : 'var(--panel)';
    el.hidden = false;
  }

  rerunWith(resolution) {
    const res = $('#res-range');
    res.value = resolution;
    $('#res-out').textContent = resolution;
    res.dispatchEvent(new Event('change'));
    this.run();
  }

  renderKPIs() {
    const r = this.result;
    const mat = r.material;
    const status = r.unreliable ? ['bad', 'Not reliable'] : r.incomplete ? ['warn', 'Partial model'] : r.minFos >= 2 ? ['good', 'Below strength'] : r.minFos >= 1 ? ['warn', 'Low margin'] : ['bad', 'Strength exceeded'];
    const loadAt = (lam) => (r.totalF > 1e-9 ? force(lam * r.totalF) : `${num(lam)} × loads`);
    const kpi = (label, value, sub, cls = '') => h(`div.kpi${cls}`, {}, h('div.k-label', {}, label), h('div.k-value', {}, value), sub ? h('div.k-sub', {}, sub) : null);
    $('#kpis').replaceChildren(
      kpi('Max von Mises', stress(r.maxVM), `yield ${num(mat.yield)} MPa`),
      kpi('Max displacement', `${num(r.maxDisp)} ${r.units}`, `net applied ${force(r.totalF)}`),
      h('div.kpi', {}, h('div.k-label', {}, 'Min factor of safety'), h('div.k-value', {}, Number.isFinite(r.minFos) ? num(r.minFos) : '∞'),
        h('div.k-sub', {}, h('span', { className: `status ${status[0]}` }, status[1]))),
      kpi(mat.brittle ? 'Estimated first crack' : 'Estimated first yield', Number.isFinite(r.lambda) ? loadAt(r.lambda) : '–', mat.brittle ? 'max principal = UTS' : 'von Mises = yield'),
    );
  }

  tick(dt) {
    if (this.app.tab !== 'structural') return;
    if (this.study !== 'static') {
      if (this.display === 'study') this.studies[this.study]?.tick(dt);
      return;
    }
    if (this.anim && this.result && this.display === 'results') {
      const a = this.anim, r = this.result;
      a.t = Math.min(a.dur, a.t + dt);
      const x = a.t / a.dur;
      a.lambda = r.lamBreak * x * x * (3 - 2 * x); // ease in and out
      const v = this.app.viewer;
      const spec = this.plotSpec(a.lambda);
      v.setDeformation(r.u, this.currentScale() * a.lambda);
      v.setScalars(spec.values, { min: spec.legend.min / a.lambda, max: spec.legend.max / a.lambda, bands: spec.bands, reverse: spec.reverse, colormap: spec.cmap });
      if (a.t - a.lastLegend > 0.1 || a.t >= a.dur) {
        a.lastLegend = a.t;
        renderLegend($('#legend'), { ...spec.legend, bands: spec.bands, colormap: spec.cmap, sub: `${r.material.name} · ${this.levelLabel(a.lambda)} · true-scale shape` });
      }
      if (a.t >= a.dur) {
        this.anim = null;
        this.setLevel('break');
        this.app.status(`${r.material.brittle ? 'Breaks' : 'Reaches its tensile strength'} at about ${r.totalF > 1e-9 ? force(r.lamBreak * r.totalF) : `${num(r.lamBreak)} × the loads`}, after bending ${num(r.maxDisp * r.lamBreak)} ${r.units}.`);
      }
      return;
    }
    if (this.display === 'results' && this.result && this.view.animate) {
      this.phase += dt;
      const peak = (this.currentScale() || (this.result.unreliable ? 0 : this.result.autoScale)) * this.loadFactor();
      const s = peak * (0.5 - 0.5 * Math.cos(this.phase * Math.PI));
      this.app.viewer.setDeformation(this.result.u, s);
      this.placeMarker(s);
    }
    if (this.playing && this.brk) {
      this.playT += dt;
      if (this.playT > 0.35) {
        this.playT = 0;
        const next = this.brk.current + 1;
        if (next >= this.brk.steps.length) this.stopBreakPlay();
        else this.setBreakStep(next);
      }
    }
  }

  probe(hit) {
    let values, fmt;
    if (this.display === 'study') return this.studies[this.study]?.probe(hit) ?? null;
    if (this.display === 'results' && this.result) {
      const spec = this.plotSpec(this.loadFactor());
      values = spec.values;
      fmt = spec.fmt;
    } else if (this.display === 'break' && this.brk && this.view.breakStress) {
      values = this.brk.shownVM;
      fmt = stress;
    }
    if (!values) return null;
    const T = this.part.tris, t = hit.tri;
    const w = [hit.bary.x, hit.bary.y, hit.bary.z];
    let s = 0;
    for (let k = 0; k < 3; k++) s += w[k] * values[T[3 * t + k]];
    return Number.isNaN(s) ? 'no data' : fmt(s);
  }

  // ---------- break test ----------

  async runBreak() {
    if (!this.checkSetup()) return;
    this.cancelJob();
    const version = this.runVersion;
    let prep;
    try {
      prep = await this.prepare(version);
    } catch (err) {
      if (version !== this.runVersion) return;
      this.preparing = false;
      this.app.busy.hide();
      return this.app.status(err.message, 'error');
    }
    if (version !== this.runVersion) return;
    const { model, asm, msg } = prep;
    const mat = prep.material;
    this.stopBreakPlay();
    const b = this.brk = { steps: [], totalF: Math.hypot(...asm.total), material: mat, current: 0, done: false, model, toMeters: prep.toMeters };
    $('#break-card').hidden = false;
    this.display = 'break';
    this.app.setXRay(true);
    this.renderBreakKPIs();
    try {
      this.job = new FEAJob(
      { type: 'break', ...msg, strength: mat.uts * 1e6, criterion: mat.brittle ? 'p1' : 'vm', maxSteps: 60 },
      {
        onProgress: (d) => {
          if (version === this.runVersion) this.app.busy.progress(Math.min(1, Math.log10(Math.max(d.res, 1e-7)) / -6), `${d.stage} · iteration ${d.it}`);
        },
        onStep: (d) => {
          if (version !== this.runVersion) return;
          b.steps.push(d);
          this.setBreakStep(b.steps.length - 1);
          this.renderBreakKPIs();
        },
      },
    );
    this.app.busy.show('Break test: growing the crack…', () => this.cancelJob());
      const done = await this.job.promise;
      if (version !== this.runVersion) return;
      b.done = true;
      b.reason = done.reason;
      this.app.status(
        done.reason === 'separated'
          ? `Damage illustration finished on the ${done.engine || 'CPU'}: the voxel load path separated after ${b.steps.length} steps.`
          : `Damage illustration stopped after ${b.steps.length} steps (${done.reason}).`,
        done.reason === 'separated' ? '' : 'warn',
      );
    } catch (err) {
      if (version !== this.runVersion) return;
      b.done = true;
      b.reason = err.cancelled ? 'stopped' : 'error';
      this.app.status(err.cancelled ? 'Break test stopped - showing the steps computed so far.' : `Solver error: ${err.message}`, err.cancelled ? 'warn' : 'error');
    }
    this.job = null;
    this.app.busy.hide();
    this.stale = false;
    this.renderBreakKPIs();
    if (this.brk.steps.length) this.setBreakStep(0);
    $('#break-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  renderBreakKPIs() {
    const b = this.brk;
    if (!b) return;
    const mat = b.material;
    const loadAt = (lam) => (b.totalF > 1e-9 ? force(lam * b.totalF) : `${num(lam)} × loads`);
    const s0 = b.steps[0];
    const peak = b.steps.reduce((m, s) => Math.max(m, s.lambda), 0);
    const kpi = (label, value, sub, cls = '') => h(`div.kpi${cls}`, {}, h('div.k-label', {}, label), h('div.k-value', {}, value), sub ? h('div.k-sub', {}, sub) : null);
    const outcome = !b.done ? 'running…' : b.reason === 'separated' ? 'load path separates' : b.reason;
    $('#break-kpis').replaceChildren(
      kpi('Estimated first crack', s0 ? loadAt(s0.lambda) : b.done ? '–' : '…', mat.brittle ? 'max principal = UTS' : `von Mises = UTS; yields at ${s0 ? loadAt((s0.lambda * mat.yield) / mat.uts) : '…'}`),
      kpi('Peak in computed steps', b.steps.length ? loadAt(peak) : b.done ? '–' : '…', `${b.steps.length} steps · ${outcome}`),
    );
    this.chart.opts.formatY = b.totalF > 1e-9 ? (value) => force(value) : (value) => `${num(value)} × loads`;
    this.chart.set(b.steps.map((s, i) => ({ x: i + 1, y: s.lambda * (b.totalF > 1e-9 ? b.totalF : 1) })), b.current);
    const slider = $('#break-step');
    slider.max = Math.max(0, b.steps.length - 1);
  }

  setBreakStep(i) {
    if (!this.brk || !this.brk.steps[i]) return;
    this.brk.current = i;
    $('#break-step').value = i;
    $('#break-step-out').textContent = `${i + 1}/${this.brk.steps.length}`;
    this.display = 'break';
    this.showBreakStep();
    this.chart.set(this.chart.points, i);
  }

  showBreakStep() {
    const b = this.brk;
    if (!b || !b.steps.length || this.app.tab !== 'structural') return;
    $('#results-alert').hidden = true;
    const v = this.app.viewer;
    const m = b.model, toM = b.toMeters;
    const k = Math.min(b.current, b.steps.length - 1);
    const s = b.steps[k];
    if (!s.mapped) {
      const W = m.vertexWeights(s.activeNode);
      const u = m.interpolate(W, s.u, 3);
      for (let i = 0; i < u.length; i++) u[i] /= toM;
      s.mapped = { vm: m.interpolate(W, s.nodeVM), u };
    }
    if (!b.scale) {
      const d0 = b.steps[0].maxDisp / toM;
      b.scale = d0 > 0 ? Math.min(1000, Math.max(1, (0.05 * this.part.bbox.diag) / d0)) : 1;
    }
    const scale = this.deformationScale(b.scale);
    const uts = b.material.uts * 1e6;
    const cmap = this.view.heat ? 'heat' : 'rainbow';
    const bands = this.view.bands ? 12 : 0;
    v.setVoxels('mesh', new Float32Array(0), 1, 0);
    if (v.mesh) { v.mesh.visible = true; v.edges.visible = v.showEdges !== false; }
    b.shownVM = s.mapped.vm;
    v.setScalars(this.view.breakStress ? s.mapped.vm : null, { min: 0, max: uts, bands, colormap: cmap });
    v.setDeformation(s.mapped.u, scale);
    // cracked and detached voxels up to this step, moved with the deformation
    const crack = [], gone = [];
    for (let j = 0; j <= k; j++) {
      for (const e of b.steps[j].cracked) crack.push(e);
      for (const e of b.steps[j].detached) gone.push(e);
    }
    v.setVoxels('cracks', this.voxelPositions(crack, s, scale), m.h * 1.02, CRACK_COLOR);
    v.setVoxels('detached', this.voxelPositions(gone, s, scale), m.h * 0.98, 0x7b8494, 0.3);
    const load = b.totalF > 1e-9 ? force(s.lambda * b.totalF) : `${num(s.lambda)} × loads`;
    renderLegend($('#legend'), this.view.breakStress
      ? { title: 'von Mises stress', sub: `Step ${k + 1} · load ${load} · ×${num(scale)}`, min: 0, max: uts, format: stressFormatter(uts), bands, colormap: cmap, markers: [{ value: b.material.yield * 1e6, label: `Yield ${num(b.material.yield)}` }] }
      : []);
    const first = b.steps[0].cracked[0];
    if (first !== undefined && this.view.marker) {
      const p = this.voxelPositions([first], s, scale);
      v.setMarker(new THREE.Vector3(p[0], p[1], p[2]), k === 0 ? 'Crack starts here' : 'Crack origin');
    } else v.clearMarker();
    this.drawOverlays();
  }

  voxelPositions(list, step, scale) {
    if (step.loaded) return this.loadedVoxelPositions(list, step, scale);
    const m = this.brk.model;
    const [nx, ny] = m.dims;
    const out = new Float32Array(list.length * 3);
    const toM = this.brk.toMeters;
    list.forEach((e, q) => {
      const i = e % nx, j = ((e / nx) | 0) % ny, k = (e / (nx * ny)) | 0;
      let ux = 0, uy = 0, uz = 0, c = 0;
      for (let a = 0; a < 8; a++) {
        const n = m.node(i + (a & 1), j + ((a >> 1) & 1), k + ((a >> 2) & 1));
        if (!step.activeNode[n]) continue;
        ux += step.u[3 * n]; uy += step.u[3 * n + 1]; uz += step.u[3 * n + 2];
        c++;
      }
      const f = c ? scale / (c * toM) : 0;
      out[3 * q] = m.origin[0] + (i + 0.5) * m.h + ux * f;
      out[3 * q + 1] = m.origin[1] + (j + 0.5) * m.h + uy * f;
      out[3 * q + 2] = m.origin[2] + (k + 0.5) * m.h + uz * f;
    });
    return out;
  }

  toggleBreakPlay() {
    if (this.playing) return this.stopBreakPlay();
    if (!this.brk?.steps.length) return;
    if (this.brk.current >= this.brk.steps.length - 1) this.setBreakStep(0);
    this.playing = true;
    this.playT = 0;
    $('#btn-break-play').textContent = '❚❚ Pause';
  }

  stopBreakPlay() {
    this.playing = false;
    $('#btn-break-play').textContent = '▶ Play';
  }

  // ---------- .psim file: setup and results as plain data ----------

  /** Material, fixtures, loads and study settings as plain data. */
  exportSetup() {
    const res = Number($('#res-range').value);
    const options = {};
    for (const [id, st] of Object.entries(this.studies)) {
      const o = st.exportOptions?.();
      if (o) options[id] = o;
    }
    return {
      study: this.study,
      resolution: res,
      resFor: { ...this.resFor, [this.study]: res },
      gravity: $('#chk-gravity').checked,
      engine: $('#fea-engine').value,
      fixtures: this.fixtures.map((f) => ({ name: f.name, patches: f.patches.map(patchToJson) })),
      loads: this.loads.map((l) => ({
        name: l.name, type: l.type, magnitude: l.magnitude, dir: l.dir.slice(), userDir: !!l.userDir,
        ...(l.type === 'wind' ? { net: l.net.slice() } : { patches: l.patches.map(patchToJson) }),
      })),
      options,
    };
  }

  /** Per-triangle forces of the wind loads, as arrays named load.<index>.forces. */
  exportWindArrays() {
    return this.loads.flatMap((l, i) => (l.type === 'wind' && l.forces ? [{ name: `load.${i}.forces`, data: l.forces, enc: 'q16' }] : []));
  }

  /** Puts the material back, library or custom, as saved. */
  importMaterial(m) {
    if (!m || typeof m !== 'object' || !Number.isFinite(m.E) || !Number.isFinite(m.density)) throw new Error('The material in the file is incomplete.');
    this.material = { ...m, fatigue: m.fatigue ? { ...m.fatigue } : m.fatigue };
    const sel = $('#mat-select');
    sel.value = MATERIALS.some((x) => x.id === m.id) ? m.id : 'custom';
    this.renderMaterial();
  }

  /** Call after the part is loaded (reset has cleared the old setup). */
  importSetup(s, arrays = new Map()) {
    const nTri = this.part.nTri;
    this.fixtures = (s.fixtures || []).map((f) => ({ id: uid++, name: String(f.name), patches: (f.patches || []).map((p) => patchFromJson(p, nTri)).filter((p) => p.tris.length) }));
    this.loads = (s.loads || []).map((l, i) => {
      const dir = Array.isArray(l.dir) && l.dir.length === 3 ? l.dir.map(Number) : [0, -1, 0];
      const base = { id: uid++, name: String(l.name), type: l.type, magnitude: Number(l.magnitude), dir, userDir: !!l.userDir };
      if (l.type === 'wind') {
        const f = arrays.get(`load.${i}.forces`);
        if (!f || f.data.length !== 3 * nTri) throw new Error('The wind load in the file has no forces for this part.');
        return { ...base, forces: Float32Array.from(f.data), net: (l.net || [0, 0, 0]).map(Number), patches: [] };
      }
      if (l.type !== 'force' && l.type !== 'pressure') throw new Error(`The file has a load of an unknown type (${l.type}).`);
      return { ...base, patches: (l.patches || []).map((p) => patchFromJson(p, nTri)).filter((p) => p.tris.length) };
    });
    this.selected = this.loads[0] || null;
    $('#chk-gravity').checked = !!s.gravity;
    if (s.engine && [...$('#fea-engine').options].some((o) => o.value === s.engine)) $('#fea-engine').value = s.engine;
    this.resFor = {};
    for (const [id, n] of Object.entries(s.resFor || {})) if (STUDY_INFO[id] && Number.isFinite(n)) this.resFor[id] = Math.min(480, Math.max(16, Math.round(n)));
    for (const [id, o] of Object.entries(s.options || {})) this.studies[id]?.importOptions?.(o);
    this.renderLists();
  }

  /** Switches to a study without running or clearing anything. */
  restoreStudy(id) {
    if (!STUDY_INFO[id]) id = 'static';
    this.study = id;
    $('#study-select').value = id;
    const n = this.resFor[id] ?? suggestResolution(this.part, STUDY_INFO[id].budget);
    $('#res-range').value = n;
    $('#res-out').textContent = n;
    this.updateMeshInfo();
    const st = this.studies[id];
    this.display = id === 'static' ? (this.result ? 'results' : this.brk?.steps.length ? 'break' : 'setup') : st?.result ? 'study' : 'setup';
    this.renderStudyOptions();
  }

  /** The linear static result as { meta, arrays }, or null. */
  exportStatic() {
    const r = this.result;
    if (!r) return null;
    const meta = {};
    for (const [k, v] of Object.entries(r)) if (!ArrayBuffer.isView(v) && k !== 'reaction') meta[k] = v;
    const field = (name) => ({ name: `static.${name}`, data: r[name], enc: 'q16', ...(name === 'u' ? { stride: 3 } : {}) });
    return { meta, arrays: ['vm', 'p1', 'p3', 'fos', 'u'].map(field) };
  }

  importStatic(meta, arrays) {
    const nV = this.part.nVert;
    const get = (name, n) => {
      const a = arrays.get(`static.${name}`);
      if (!a || a.data.length !== n) throw new Error(`The static result in the file does not fit this part (${name}).`);
      return a.data;
    };
    const u = get('u', 3 * nV);
    const dmag = new Float32Array(nV);
    for (let v = 0; v < nV; v++) dmag[v] = Number.isNaN(u[3 * v]) ? NaN : Math.hypot(u[3 * v], u[3 * v + 1], u[3 * v + 2]);
    this.result = { ...meta, vm: get('vm', nV), p1: get('p1', nV), p3: get('p3', nV), fos: get('fos', nV), u, dmag, reaction: null };
    this.stale = false;
    this.display = 'results';
    $('#results-card').hidden = false;
  }

  /** The break test as { meta, arrays }, or null. Only what the pictures need is kept. */
  exportBreak() {
    const b = this.brk;
    if (!b || !b.steps.length) return null;
    const m = b.model;
    const arrays = [];
    const steps = b.steps.map((s) => {
      if (!s.mapped) {
        const W = m.vertexWeights(s.activeNode);
        const u = m.interpolate(W, s.u, 3);
        for (let q = 0; q < u.length; q++) u[q] /= b.toMeters;
        s.mapped = { vm: m.interpolate(W, s.nodeVM), u };
      }
      return { step: s.step, lambda: s.lambda, maxDisp: s.maxDisp };
    });
    // the steps are two series (stress, displacement): one shared range each, every step predicted from the one before
    // Each step's stress and displacement are those of the unit load times the step's load factor (lambda), plus the effect of
    // the cracks so far. Dividing by lambda leaves what the cracks changed, which is what the step before predicts well:
    // the series is then less than half the size (meta.fieldScale = 'perLambda').
    const perLambda = (arr, s) => (s.lambda > 0 ? Float32Array.from(arr, (v) => v / s.lambda) : arr);
    arrays.push(...seriesSpecs(b.steps.map((s) => perLambda(s.mapped.vm, s)), (i) => `break.vm.${i}`), ...seriesSpecs(b.steps.map((s) => perLambda(s.mapped.u, s)), (i) => `break.u.${i}`, { stride: 3 }));
    b.steps.forEach((s, i) => arrays.push({ name: `break.cracked.${i}`, data: Int32Array.from(s.cracked), enc: 'i32' }, { name: `break.detached.${i}`, data: Int32Array.from(s.detached), enc: 'i32' }));
    return {
      meta: { fieldScale: 'perLambda', steps, totalF: b.totalF, material: b.material, toMeters: b.toMeters, reason: b.reason ?? null, done: !!b.done, scale: b.scale ?? null, grid: { dims: Array.from(m.dims), origin: Array.from(m.origin), h: m.h, resolution: m.resolution } },
      arrays,
    };
  }

  importBreak(meta, arrays) {
    const nV = this.part.nVert;
    const grid = meta.grid;
    if (!grid || !Array.isArray(grid.dims) || grid.dims.length !== 3 || !Array.isArray(grid.origin) || !(grid.h > 0)) throw new Error('The break test in the file has no voxel grid.');
    const need = (name, n) => {
      const a = arrays.get(name);
      if (!a || (n !== undefined && a.data.length !== n)) throw new Error('The break test in the file does not fit this part.');
      return a.data;
    };
    const steps = meta.steps.map((s, i) => ({
      step: s.step, lambda: s.lambda, maxDisp: s.maxDisp, loaded: true,
      cracked: need(`break.cracked.${i}`), detached: need(`break.detached.${i}`),
      mapped: { vm: need(`break.vm.${i}`, nV), u: need(`break.u.${i}`, 3 * nV) },
    }));
    if (meta.fieldScale === 'perLambda') {
      for (const s of steps) {
        if (!(s.lambda > 0)) continue;
        for (const k of ['vm', 'u']) { const a = s.mapped[k]; for (let q = 0; q < a.length; q++) a[q] *= s.lambda; }
      }
    }
    const model = { dims: grid.dims, origin: grid.origin, h: grid.h, resolution: grid.resolution };
    this.brk = { steps, totalF: meta.totalF, material: meta.material, current: 0, done: true, reason: meta.reason, model, toMeters: meta.toMeters, scale: meta.scale || undefined, loaded: true };
    if (!this.result) this.display = 'break';
    $('#break-card').hidden = false;
  }

  /** Nearest-vertex displacement of a voxel centre, for break steps loaded from a file. */
  loadedVoxelPositions(list, step, scale) {
    const b = this.brk, m = b.model;
    if (!b.hash) {
      const V = this.part.vertices, cell = m.h * 2, map = new Map();
      for (let i = 0; i < this.part.nVert; i++) {
        const key = `${Math.floor(V[3 * i] / cell)},${Math.floor(V[3 * i + 1] / cell)},${Math.floor(V[3 * i + 2] / cell)}`;
        let a = map.get(key);
        if (!a) map.set(key, (a = []));
        a.push(i);
      }
      b.hash = { map, cell };
    }
    const { map, cell } = b.hash, V = this.part.vertices, u = step.mapped.u;
    const [nx, ny] = m.dims;
    const out = new Float32Array(list.length * 3);
    list.forEach((e, q) => {
      const i = e % nx, j = ((e / nx) | 0) % ny, k = (e / (nx * ny)) | 0;
      const x = m.origin[0] + (i + 0.5) * m.h, y = m.origin[1] + (j + 0.5) * m.h, z = m.origin[2] + (k + 0.5) * m.h;
      const cx = Math.floor(x / cell), cy = Math.floor(y / cell), cz = Math.floor(z / cell);
      let best = -1, bd = Infinity;
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
        for (const v of map.get(`${cx + dx},${cy + dy},${cz + dz}`) || []) {
          const d = (V[3 * v] - x) ** 2 + (V[3 * v + 1] - y) ** 2 + (V[3 * v + 2] - z) ** 2;
          if (d < bd) { bd = d; best = v; }
        }
      }
      const f = best >= 0 && !Number.isNaN(u[3 * best]) ? scale : 0;
      out[3 * q] = x + (f ? u[3 * best] * f : 0);
      out[3 * q + 1] = y + (f ? u[3 * best + 1] * f : 0);
      out[3 * q + 2] = z + (f ? u[3 * best + 2] * f : 0);
    });
    return out;
  }

  exportView() {
    const v = this.view;
    return { plot: v.plot, scalePct: v.scalePct, level: v.level, bands: v.bands, heat: v.heat, bcs: v.bcs, marker: v.marker, breakStress: v.breakStress, breakStep: this.brk?.current ?? 0 };
  }

  importView(v = {}) {
    const take = (k, ok) => { if (k in v && ok(v[k])) this.view[k] = v[k]; };
    take('plot', (x) => ['vm', 'disp', 'fos', 'p1', 'p3'].includes(x));
    take('scalePct', Number.isFinite);
    take('level', (x) => ['applied', 'yield', 'break'].includes(x));
    for (const k of ['bands', 'heat', 'bcs', 'marker', 'breakStress']) take(k, (x) => typeof x === 'boolean');
    $('#plot-select').value = this.view.plot;
    $('#scale-range').value = this.view.scalePct;
    $('#chk-bands').checked = this.view.bands;
    $('#chk-heat').checked = this.view.heat;
    $('#chk-bcs').checked = this.view.bcs;
    $('#chk-marker').checked = this.view.marker;
    $('#chk-break-stress').checked = this.view.breakStress;
    if (this.result) this.setLevel(this.view.level, false);
    if (this.brk && Number.isInteger(v.breakStep)) this.brk.current = Math.max(0, Math.min(this.brk.steps.length - 1, v.breakStep));
  }

}

/**
 * Voxels along the longest side: aim for ~5 voxels across chunky sections and ~2 through thin
 * walls, within a budget that keeps a run to seconds (~300k grid nodes, ~80k voxels).
 */
function suggestResolution(part, budget = 8e4) {
  const size = part.bbox.size;
  const maxDim = Math.max(...size);
  const bulk = (3 * part.volume) / part.area;
  const wall = typicalWallThickness(part);
  const h = Math.min(bulk / 5, Number.isFinite(wall) ? wall / 2 : Infinity);
  let n = Math.min(480, Math.max(32, Math.round(maxDim / h)));
  const fits = (n) => {
    const hv = maxDim / n;
    const nodes = size.reduce((p, s) => p * (Math.ceil(s / hv) + 1), 1);
    const voxels = Math.max(part.volume / hv ** 3, (0.75 * part.area) / hv ** 2);
    return nodes <= 3e5 * (budget / 8e4) ** 0.8 && voxels <= budget; // keeps a default run to seconds; raise the slider for more
  };
  while (n > 24 && !fits(n)) n -= 4;
  return Math.round(n / 4) * 4;
}

function allTris(feature) {
  if (feature.patches.length === 1) return feature.patches[0].tris;
  const set = new Set();
  for (const p of feature.patches) for (const t of p.tris) set.add(t);
  return Int32Array.from(set);
}
