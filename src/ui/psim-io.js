// Saving and opening .psim files in the app: what goes into a file, the Save dialog with live sizes,
// restoring a file's part, setup and results into the panels, and the "loaded from file" banner and
// info panel. The container and codecs are in core/psim.js; this file talks to the panels.
import { writePsim, readPsim, deflateRaw, encodeGeometry, encodeArrays, stringifyJson, PsimError, PSIM_VERSION, MIME } from '../core/psim.js';
import { restorePart } from '../core/mesh.js';
import { $, h, num } from './dom.js';

export const PSIM_ACCEPT = '.psim';
export const isPsimName = (name) => /\.psim$/i.test(name || '');

export const RESULT_LABELS = {
  static: 'Bend test (static stresses)',
  break: 'Break test',
  nonlinear: 'Nonlinear static',
  modal: 'Frequency',
  buckling: 'Buckling',
  fatigue: 'Fatigue',
  drop: 'Drop test',
  optimize: 'Optimization',
  thermal: 'Thermal',
  airflow: 'Airflow',
};
const STUDY_IDS = ['nonlinear', 'modal', 'buckling', 'fatigue', 'drop', 'optimize'];

const kb = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)} MB` : n >= 1e3 ? `${(n / 1e3).toFixed(1)} KB` : `${n} B`);
const bytesOf = (value) => new TextEncoder().encode(stringifyJson(value));

// ---------------------------------------------------------------- what a file can hold

/** Everything the app could write now, as plain data. Nothing is compressed yet. */
export function gather(app) {
  const part = app.part;
  if (!part) throw new Error('Open a part or a sample first.');
  const st = app.structural;
  const results = new Map();
  const put = (id, r) => { if (r && r.arrays) results.set(id, r); };
  put('static', st.exportStatic());
  put('break', st.exportBreak());
  for (const id of STUDY_IDS) put(id, st.studies[id]?.exportResult?.());
  put('thermal', app.thermal.exportResult());
  put('airflow', app.airflow.study?.exportResults?.());
  const cam = app.viewer.camera, target = app.viewer.controls.target;
  return {
    name: part.name,
    units: app.units,
    geometry: { vertices: part.vertices, tris: part.tris, faceOf: part.brepFaces ? part.faceOf : null, brepFaces: part.brepFaces, faceCount: part.faceCount, faceAngle: part.faceAngle ?? 20 },
    setup: {
      units: app.units,
      material: { ...st.material },
      structural: st.exportSetup(),
      thermal: app.thermal.exportState(),
      airflow: app.airflow.exportState?.() ?? null,
    },
    windArrays: st.exportWindArrays(),
    results,
    view: {
      tab: app.tab,
      study: st.study,
      camera: { position: cam.position.toArray(), target: target.toArray(), up: cam.up.toArray() },
      structural: st.exportView(),
      thermal: app.thermal.exportView(),
      airflow: app.airflow.exportView?.() ?? null,
    },
    cad: app.sourceFile ?? null,
    partInfo: app.partInfo,
  };
}

/** A small JPEG of the viewport. */
export async function makeThumb(app, width = 320) {
  try {
    const png = app.viewer.screenshot(getComputedStyle(document.documentElement).getPropertyValue('--vp-bottom').trim() || '#e9edf2');
    const img = new Image();
    img.src = png;
    await img.decode();
    const c = document.createElement('canvas');
    c.width = width;
    c.height = Math.max(1, Math.round((width * img.height) / img.width));
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    const blob = await new Promise((ok) => c.toBlob(ok, 'image/jpeg', 0.72));
    return blob ? new Uint8Array(await blob.arrayBuffer()) : null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------- building a file

/**
 * options: { geometry: 'quantised'|'exact'|null, setup: bool, results: string[], cad: bool, thumb: bool, name, notes }
 * Returns { bytes, info }.
 */
export async function buildFile(app, draft, options, thumb = null, created = null) {
  const t0 = performance.now();
  const wantResults = options.results.filter((id) => draft.results.has(id));
  if ((options.setup || wantResults.length) && !options.geometry) throw new Error('Setup and results are saved together with the geometry.');
  const content = {};
  const info = {
    app: { name: 'Parts Sim', version: typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : 'dev', kind: app.desktop ? 'electron' : 'web' },
    name: options.name || draft.name,
    notes: options.notes || '',
    units: draft.units,
    contains: { geometry: options.geometry === 'quantised' ? 'quantised16' : options.geometry === 'exact' ? 'exact' : null, setup: !!options.setup, results: [], cad: false },
  };
  if (options.geometry) {
    content.geometry = draft.geometry;
    const V = draft.geometry.vertices;
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < V.length; i += 3) for (let d = 0; d < 3; d++) { if (V[i + d] < min[d]) min[d] = V[i + d]; if (V[i + d] > max[d]) max[d] = V[i + d]; }
    info.part = { vertices: V.length / 3, triangles: draft.geometry.tris.length / 3, bbox: { min, max } };
  }
  if (options.setup) {
    content.setup = draft.setup;
    if (draft.partInfo) info.partSource = String(draft.partInfo).slice(0, 200);
  }
  const arrays = [];
  const meta = { results: [] };
  for (const id of wantResults.filter((x) => x !== 'airflow')) {
    const r = draft.results.get(id);
    meta.results.push(id);
    meta[id] = r.meta;
    arrays.push(...r.arrays);
  }
  if (options.setup) arrays.push(...draft.windArrays);
  if (meta.results.length || (options.setup && draft.windArrays.length)) content.rfea = { meta, arrays };
  if (wantResults.includes('airflow')) {
    const r = draft.results.get('airflow');
    content.rair = { meta: r.meta, arrays: r.arrays };
  }
  info.contains.results = wantResults;
  if (options.cad && draft.cad) {
    content.cad = new Uint8Array(await draft.cad.arrayBuffer());
    info.contains.cad = true;
    info.contains.cadName = draft.cad.name.replace(/^.*[\\/]/, '');
  }
  if (options.thumb && thumb) content.thumb = thumb;
  if (options.setup || wantResults.length) content.view = draft.view;
  content.info = info;
  const t1 = performance.now();
  const bytes = await writePsim(content, { geometryMode: options.geometry ?? 'quantised', created: created ?? new Date().toISOString().replace(/\.\d+Z$/, 'Z') });
  app.psimTimings = { ...app.psimTimings, write: { prepare: t1 - t0, writePsim: performance.now() - t1, total: performance.now() - t0 } };
  return { bytes, info };
}

/** Compressed size of each piece, for the Save dialog. */
export async function measure(draft) {
  const z = async (raw) => (await deflateRaw(raw)).length;
  const sizes = { geometry: {}, results: {} };
  for (const mode of ['quantised', 'exact']) sizes.geometry[mode] = await z(encodeGeometry(draft.geometry, mode).bytes);
  sizes.setup = (await z(bytesOf(draft.setup))) + (draft.windArrays.length ? await z(encodeArrays({}, draft.windArrays).bytes) : 0);
  for (const [id, r] of draft.results) sizes.results[id] = await z(encodeArrays(r.meta, r.arrays).bytes);
  return sizes;
}

// ---------------------------------------------------------------- saving

async function deliver(app, bytes, fileName) {
  if (app.desktop) return app.desktop.saveFile(bytes, fileName);
  if ('showSaveFilePicker' in window) {
    try {
      const handle = await window.showSaveFilePicker({ suggestedName: fileName, types: [{ description: 'Parts Sim file', accept: { [MIME]: ['.psim'] } }] });
      const w = await handle.createWritable();
      await w.write(bytes);
      await w.close();
      return true;
    } catch (err) {
      if (err.name === 'AbortError') return false;
      // fall through to a plain download
    }
  }
  const url = URL.createObjectURL(new Blob([bytes], { type: MIME }));
  h('a', { href: url, download: fileName }).click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  return true;
}

const safeName = (s) => String(s || 'part').replace(/[\\/:*?"<>|]+/g, '_').replace(/\.psim$/i, '').trim() || 'part';

export async function saveDialog(app) {
  let draft;
  try { draft = gather(app); } catch (err) { return app.status(err.message, 'error'); }
  const dlg = $('#save-dialog');
  const body = $('#save-body');
  body.replaceChildren(h('p', {}, 'Measuring…'));
  dlg.showModal();
  const [sizes, thumb] = await Promise.all([measure(draft), makeThumb(app)]);
  let cadZ = null;
  if (draft.cad) cadZ = draft.cad.size;

  // a result over 4 MB (a long break test, many drop frames) starts unchecked; the size next to it says why
  const state = { geometry: 'quantised', setup: true, results: new Set([...draft.results.keys()].filter((id) => (sizes.results[id] ?? 0) < 4e6)), cad: false, thumb: true };
  const total = h('strong');
  const refresh = () => {
    let n = 2000 + (state.thumb && thumb ? thumb.length : 0);
    if (state.geometry) n += sizes.geometry[state.geometry];
    if (state.setup) n += sizes.setup;
    for (const id of state.results) n += sizes.results[id] ?? 0;
    if (state.cad && cadZ) n += cadZ;
    total.textContent = `About ${kb(n)}`;
    save.disabled = false;
  };
  const check = (label, size, key, extra = {}) => h('label.check', {}, h('input', { type: 'checkbox', checked: extra.checked ?? true, disabled: extra.disabled, onchange: (e) => { extra.set(e.target.checked); refresh(); } }), ` ${label} `, h('span.muted', {}, size));
  const geomRadio = (mode, label) => h('label.check', {}, h('input', { type: 'radio', name: 'psim-geom', value: mode, checked: mode === 'quantised', onchange: () => { state.geometry = mode; refresh(); } }), ` ${label} `, h('span.muted', {}, kb(sizes.geometry[mode])));
  const name = h('input', { type: 'text', value: draft.name, 'aria-label': 'Name' });
  const notes = h('textarea', { rows: 2, placeholder: 'Notes for whoever opens the file (optional)', 'aria-label': 'Notes' });
  const resultRows = [...draft.results.keys()].map((id) => check(RESULT_LABELS[id] ?? id, kb(sizes.results[id]), id, { checked: state.results.has(id), set: (on) => (on ? state.results.add(id) : state.results.delete(id)) }));
  const save = h('button.btn.primary', {
    type: 'button',
    onclick: async () => {
      save.disabled = true;
      save.textContent = 'Saving…';
      try {
        const options = {
          geometry: state.geometry, setup: state.setup, results: [...state.results], cad: state.cad, thumb: state.thumb,
          name: name.value.trim() || draft.name, notes: notes.value.trim(),
        };
        if (!state.setup && !state.results.size) { /* geometry only */ }
        const { bytes } = await buildFile(app, draft, options, thumb);
        const ok = await deliver(app, bytes, `${safeName(name.value || draft.name)}.psim`);
        dlg.close();
        if (ok) app.status(`Saved ${kb(bytes.length)} as a .psim file.`);
      } catch (err) {
        save.disabled = false;
        save.textContent = 'Save';
        app.status(`Could not save: ${err.message}`, 'error');
      }
    },
  }, 'Save');
  body.replaceChildren(
    h('label.field', {}, h('span', {}, 'Name'), name),
    notes,
    h('fieldset', {}, h('legend', {}, 'Part'),
      geomRadio('quantised', 'Geometry, compact (positions to 16 bits)'),
      geomRadio('exact', 'Geometry, exact (32-bit positions)')),
    h('fieldset', {}, h('legend', {}, 'Include'),
      check('Setup (material, supports, loads, study settings)', kb(sizes.setup), 'setup', { set: (on) => (state.setup = on) }),
      ...(resultRows.length ? resultRows : [h('p.muted', {}, 'No results yet: run a study to include its results.')]),
      check('Original CAD file', draft.cad ? kb(draft.cad.size) : 'not available', 'cad', { checked: false, disabled: !draft.cad, set: (on) => (state.cad = on) }),
      check('Preview picture', thumb ? kb(thumb.length) : '', 'thumb', { disabled: !thumb, set: (on) => (state.thumb = on) })),
    h('p.muted', {}, 'Lossy fields are listed with their largest error in the file’s info panel. The file never holds your user name, folders or computer details, but it does hold the part’s geometry: share it only with people who may see the part.'),
    h('div.dialog-actions', {}, h('span', {}, total), h('button.btn', { type: 'button', onclick: () => dlg.close() }, 'Cancel'), save),
  );
  refresh();
}

// ---------------------------------------------------------------- opening

/** Opens a .psim file's bytes: the part, its setup and its results, shown without solving. */
export async function openPsim(app, bytes, fileName, install) {
  const t = { start: performance.now() };
  const mark = (k) => { t[k] = performance.now() - t.start; };
  let f;
  try {
    f = await readPsim(bytes);
  } catch (err) {
    if (err instanceof PsimError) return app.status(`${fileName}: ${err.message}`, 'error');
    throw err;
  }
  mark('read');
  const info = f.info ?? {};
  if (!f.geometry) {
    showInfo(app, { name: fileName, info, table: f.table, size: bytes.length, version: f.version });
    return app.status(`${fileName} holds no geometry, so there is nothing to show. Its info panel is open.`, 'warn');
  }
  const part = restorePart({ name: info.name || fileName.replace(/\.psim$/i, ''), ...f.geometry });
  mark('restorePart');
  const units = f.setup?.units || info.units || 'mm';
  const loaded = await install(part, { units, info: info.partSource || `From ${fileName}` });
  if (!loaded) return;
  mark('install');
  const problems = [];
  const guard = (what, fn) => { try { fn(); } catch (err) { problems.push(`${what}: ${err.message}`); } };
  const st = app.structural;
  const setup = f.setup ?? {};
  const arrays = f.rfea?.arrays ?? new Map();
  const rf = f.rfea?.meta ?? {};
  guard('material', () => setup.material && st.importMaterial(setup.material));
  guard('setup', () => setup.structural && st.importSetup(setup.structural, arrays));
  guard('thermal setup', () => setup.thermal && app.thermal.importState(setup.thermal));
  guard('airflow setup', () => setup.airflow && app.airflow.importState?.(setup.airflow));
  const stored = {};
  for (const id of rf.results ?? []) {
    const m = rf[id];
    if (id === 'static') guard('static result', () => { st.importStatic(m, arrays); stored.static = { maxVM: m.maxVM, maxDisp: m.maxDisp, minFos: m.minFos }; });
    else if (id === 'break') guard('break test', () => st.importBreak(m, arrays));
    else if (id === 'thermal') guard('thermal result', () => { app.thermal.importResult(m, arrays); stored.thermal = { min: m.min, max: m.max }; });
    else if (st.studies[id]?.importResult) guard(`${RESULT_LABELS[id] ?? id} result`, () => st.studies[id].importResult(m, arrays));
    else problems.push(`${RESULT_LABELS[id] ?? id}: this app cannot show it`);
  }
  // study settings again, now that the results are in: the view (mode, step, frame) is clamped to them
  for (const id of rf.results ?? []) {
    const o = setup.structural?.options?.[id];
    if (o && st.studies[id]?.result) guard(`${RESULT_LABELS[id] ?? id} settings`, () => st.studies[id].importOptions?.(o));
  }
  if (f.rair) {
    // the airflow panel restores its settings, the frozen study and everything a finished run shows
    const r = f.rair.meta;
    await app.airflow.loadResults(r, f.rair.arrays, setup.airflow ?? null).then(
      () => { const a = r.results ?? {}; stored.airflow = { drag: a.drag, lift: a.lift, cd: a.cd, cl: a.cl }; },
      (err) => problems.push(`airflow result: ${err.message}`),
    );
  }
  mark('results');
  const view = f.view ?? {};
  guard('view', () => {
    st.importView(view.structural);
    app.thermal.importView(view.thermal);
    app.airflow.importView?.(view.airflow);
    st.restoreStudy(view.study || setup.structural?.study || 'static');
    const c = view.camera;
    if (c && c.position?.length === 3 && c.target?.length === 3) {
      app.viewer.camera.position.fromArray(c.position);
      app.viewer.camera.up.fromArray(c.up ?? [0, 1, 0]);
      app.viewer.controls.target.fromArray(c.target);
      app.viewer.controls.update();
    }
  });
  const hasResults = (rf.results?.length ?? 0) > 0 || !!f.rair;
  const tab = view.tab && ['part', 'structural', 'thermal', 'airflow'].includes(view.tab) ? view.tab : f.rair ? 'airflow' : rf.results?.includes('thermal') ? 'thermal' : hasResults ? 'structural' : 'part';
  app.setTab(tab);
  mark('total');
  app.psimTimings = { ...app.psimTimings, open: t };
  app.loadedFile = { name: fileName, info, table: f.table, size: bytes.length, version: f.version, hasResults, hasSetup: !!f.setup, stored, skipped: f.skipped };
  showBanner(app);
  app.status(
    `Opened ${fileName}: ${hasResults ? 'results shown from the file, not computed here' : 'part and setup'}.${problems.length ? ` Could not restore: ${problems.join('; ')}.` : ''}`,
    problems.length ? 'warn' : '',
  );
}

// ---------------------------------------------------------------- banner and info panel

export function showBanner(app) {
  const el = $('#file-banner');
  const f = app.loadedFile;
  if (!f) { el.hidden = true; return; }
  el.hidden = false;
  el.replaceChildren(
    h('span.banner-text', {}, h('b', {}, f.hasResults ? (f.edited ? 'Results are from the file; the setup has changed since' : 'Loaded from file, not computed here') : 'Opened from file'), ` · ${f.name}`),
    h('button.btn.small', { type: 'button', onclick: () => showInfo(app, f) }, 'File info'),
    ...(f.hasSetup && f.hasResults ? [h('button.btn.small', { type: 'button', onclick: () => rerun(app) }, 'Re-run')] : []),
  );
}

export function clearLoaded(app) {
  app.loadedFile = null;
  $('#file-banner').hidden = true;
}

/** Re-solves what the file showed and, when it finishes, says how the new numbers differ. */
function rerun(app) {
  const f = app.loadedFile;
  if (!f) return;
  const tab = app.tab;
  app.rerunOf = tab === 'airflow' ? null : { stored: f.stored, name: f.name };
  $('#file-banner').hidden = true;
  if (tab === 'thermal') $('#btn-th-run').click();
  else if (tab === 'airflow') $('#btn-flow-run').click();
  else if (app.structural.study === 'static' && app.structural.display === 'break') $('#btn-break').click();
  else $('#btn-run').click();
}

const pct = (a, b) => (Number.isFinite(a) && Number.isFinite(b) && b !== 0 ? `${a >= b ? '+' : '−'}${Math.abs(((a - b) / b) * 100).toFixed(1)}%` : '');

/** Called by the panels when a run finishes; compares with the numbers stored in the file. */
export function runDone(app, kind, now) {
  const r = app.rerunOf;
  if (!r || !r.stored?.[kind]) return;
  const s = r.stored[kind];
  const bits = Object.keys(s).filter((k) => Number.isFinite(s[k]) && Number.isFinite(now[k])).map((k) => `${k} ${num(now[k])} (file ${num(s[k])}, ${pct(now[k], s[k]) || 'same'})`);
  app.rerunOf = null;
  clearLoaded(app);
  if (bits.length) app.status(`Re-run finished. Against ${r.name}: ${bits.join('; ')}.`);
}

export function showInfo(app, f) {
  const dlg = $('#info-dialog');
  const info = f.info ?? {};
  const rows = (pairs) => h('dl.info-list', {}, ...pairs.flatMap(([k, v]) => (v === undefined || v === null || v === '' ? [] : [h('dt', {}, k), h('dd', {}, String(v))])));
  const c = info.contains ?? {};
  const bounds = info.bounds ?? [];
  dlg.querySelector('.info-body').replaceChildren(
    h('h2', {}, f.name),
    rows([
      ['Part', info.name],
      ['Notes', info.notes],
      ['Made by', info.app ? `${info.app.name} ${info.app.version} (${info.app.kind})` : ''],
      ['Created', info.created],
      ['Format', `version ${f.version}`],
      ['Units', info.units],
      ['Part size', info.part ? `${info.part.vertices.toLocaleString()} vertices, ${info.part.triangles.toLocaleString()} triangles` : ''],
      ['Geometry', c.geometry === 'exact' ? 'exact' : c.geometry ? 'compact (16-bit positions)' : 'not included'],
      ['Setup', c.setup ? 'included' : 'not included'],
      ['Results', (c.results ?? []).map((id) => RESULT_LABELS[id] ?? id).join(', ') || 'none'],
      ['CAD source', c.cad ? c.cadName || 'included' : 'not included'],
      ['File size', kb(f.size)],
    ]),
    h('h3', {}, 'Sections'),
    h('table', {}, h('thead', {}, h('tr', {}, ...['Section', 'Stored', 'Unpacked'].map((t) => h('th', {}, t)))),
      h('tbody', {}, ...(f.table ?? []).map((e) => h('tr', {}, h('td', {}, sectionName(e.id)), h('td', {}, kb(e.stored)), h('td', {}, kb(e.raw)))))),
    h('h3', {}, 'Largest error of each lossy field'),
    bounds.length
      ? h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'Field'), h('th', {}, 'Largest error'))),
        h('tbody', {}, ...bounds.map((b) => h('tr', {}, h('td', {}, b.field), h('td', {}, num(b.absolute, 3))))))
      : h('p.muted', {}, 'Everything in this file is exact.'),
    ...(f.skipped?.length ? [h('p.muted', {}, `Sections this version does not know were skipped: ${f.skipped.join(', ')}.`)] : []),
  );
  dlg.showModal();
}

const SECTION_NAMES = { INFO: 'Info', THMB: 'Preview picture', GEOM: 'Geometry', CADS: 'CAD source', SETP: 'Setup', RFEA: 'Structural and thermal results', RAIR: 'Airflow results', VIEW: 'View' };
const sectionName = (id) => SECTION_NAMES[id] ?? `${id} (unknown)`;

export { PSIM_VERSION };
