import { Viewer } from './viewer/viewer.js';
import { buildPart, rotatePositions90, setSmoothFaces, scalePart } from './core/mesh.js';
import { importFile, primaryFile, ACCEPT } from './core/importers.js';
import { matchMaterial } from './core/solidworks.js';
import { MATERIALS } from './core/materials.js';
import { SAMPLES } from './core/samples.js';
import { UNITS } from './core/materials.js';
import { StructuralPanel } from './ui/structural-panel.js';
import { AirflowPanel } from './ui/airflow-panel.js';
import { ThermalPanel } from './ui/thermal-panel.js';
import { Picker } from './ui/picker.js';
import { openPsim, isPsimName, saveDialog, clearLoaded, runDone, showBanner, gather, buildFile } from './ui/psim-io.js';
import { $, $$, h, num, nextFrame } from './ui/dom.js';

const app = {
  viewport: $('#viewport'),
  part: null,
  partInfo: '',
  units: 'mm',
  tab: 'part',
  get toMeters() {
    return UNITS[this.units].toMeters;
  },
  status(msg, kind = '') {
    const el = $('#status');
    el.textContent = msg;
    el.className = kind;
  },
  hint(text) {
    const el = $('#hint');
    el.hidden = !text;
    if (text) el.textContent = text;
  },
  busy: {
    show(text, onCancel = null) {
      $('#busy').hidden = false;
      app.viewer.busy = true;
      $('#busy-text').textContent = text;
      $('#busy-bar').style.width = '0';
      $('#busy-cancel').hidden = !onCancel;
      this.onCancel = onCancel;
    },
    progress(frac, text) {
      $('#busy-bar').style.width = `${Math.round(Math.max(0, Math.min(1, frac)) * 100)}%`;
      if (text) $('#busy-text').textContent = text;
    },
    hide() {
      $('#busy').hidden = true;
      app.viewer.busy = false;
      this.onCancel = null;
    },
  },
  setTab(name) {
    if (this.picker.active) this.picker.finish(false);
    const prev = this.tab;
    this.tab = name;
    for (const b of $$('.tabs button')) {
      b.classList.toggle('active', b.dataset.tab === name);
      b.setAttribute('aria-selected', String(b.dataset.tab === name));
    }
    for (const p of $$('.panel')) p.hidden = p.dataset.panel !== name;
    if (prev === 'structural' && name !== 'structural') this.structural.deactivate();
    if (prev === 'thermal' && name !== 'thermal') this.thermal.deactivate();
    if (prev === 'airflow' && name !== 'airflow') this.airflow.deactivate();
    $('#legend').hidden = true;
    $('#probe').hidden = true;
    if (name === 'structural') this.structural.activate();
    if (name === 'thermal') this.thermal.activate();
    if (name === 'airflow') this.airflow.activate();
    if (name === 'part') renderPartInfo();
  },
  setXRay(on) {
    $('#chk-xray').checked = on;
    this.viewer.setXRay(on);
  },
};

app.viewer = new Viewer($('#viewport'));
app.picker = new Picker(app);
app.structural = new StructuralPanel(app);
app.thermal = new ThermalPanel(app);
app.airflow = new AirflowPanel(app);
window.partsSim = app; // handy for debugging from the console
app.desktop = window.partsSimDesktop || null; // set when running as the desktop app

// ---------- part loading ----------

let partRequest = 0;

async function loadPart(src, { units = null, setup = null, airflow = null, info = '', request = ++partRequest } = {}) {
  if (request !== partRequest) return false;
  app.busy.show('Preparing mesh…');
  await nextFrame();
  if (request !== partRequest) return false;
  let part;
  try {
    part = buildPart(src, { faceAngle: Number($('#face-angle').value) });
  } catch (err) {
    app.busy.hide();
    app.status(`Could not process the mesh: ${err.message}`, 'error');
    return false;
  }
  return installPart(part, { units, setup, airflow, info });
}

/** Makes a built part the current one and resets every panel for it. */
function installPart(part, { units = null, setup = null, airflow = null, info = '' } = {}) {
  if (app.picker.active) app.picker.finish(false);
  app.structural.deactivate();
  app.thermal.deactivate();
  app.airflow.deactivate();
  app.part = part;
  app.partInfo = info;
  app.sourceFile = null;
  clearLoaded(app);
  if (units) setUnits(units, false);
  app.viewer.setPart(part);
  $('#empty-state').hidden = true;
  app.structural.reset(setup ? setup(part) : null);
  app.thermal.reset();
  app.airflow.reset(airflow);
  renderPartInfo();
  app.busy.hide();
  app.setTab(app.tab);
  document.title = `${part.name} — Parts Sim`;
  const size = part.bbox.size.map((s) => num(s)).join(' × ');
  app.status(`Loaded “${part.name}”: ${part.nTri.toLocaleString()} triangles, ${size} ${app.units}.`);
  return true;
}

/**
 * Open one file; `companions` are other files selected with it (a SolidWorks assembly's parts).
 * The desktop app passes readSibling so an assembly finds its parts in the same folder.
 */
async function importFromFile(file, companions = [], readSibling = null) {
  if (!file) return;
  const request = ++partRequest;
  app.busy.show(`Reading ${file.name}…`);
  await nextFrame();
  if (isPsimName(file.name)) {
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      await openPsim(app, bytes, file.name, async (part, o) => {
        if (request !== partRequest) return false;
        return installPart(part, o);
      });
    } catch (err) {
      app.status(`Could not open ${file.name}: ${err.message}`, 'error');
    }
    app.busy.hide();
    return;
  }
  try {
    const src = await importFile(file, { companions, readSibling });
    if (request !== partRequest) return;
    const loaded = await loadPart(src, { units: src.units, info: src.info, request });
    if (!loaded) return;
    if (/\.(step|stp|iges|igs|sldprt)$/i.test(file.name) && !companions.length) app.sourceFile = file;
    const notes = [...(src.warnings || [])];
    const matId = matchMaterial(src.material, MATERIALS);
    if (matId) {
      app.structural.setMaterial(matId);
      renderPartInfo();
      notes.unshift(`Material from SolidWorks: “${src.material}” → ${MATERIALS.find((m) => m.id === matId).name}.`);
    } else if (src.material) {
      notes.unshift(`SolidWorks material “${src.material}” is not in the library - pick the closest one or enter its properties (Bend & break tab).`);
    }
    if (!src.units) notes.unshift(`Loaded ${file.name}. Mesh files have no units - check the unit (Part tab) matches your model.`);
    if (notes.length) app.status(notes.join(' '), src.warnings?.length || !src.units ? 'warn' : '');
  } catch (err) {
    if (request !== partRequest) return;
    app.busy.hide();
    app.status(`Import failed: ${err.message}`, 'error');
  }
}

async function loadSample(id) {
  const s = SAMPLES.find((x) => x.id === id);
  if (!s) return;
  const request = ++partRequest;
  try {
    const src = { ...s.make(), name: s.name };
    const loaded = await loadPart(src, {
      units: 'mm',
      info: s.note,
      setup: (part) => ({ ...s.setup(part), material: s.material }),
      airflow: s.airflow || null,
      request,
    });
    if (!loaded) return;
    if (s.airflow && app.tab === 'part') app.setTab('airflow');
    else if (app.tab === 'part') app.setTab('structural');
    app.status(`${s.name}: ${s.note}. Press “${s.airflow ? 'Run airflow' : 'Run bend test'}”.`);
  } catch (err) {
    if (request !== partRequest) return;
    app.busy.hide();
    app.status(`Could not load sample: ${err.message}`, 'error');
  }
}

function renderPartInfo() {
  const p = app.part;
  const dl = $('#part-stats');
  $('#part-empty').hidden = !!p;
  dl.hidden = !p;
  $('#face-card').hidden = !p || p.brepFaces;
  if (!p) return;
  const u = app.units, m3 = app.toMeters ** 3;
  const mass = p.volume * m3 * app.structural.material.density;
  const rows = [
    ['Name', p.name],
    ['Source', app.partInfo || '–'],
    ['Size', `${p.bbox.size.map((s) => num(s)).join(' × ')} ${u}`],
    ['Volume', `${num(p.volume)} ${u}³`],
    ['Surface', `${num(p.area)} ${u}²`],
    ['Mass', `${num(mass * 1000)} g (${app.structural.material.name.split(' (')[0]})`],
    ['Triangles', p.nTri.toLocaleString()],
    ['Faces', `${p.faceCount.toLocaleString()}${p.brepFaces ? ' (CAD)' : ''}`],
  ];
  dl.replaceChildren(...rows.flatMap(([k, v]) => [h('dt', {}, k), h('dd', { title: v }, v)]));
  $('#target-size').placeholder = num(Math.max(...p.bbox.size));
  $('#size-unit').textContent = u;
}

function setUnits(u, notify = true) {
  app.units = u;
  for (const b of $$('#units-seg button')) b.classList.toggle('active', b.dataset.unit === u);
  if (notify) {
    app.structural.markStale();
    app.thermal.markStale();
    app.airflow.markDirty();
  }
  app.structural.updateMeshInfo();
  app.thermal.updateMeshInfo();
  renderPartInfo();
}

// ---------- toolbar & panels ----------

const fileInput = $('#file-input');
fileInput.accept = ACCEPT;
fileInput.multiple = true;
fileInput.addEventListener('change', () => {
  const { primary, companions } = primaryFile(fileInput.files);
  importFromFile(primary, companions);
  fileInput.value = '';
});
const openFiles = () => (app.desktop ? app.desktop.openDialog() : fileInput.click());
$('#btn-open').addEventListener('click', openFiles);
$('#empty-open').addEventListener('click', openFiles);

const dd = $('#samples-dd');
$('#samples-menu').append(
  ...SAMPLES.map((s) =>
    h('button', { role: 'menuitem', onclick: () => { dd.classList.remove('open'); loadSample(s.id); } },
      h('div.m-title', {}, s.name), h('div.m-note', {}, s.note))),
);
$('#btn-samples').addEventListener('click', (e) => { e.stopPropagation(); dd.classList.toggle('open'); });
$('#empty-sample').addEventListener('click', (e) => { e.stopPropagation(); dd.classList.add('open'); });
document.addEventListener('click', (e) => { if (!dd.contains(e.target)) dd.classList.remove('open'); });

for (const b of $$('[data-view]')) b.addEventListener('click', () => app.viewer.setView(b.dataset.view));
$('#chk-edges').addEventListener('change', (e) => app.viewer.setEdgesVisible(e.target.checked));
$('#chk-xray').addEventListener('change', (e) => app.viewer.setXRay(e.target.checked));
$('#btn-shot').addEventListener('click', async () => {
  const bg = getComputedStyle(document.documentElement).getPropertyValue('--vp-bottom').trim() || '#e9edf2';
  const png = app.viewer.screenshot(bg);
  const name = app.part?.name || 'parts-sim';
  if (app.desktop) {
    if (await app.desktop.saveImage(png, name)) app.status('Screenshot saved.');
    return;
  }
  h('a', { href: png, download: `${name}.png` }).click();
});
$('#btn-help').addEventListener('click', () => $('#help').showModal());
$('#btn-save').addEventListener('click', () => saveDialog(app));
// results loaded from a file stay on screen until something new is computed or the setup changes
app.runDone = (kind, now) => runDone(app, kind, now);
for (const id of ['#btn-run', '#btn-break', '#btn-th-run', '#btn-flow-run']) {
  $(id).addEventListener('click', () => { if (!app.rerunOf) clearLoaded(app); });
}
for (const panel of [app.structural, app.thermal]) {
  const markStale = panel.markStale.bind(panel);
  panel.markStale = (...args) => {
    if (app.loadedFile && !app.loadedFile.edited) { app.loadedFile.edited = true; showBanner(app); }
    return markStale(...args);
  };
}
$('#busy-cancel').addEventListener('click', () => app.busy.onCancel?.());

for (const b of $$('.tabs button')) b.addEventListener('click', () => app.setTab(b.dataset.tab));
for (const b of $$('#units-seg button')) b.addEventListener('click', () => setUnits(b.dataset.unit));
for (const b of $$('[data-rotate]')) {
  b.addEventListener('click', () => {
    const p = app.part;
    if (!p) return;
    const src = {
      positions: rotatePositions90(p.vertices, Number(b.dataset.rotate)),
      index: p.tris,
      faceIds: p.brepFaces ? p.faceOf : null,
      name: p.name,
    };
    const hadSetup = app.structural.fixtures.length || app.structural.loads.length;
    loadPart(src, { info: app.partInfo }).then((loaded) => {
      if (loaded && hadSetup) app.status('Part rotated - fixtures and loads were cleared.', 'warn');
    });
  });
}
// ---------- size ----------

function applyScale(s) {
  const p = app.part;
  if (!p) return app.status('Import a part or open a sample first.', 'error');
  if (!(s > 0) || !Number.isFinite(s) || s === 1) return;
  const size = Math.max(...p.bbox.size) * s;
  if (size > 1e7 || size < 1e-6) return app.status('That size is outside what the simulations can handle.', 'error');
  if (app.picker.active) app.picker.finish(false);
  const pivot = scalePart(p, s);
  app.structural.onPartScaled(s, pivot);
  app.thermal.onPartScaled(s, pivot);
  app.airflow.onPartScaled();
  const tab = app.tab;
  app.structural.deactivate();
  app.thermal.deactivate();
  app.airflow.deactivate();
  app.viewer.setPart(p);
  app.setTab(tab);
  renderPartInfo();
  $('#scale-factor').value = 1;
  app.status(`Scaled × ${num(s)}: now ${p.bbox.size.map((v) => num(v)).join(' × ')} ${app.units}. Fixtures and loads were kept; run the tests again.`);
}

app.applyScale = applyScale;

/** Save and open without dialogs (tests, scripts): bytes in, bytes out. */
app.psim = {
  async save(options = {}) {
    const draft = gather(app);
    const all = { geometry: 'quantised', setup: true, results: [...draft.results.keys()], cad: false, thumb: false, ...options };
    return (await buildFile(app, draft, all, null)).bytes;
  },
  open: (bytes, name = 'file.psim') => openPsim(app, bytes, name, async (part, o) => installPart(part, o)),
};

/** Replace the part with a generated mesh (e.g. an optimized shape), keeping the units. */
app.loadGeneratedPart = (src) => {
  const request = ++partRequest;
  loadPart({ positions: src.positions, index: src.index, faceIds: null, name: src.name }, { info: 'Generated by optimization', request }).then((loaded) => {
    if (loaded) app.status(`Loaded “${src.name}”. Add fixtures and loads, then run a static study to check it.`);
  });
};

$('#btn-scale').addEventListener('click', () => applyScale(Number($('#scale-factor').value)));
for (const b of $$('[data-scale]')) b.addEventListener('click', () => applyScale(Number(b.dataset.scale)));
$('#btn-size').addEventListener('click', () => {
  const target = Number($('#target-size').value);
  if (!app.part || !(target > 0)) return app.status('Enter the new length of the longest side.', 'error');
  applyScale(target / Math.max(...app.part.bbox.size));
});
for (const sel of ['#mat-change', '#th-mat-change']) {
  $(sel).addEventListener('click', () => {
    app.setTab('part');
    $('#material-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
    $('#mat-select').focus();
  });
}
app.onMaterialChange = () => {
  renderPartInfo();
  app.airflow.update();
  app.thermal?.renderMaterial();
};

const faceAngle = $('#face-angle');
faceAngle.addEventListener('input', () => {
  $('#face-angle-out').textContent = `${faceAngle.value}°`;
});
faceAngle.addEventListener('change', () => {
  if (!app.part || app.part.brepFaces) return;
  setSmoothFaces(app.part, Number(faceAngle.value));
  renderPartInfo();
});

// ---------- viewport interaction ----------

const probe = $('#probe');
app.viewer.handlers.move = (ev) => {
  if (!app.part) return;
  if (app.picker.active) return app.picker.move(ev);
  const panel = app.tab === 'structural' ? app.structural : app.tab === 'thermal' ? app.thermal : app.tab === 'airflow' ? app.airflow : null;
  if (!panel) return (probe.hidden = true);
  const hit = app.viewer.pickPart(ev);
  const text = hit && panel.probe(hit);
  probe.hidden = !text;
  if (text) {
    const r = app.viewport.getBoundingClientRect();
    probe.textContent = text;
    probe.style.left = `${ev.clientX - r.left + 14}px`;
    probe.style.top = `${ev.clientY - r.top + 14}px`;
  }
};
app.viewer.handlers.click = (ev, button) => {
  if (app.picker.active && button === 0) app.picker.click(ev);
};
app.viewer.handlers.leave = () => {
  probe.hidden = true;
  if (app.picker.active) app.viewer.clearHover();
};

const vp = app.viewport;
let dragDepth = 0;
vp.addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; vp.classList.add('dragging'); });
vp.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; vp.classList.remove('dragging'); } });
vp.addEventListener('dragover', (e) => e.preventDefault());
vp.addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  vp.classList.remove('dragging');
  const { primary, companions } = primaryFile(e.dataTransfer.files);
  importFromFile(primary, companions, siblingReader(primary));
});

/** Desktop app: lets an assembly read its part files from the folder it was opened from. */
function siblingReader(file) {
  const opened = app.desktop && file && (file.desktopPath || app.desktop.pathForFile(file));
  return opened ? (name) => app.desktop.readSibling(opened, name) : null;
}

window.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === 's') {
    e.preventDefault();
    if (app.part) saveDialog(app);
    return;
  }
  if (e.target.closest('input, select, textarea') || app.picker.active) return;
  if (e.key === 'f') app.viewer.setView('iso');
});

const params = new URLSearchParams(location.search);
if (params.get('sample')) loadSample(params.get('sample'));

// ---------- desktop app (Electron) ----------

if (app.desktop) {
  const d = app.desktop;
  document.documentElement.classList.add('desktop', `platform-${d.platform}`);
  d.onOpenFiles((files) => {
    const list = files.map((f) => Object.assign(new File([f.data], f.name), { desktopPath: f.path }));
    const { primary, companions } = primaryFile(list);
    importFromFile(primary, companions, siblingReader(primary));
  });
  const click = (sel) => $(sel).click();
  const toggle = (sel) => {
    const el = $(sel);
    el.checked = !el.checked;
    el.dispatchEvent(new Event('change'));
  };
  d.onCommand((command) => {
    const [kind, arg] = command.split(':');
    if (kind === 'sample') loadSample(arg);
    else if (kind === 'view') app.viewer.setView(arg);
    else if (command === 'toggle:edges') toggle('#chk-edges');
    else if (command === 'toggle:xray') toggle('#chk-xray');
    else if (command === 'screenshot') click('#btn-shot');
    else if (command === 'save:psim') { if (app.part) saveDialog(app); else app.status('Open a part or a sample first.', 'error'); }
    else if (command === 'help') $('#help').showModal();
    else if (kind === 'study') {
      app.setTab('structural');
      app.structural.setStudy(arg);
    } else if (command === 'run:study' || command === 'run:bend' || command === 'run:break') {
      if (command === 'run:study' && app.tab === 'thermal') click('#btn-th-run');
      else if (command === 'run:study' && app.tab === 'airflow') click('#btn-flow-run');
      else {
        app.setTab('structural');
        if (command === 'run:break') app.structural.setStudy('static');
        click(command === 'run:break' ? '#btn-break' : '#btn-run');
      }
    } else if (command === 'run:thermal') {
      app.setTab('thermal');
      click('#btn-th-run');
    } else if (command === 'run:airflow') {
      app.setTab('airflow');
      click('#btn-flow-run');
    }
  });
  d.ready();
}
