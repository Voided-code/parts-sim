// Three.js viewport: part display, result coloring, deformation, overlays and picking.
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { computeBoundsTree, disposeBoundsTree, acceleratedRaycast } from 'three-mesh-bvh';
import { makeColormapTexture } from './colormap.js';

THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

const UP = new THREE.Vector3(0, 1, 0);

export class Viewer {
  constructor(container) {
    this.container = container;
    const renderer = (this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true }));
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0x000000, 0);
    renderer.autoClear = false;
    container.appendChild(renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1e6);
    this.camera.position.set(200, 160, 240);
    this.scene.add(this.camera);
    this.controls = new OrbitControls(this.camera, renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.screenSpacePanning = true;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a94a3, 2.4));
    const head = new THREE.DirectionalLight(0xffffff, 2.2);
    head.position.set(0.3, 0.4, 1);
    this.camera.add(head);
    this.camera.add(head.target);
    const top = new THREE.DirectionalLight(0xffffff, 0.6);
    top.position.set(0.2, 1, 0.3);
    this.scene.add(top);

    this.partGroup = new THREE.Group();
    this.overlayGroup = new THREE.Group();
    this.hoverGroup = new THREE.Group();
    this.markerGroup = new THREE.Group();
    this.voxelGroup = new THREE.Group();
    this.flowGroup = new THREE.Group();
    this.shapeGroup = new THREE.Group();
    this.scene.add(this.partGroup, this.overlayGroup, this.hoverGroup, this.markerGroup, this.voxelGroup, this.flowGroup, this.shapeGroup);

    this.grid = new THREE.GridHelper(400, 20, 0x8b98a8, 0xc6ced8);
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.55;
    this.scene.add(this.grid);

    this.plainMaterial = new THREE.MeshStandardMaterial({
      color: 0xc9d2dc, metalness: 0, roughness: 0.55, side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    });
    this.resultMaterial = new THREE.MeshStandardMaterial({
      color: 0xffffff, metalness: 0, roughness: 0.85, side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    });
    this.textures = new Map();
    this.edgeMaterial = new THREE.LineBasicMaterial({ color: 0x1b2330, transparent: true, opacity: 0.55 });

    this.raycaster = new THREE.Raycaster();
    this.raycaster.firstHitOnly = true;
    this.pointer = new THREE.Vector2();
    this.frameCallbacks = new Set();
    this.draggables = [];
    this.handlers = { move: null, click: null, leave: null };
    this.lastFrame = performance.now();

    this.initTriad();
    this.initPointer();
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    renderer.setAnimationLoop(() => this.frame());
  }

  // ---------- part ----------

  setPart(part) {
    this.clearPart();
    this.part = part;
    const d = part.display;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(d.position), 3));
    g.setAttribute('normal', new THREE.BufferAttribute(d.normal, 3));
    const uv = new Float32Array((d.position.length / 3) * 2);
    for (let i = 0; i < uv.length; i += 2) { uv[i] = 0.5; uv[i + 1] = 0.25; }
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    g.computeBoundingSphere();
    // Picking, face patches and scalar probes all address the original triangles.
    // Keep their indices stable while the BVH partitions its acceleration data.
    g.computeBoundsTree({ indirect: true });
    this.mesh = new THREE.Mesh(g, this.plainMaterial);
    this.partGroup.add(this.mesh);

    const E = part.edges;
    const ep = new Float32Array(E.length * 3);
    for (let i = 0; i < E.length; i++) ep.set(part.vertices.subarray(3 * E[i], 3 * E[i] + 3), 3 * i);
    const eg = new THREE.BufferGeometry();
    eg.setAttribute('position', new THREE.BufferAttribute(ep, 3));
    this.edges = new THREE.LineSegments(eg, this.edgeMaterial);
    this.edges.visible = this.showEdges !== false;
    this.partGroup.add(this.edges);

    const size = part.bbox.diag;
    this.scene.remove(this.grid);
    this.grid.geometry.dispose();
    this.grid.material.dispose();
    const gs = Math.pow(10, Math.ceil(Math.log10(size * 2)));
    this.grid = new THREE.GridHelper(gs, 20, 0x8b98a8, 0xc6ced8);
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.5;
    this.grid.position.y = -size * 0.002;
    this.scene.add(this.grid);
    this.camera.near = size / 1000;
    this.camera.far = size * 200;
    this.camera.updateProjectionMatrix();
    this.setView('iso');
  }

  clearPart() {
    if (this.mesh) {
      this.mesh.geometry.disposeBoundsTree();
      this.mesh.geometry.dispose();
      this.edges.geometry.dispose();
      this.partGroup.clear();
    }
    this.mesh = null;
    this.part = null;
    this.clearOverlays();
    this.clearHover();
    this.clearMarker();
    this.clearVoxels();
  }

  setEdgesVisible(v) {
    this.showEdges = v;
    if (this.edges) this.edges.visible = v;
  }

  setXRay(on) {
    for (const m of [this.plainMaterial, this.resultMaterial]) {
      m.transparent = on;
      m.opacity = on ? (m === this.plainMaterial ? 0.28 : 0.45) : 1;
      m.depthWrite = !on;
      m.needsUpdate = true;
    }
  }

  /** Per-unique-vertex values -> colors. Pass null for the plain CAD look. NaN = no data (grey). */
  setScalars(values, { min = 0, max = 1, bands = 0, reverse = false, colormap = 'rainbow' } = {}) {
    if (!this.mesh) return;
    if (!values) {
      this.mesh.material = this.plainMaterial;
      this.scalars = null;
      return;
    }
    const key = `${bands}:${reverse}:${colormap}`;
    if (!this.textures.has(key)) this.textures.set(key, makeColormapTexture(bands, reverse, colormap));
    this.resultMaterial.map = this.textures.get(key);
    this.resultMaterial.needsUpdate = true;
    this.mesh.material = this.resultMaterial;
    const uv = this.mesh.geometry.attributes.uv.array;
    const src = this.part.display.srcVert;
    const span = max - min || 1;
    for (let i = 0; i < src.length; i++) {
      const v = values[src[i]];
      if (Number.isNaN(v)) { uv[2 * i] = 0.5; uv[2 * i + 1] = 0.75; }
      else { uv[2 * i] = Math.min(1, Math.max(0, (v - min) / span)); uv[2 * i + 1] = 0.25; }
    }
    this.mesh.geometry.attributes.uv.needsUpdate = true;
    this.scalars = values;
  }

  /** Displace by per-unique-vertex vectors (model units) times `scale`; null restores the shape. */
  setDeformation(disp, scale = 1) {
    if (!this.mesh) return;
    const pos = this.mesh.geometry.attributes.position.array;
    const base = this.part.display.position;
    const src = this.part.display.srcVert;
    if (!disp || scale === 0) pos.set(base);
    else {
      for (let i = 0; i < src.length; i++) {
        const v = 3 * src[i];
        pos[3 * i] = base[3 * i] + scale * disp[v];
        pos[3 * i + 1] = base[3 * i + 1] + scale * disp[v + 1];
        pos[3 * i + 2] = base[3 * i + 2] + scale * disp[v + 2];
      }
    }
    this.mesh.geometry.attributes.position.needsUpdate = true;
    this.mesh.geometry.computeBoundingSphere();
    this.mesh.geometry.boundsTree.refit();
    const E = this.part.edges, V = this.part.vertices;
    const ep = this.edges.geometry.attributes.position.array;
    for (let i = 0; i < E.length; i++) {
      const v = 3 * E[i];
      for (let d = 0; d < 3; d++) ep[3 * i + d] = V[v + d] + (disp && scale ? scale * disp[v + d] : 0);
    }
    this.edges.geometry.attributes.position.needsUpdate = true;
    this.edges.geometry.computeBoundingSphere();
  }

  /** An extra solid shape drawn with the part (e.g. an optimized design); null removes it. */
  setShape(positions, index = null, color = 0x3f7fd9) {
    disposeTree(this.shapeGroup);
    this.shapeGroup.clear();
    if (!positions || !positions.length) return;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(positions), 3));
    if (index) g.setIndex(new THREE.BufferAttribute(Uint32Array.from(index), 1));
    g.computeVertexNormals();
    g.computeBoundingSphere();
    const mat = new THREE.MeshStandardMaterial({ color, metalness: 0, roughness: 0.5, side: THREE.DoubleSide });
    this.shapeGroup.add(new THREE.Mesh(g, mat));
  }

  // ---------- overlays ----------

  clearOverlays() {
    disposeTree(this.overlayGroup);
    this.overlayGroup.clear();
    this.draggables = [];
  }

  /** Semi-transparent highlight of a set of part triangles. */
  patchMesh(tris, color, opacity = 0.55) {
    const part = this.part;
    const pos = new Float32Array(tris.length * 9);
    const off = part.bbox.diag * 0.0008;
    for (let q = 0; q < tris.length; q++) {
      const t = tris[q];
      for (let c = 0; c < 3; c++) {
        const v = part.tris[3 * t + c];
        for (let d = 0; d < 3; d++) pos[9 * q + 3 * c + d] = part.vertices[3 * v + d] + part.triNormal[3 * t + d] * off;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const m = new THREE.MeshBasicMaterial({
      color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false,
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
    });
    return new THREE.Mesh(g, m);
  }

  /** Arrow whose tip sits at `tip` and points along `dir`. */
  arrow(tip, dir, length, color, radius = length * 0.035) {
    const group = new THREE.Group();
    const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.4, metalness: 0.1 });
    const head = length * 0.28;
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, length - head, 12), mat);
    shaft.position.y = -(head + (length - head) / 2);
    const cone = new THREE.Mesh(new THREE.ConeGeometry(radius * 2.6, head, 18), mat);
    cone.position.y = -head / 2;
    group.add(shaft, cone);
    group.position.copy(tip);
    // local +Y runs from the tail (y = -length) to the tip (origin)
    group.quaternion.setFromUnitVectors(UP, new THREE.Vector3().copy(dir).normalize());
    group.userData = { tip: tip.clone(), dir: dir.clone().normalize(), length };
    return group;
  }

  /** Small cone "anchor" glyph pointing into the surface (fixtures). */
  anchor(point, normal, size, color) {
    const m = new THREE.Mesh(
      new THREE.ConeGeometry(size * 0.35, size, 4),
      new THREE.MeshStandardMaterial({ color, roughness: 0.5 }),
    );
    const n = new THREE.Vector3().fromArray(normal).normalize();
    m.quaternion.setFromUnitVectors(UP, n.clone().negate());
    m.position.copy(point).addScaledVector(n, size / 2);
    return m;
  }

  /** Register a mesh that can be dragged; onDrag receives the pointer's point on a camera-facing plane through `pivot`. */
  addDraggable(object, pivot, onDrag, onEnd) {
    this.draggables.push({ object, pivot: pivot.clone(), onDrag, onEnd });
  }

  // ---------- hover helpers ----------

  clearHover() {
    disposeTree(this.hoverGroup);
    this.hoverGroup.clear();
    this.hoverKey = null;
  }

  showHoverPatch(key, tris, color) {
    if (this.hoverKey === key) return;
    this.clearHover();
    this.hoverKey = key;
    if (tris && tris.length) this.hoverGroup.add(this.patchMesh(tris, color, 0.45));
  }

  showBrush(point, normal, radius, color) {
    this.clearHover();
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(radius * 0.92, radius, 48),
      new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, transparent: true, opacity: 0.9, depthTest: false }),
    );
    const disc = new THREE.Mesh(
      new THREE.CircleGeometry(radius, 48),
      new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, transparent: true, opacity: 0.18, depthWrite: false }),
    );
    const g = new THREE.Group();
    g.add(ring, disc);
    g.position.copy(point).addScaledVector(normal, this.part.bbox.diag * 0.001);
    g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
    g.renderOrder = 10;
    this.hoverGroup.add(g);
    this.hoverKey = null;
  }

  // ---------- markers ----------

  setMarker(point, label, color = 0xff2a2a) {
    this.clearMarker();
    const r = this.part.bbox.diag * 0.012;
    const s = new THREE.Mesh(
      new THREE.SphereGeometry(r, 24, 16),
      new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.95 }),
    );
    const halo = new THREE.Mesh(
      new THREE.SphereGeometry(r, 24, 16),
      new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.35 }),
    );
    s.renderOrder = halo.renderOrder = 20;
    s.position.copy(point);
    halo.position.copy(point);
    this.markerGroup.add(s, halo);
    this.marker = { point: point.clone(), halo, label };
    if (!this.markerEl) {
      this.markerEl = document.createElement('div');
      this.markerEl.className = 'marker-label';
      this.container.appendChild(this.markerEl);
    }
    this.markerEl.textContent = label;
    this.markerEl.style.display = '';
  }

  moveMarker(point) {
    if (!this.marker) return;
    this.marker.point.copy(point);
    for (const o of this.markerGroup.children) o.position.copy(point);
  }

  clearMarker() {
    disposeTree(this.markerGroup);
    this.markerGroup.clear();
    this.marker = null;
    if (this.markerEl) this.markerEl.style.display = 'none';
  }

  // ---------- voxels ----------

  /** Instanced cubes at `centers` (xyz, world). */
  setVoxels(name, centers, size, color, opacity = 1) {
    const old = this.voxelGroup.getObjectByName(name);
    if (old) {
      old.geometry.dispose();
      old.material.dispose();
      this.voxelGroup.remove(old);
    }
    const n = centers.length / 3;
    if (!n) return;
    const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.6, transparent: opacity < 1, opacity });
    const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(size, size, size), mat, n);
    mesh.name = name;
    const m = new THREE.Matrix4();
    for (let i = 0; i < n; i++) {
      m.makeTranslation(centers[3 * i], centers[3 * i + 1], centers[3 * i + 2]);
      mesh.setMatrixAt(i, m);
    }
    mesh.instanceMatrix.needsUpdate = true;
    this.voxelGroup.add(mesh);
  }

  clearVoxels() {
    disposeTree(this.voxelGroup);
    this.voxelGroup.clear();
  }

  // ---------- camera ----------

  setView(name) {
    if (!this.part) return;
    const s = this.mesh.geometry.boundingSphere;
    const center = s.center.clone();
    const vertical = THREE.MathUtils.degToRad(this.camera.fov / 2);
    const horizontal = Math.atan(Math.tan(vertical) * this.camera.aspect);
    const dist = (s.radius / Math.sin(Math.min(vertical, horizontal))) * 1.08;
    const dirs = {
      iso: [1, 0.75, 1.25], front: [0, 0, 1], back: [0, 0, -1], right: [1, 0, 0],
      left: [-1, 0, 0], top: [0, 1, 0], bottom: [0, -1, 0],
    };
    const d = new THREE.Vector3(...(dirs[name] || dirs.iso)).normalize();
    this.camera.up.set(0, 1, 0);
    if (name === 'top') this.camera.up.set(0, 0, -1);
    if (name === 'bottom') this.camera.up.set(0, 0, 1);
    this.camera.position.copy(center).addScaledVector(d, dist);
    this.controls.target.copy(center);
    this.camera.lookAt(center);
    this.controls.update();
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Screen position (px, relative to the container) of a world point. */
  project(p) {
    const v = p.clone().project(this.camera);
    return {
      x: ((v.x + 1) / 2) * this.container.clientWidth,
      y: ((1 - v.y) / 2) * this.container.clientHeight,
      visible: v.z < 1 && v.z > -1,
    };
  }

  // ---------- picking ----------

  setPointer(ev) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
  }

  pickPart(ev) {
    if (!this.mesh) return null;
    this.setPointer(ev);
    const hit = this.raycaster.intersectObject(this.mesh, false)[0];
    if (!hit) return null;
    const t = hit.faceIndex;
    const n = new THREE.Vector3().fromArray(this.part.triNormal, 3 * t);
    // barycentric coordinates on the (possibly deformed) display triangle
    const pos = this.mesh.geometry.attributes.position;
    const a = new THREE.Vector3().fromBufferAttribute(pos, 3 * t);
    const b = new THREE.Vector3().fromBufferAttribute(pos, 3 * t + 1);
    const c = new THREE.Vector3().fromBufferAttribute(pos, 3 * t + 2);
    const bary = new THREE.Triangle(a, b, c).getBarycoord(hit.point, new THREE.Vector3());
    // undeformed location of the hit
    const V = this.part.vertices, T = this.part.tris;
    const rest = new THREE.Vector3();
    for (let k = 0; k < 3; k++) {
      const w = [bary.x, bary.y, bary.z][k];
      rest.x += w * V[3 * T[3 * t + k]];
      rest.y += w * V[3 * T[3 * t + k] + 1];
      rest.z += w * V[3 * T[3 * t + k] + 2];
    }
    return { point: hit.point.clone(), rest, tri: t, normal: n, bary, face: this.part.faceOf[t] };
  }

  initPointer() {
    const el = this.renderer.domElement;
    let down = null;
    let drag = null;
    el.addEventListener('pointerdown', (ev) => {
      down = { x: ev.clientX, y: ev.clientY, button: ev.button };
      if (ev.button !== 0 || !this.draggables.length) return;
      this.setPointer(ev);
      const objs = this.draggables.map((d) => d.object);
      const hit = this.raycaster.intersectObjects(objs, true)[0];
      if (!hit) return;
      drag = this.draggables.find((d) => d.object === hit.object || d.object.getObjectById(hit.object.id));
      if (drag) {
        this.controls.enabled = false;
        el.setPointerCapture(ev.pointerId);
        const normal = this.camera.getWorldDirection(new THREE.Vector3());
        drag.plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, drag.pivot);
      }
    });
    el.addEventListener('pointermove', (ev) => {
      if (drag) {
        this.setPointer(ev);
        const p = new THREE.Vector3();
        if (this.raycaster.ray.intersectPlane(drag.plane, p)) drag.onDrag(p);
        return;
      }
      if (ev.buttons) return;
      this.handlers.move?.(ev);
    });
    el.addEventListener('pointerup', (ev) => {
      if (drag) {
        drag.onEnd?.();
        drag = null;
        this.controls.enabled = true;
        down = null;
        return;
      }
      if (down && Math.hypot(ev.clientX - down.x, ev.clientY - down.y) < 5) this.handlers.click?.(ev, down.button);
      down = null;
    });
    el.addEventListener('pointerleave', () => this.handlers.leave?.());
    el.addEventListener('pointercancel', () => {
      drag?.onEnd?.();
      drag = null;
      down = null;
      this.controls.enabled = true;
    });
    el.addEventListener('contextmenu', (ev) => ev.preventDefault());
  }

  // ---------- triad & loop ----------

  initTriad() {
    this.triadScene = new THREE.Scene();
    this.triadCam = new THREE.OrthographicCamera(-1.6, 1.6, 1.6, -1.6, 0.1, 10);
    const axes = [
      [new THREE.Vector3(1, 0, 0), 0xe5484d, 'X'],
      [new THREE.Vector3(0, 1, 0), 0x30a46c, 'Y'],
      [new THREE.Vector3(0, 0, 1), 0x3e63dd, 'Z'],
    ];
    for (const [dir, color, label] of axes) {
      this.triadScene.add(new THREE.ArrowHelper(dir, new THREE.Vector3(), 1, color, 0.3, 0.16));
      const c = document.createElement('canvas');
      c.width = c.height = 64;
      const ctx = c.getContext('2d');
      ctx.fillStyle = '#' + color.toString(16).padStart(6, '0');
      ctx.font = 'bold 44px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, 32, 34);
      const tex = new THREE.CanvasTexture(c);
      tex.colorSpace = THREE.SRGBColorSpace;
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
      sp.scale.setScalar(0.45);
      sp.position.copy(dir).multiplyScalar(1.3);
      this.triadScene.add(sp);
    }
  }

  onFrame(cb) {
    this.frameCallbacks.add(cb);
    return () => this.frameCallbacks.delete(cb);
  }

  frame() {
    const now = performance.now();
    const dt = Math.min(0.1, (now - this.lastFrame) / 1000);
    this.lastFrame = now;
    this.controls.update();
    for (const cb of this.frameCallbacks) cb(dt);
    if (this.marker) {
      const s = 1 + 0.6 * (0.5 + 0.5 * Math.sin(performance.now() / 220));
      this.marker.halo.scale.setScalar(s * 1.6);
      const p = this.project(this.marker.point);
      this.markerEl.style.transform = `translate(${p.x + 14}px, ${p.y - 14}px)`;
      this.markerEl.style.display = p.visible ? '' : 'none';
    }
    const r = this.renderer;
    const w = this.container.clientWidth, h = this.container.clientHeight;
    r.setViewport(0, 0, w, h);
    r.setScissorTest(false);
    r.clear();
    r.render(this.scene, this.camera);
    // orientation triad in the bottom-left corner
    const s = 96;
    r.clearDepth();
    r.setScissorTest(true);
    r.setScissor(8, 8, s, s);
    r.setViewport(8, 8, s, s);
    this.triadCam.position.copy(this.camera.position).sub(this.controls.target).normalize().multiplyScalar(4);
    this.triadCam.up.copy(this.camera.up);
    this.triadCam.lookAt(0, 0, 0);
    r.render(this.triadScene, this.triadCam);
    r.setScissorTest(false);
  }

  /** PNG data URL of the viewport over the page background color. */
  screenshot(background = '#e9edf2') {
    const src = this.renderer.domElement;
    const c = document.createElement('canvas');
    c.width = src.width;
    c.height = src.height;
    const ctx = c.getContext('2d');
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(src, 0, 0);
    return c.toDataURL('image/png');
  }
}

function disposeTree(obj) {
  obj.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
        m.map?.dispose?.();
        m.dispose();
      }
    }
  });
}
