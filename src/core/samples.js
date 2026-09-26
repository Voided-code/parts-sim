// Built-in sample parts, each with a ready-made study setup.
import * as THREE from 'three';

function geometryData(geo) {
  const g = geo.index ? geo.toNonIndexed() : geo;
  return { positions: Float32Array.from(g.getAttribute('position').array), index: null, faceIds: null };
}

function extrude(shape, depth, curveSegments = 32) {
  return new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: false, curveSegments, steps: 1 });
}

/** Triangles whose centroid/normal pass `test`. */
export function selectTris(part, test) {
  const out = [];
  const V = part.vertices, T = part.tris, N = part.triNormal;
  const c = [0, 0, 0], n = [0, 0, 0];
  for (let t = 0; t < part.nTri; t++) {
    for (let d = 0; d < 3; d++) {
      c[d] = (V[3 * T[3 * t] + d] + V[3 * T[3 * t + 1] + d] + V[3 * T[3 * t + 2] + d]) / 3;
      n[d] = N[3 * t + d];
    }
    if (test(c, n, part.bbox)) out.push(t);
  }
  return Int32Array.from(out);
}

const patch = (part, test) => ({ tris: selectTris(part, test), clip: null });

/**
 * Ahmed body (1044 x 288 x 389 mm): all four nose edges rounded with R = 100 mm, sharp
 * longitudinal edges and a 25 degree rear slant. Built as a loft of rectangular sections
 * (each nose edge is a quarter cylinder, so a section at depth x is inset by R - sqrt(R^2 - (R - x)^2)).
 */
function ahmedBody() {
  const L = 1044, H = 288, W = 389, R = 100, sl = 222;
  const tan = Math.tan(THREE.MathUtils.degToRad(25));
  const xs = [];
  for (let i = 0; i <= 32; i++) xs.push(R * (1 - Math.cos((i / 32) * (Math.PI / 2)))); // dense near the nose tip
  xs.push(L - sl, L);
  const rings = xs.map((x) => {
    const d = x < R ? R - Math.sqrt(R * R - (R - x) ** 2) : 0;
    const top = x > L - sl ? H - (x - (L - sl)) * tan : H;
    const hw = W / 2 - d;
    return [[x, d, -hw], [x, d, hw], [x, top - d, hw], [x, top - d, -hw]];
  });
  const pos = [];
  const tri = (a, b, c) => pos.push(...a, ...b, ...c);
  for (let k = 0; k + 1 < rings.length; k++) {
    for (let e = 0; e < 4; e++) {
      const a = rings[k][e], b = rings[k][(e + 1) % 4], c = rings[k + 1][(e + 1) % 4], d = rings[k + 1][e];
      tri(a, d, c);
      tri(a, c, b);
    }
  }
  const [f, r] = [rings[0], rings[rings.length - 1]];
  tri(f[0], f[1], f[2]);
  tri(f[0], f[2], f[3]);
  tri(r[0], r[2], r[1]);
  tri(r[0], r[3], r[2]);
  return { positions: Float32Array.from(pos), index: null, faceIds: null };
}
const EPS = 0.02;

export const SAMPLES = [
  {
    id: 'beam',
    name: 'Cantilever beam',
    note: '200 x 20 x 10 mm steel bar, clamped at one end, 800 N at the tip',
    material: 'steel-1020',
    make: () => geometryData(new THREE.BoxGeometry(200, 20, 10)),
    setup: (part) => ({
      fixtures: [{ name: 'Clamped end', patches: [patch(part, (c, n, b) => c[0] < b.min[0] + EPS && n[0] < -0.9)] }],
      loads: [{ name: 'Tip load', type: 'force', magnitude: 800, dir: [0, -1, 0], patches: [patch(part, (c, n, b) => c[0] > b.max[0] - EPS && n[0] > 0.9)] }],
    }),
  },
  {
    id: 'lbracket',
    name: 'L-bracket (PLA print)',
    note: '3D-printed PLA shelf bracket, bolted base, 180 N pulling the top outward',
    material: 'pla',
    make: () => {
      const s = new THREE.Shape();
      const t = 8, r = 4;
      s.moveTo(0, 0);
      s.lineTo(80, 0);
      s.lineTo(80, t);
      s.lineTo(t + r, t);
      s.absarc(t + r, t + r, r, -Math.PI / 2, -Math.PI, true);
      s.lineTo(t, 60);
      s.lineTo(0, 60);
      s.lineTo(0, 0);
      return geometryData(extrude(s, 40));
    },
    setup: (part) => ({
      fixtures: [{ name: 'Bolted base', patches: [patch(part, (c, n, b) => c[1] < b.min[1] + EPS && n[1] < -0.9 && c[0] > b.min[0] + 30)] }],
      loads: [{ name: 'Pull on top', type: 'force', magnitude: 180, dir: [-1, 0, 0], patches: [patch(part, (c, n, b) => c[1] > b.max[1] - 12 && n[0] > 0.9 && c[0] < b.min[0] + 9)] }],
    }),
  },
  {
    id: 'bracket',
    name: 'Bolted mounting bracket',
    note: 'Aluminium 6061 plate with two bolt holes and a lightening slot, 250 N at the tip',
    material: 'al-6061',
    make: () => {
      const s = new THREE.Shape();
      s.moveTo(0, -25);
      s.lineTo(80, -25);
      s.lineTo(140, -12);
      s.lineTo(140, 12);
      s.lineTo(80, 25);
      s.lineTo(0, 25);
      s.lineTo(0, -25);
      for (const y of [-12, 12]) {
        const h = new THREE.Path();
        h.absarc(15, y, 5, 0, Math.PI * 2, true);
        s.holes.push(h);
      }
      const slot = new THREE.Path();
      slot.moveTo(50, -6);
      slot.lineTo(95, -6);
      slot.absarc(95, 0, 6, -Math.PI / 2, Math.PI / 2, false);
      slot.lineTo(50, 6);
      slot.absarc(50, 0, 6, Math.PI / 2, (3 * Math.PI) / 2, false);
      s.holes.push(slot);
      const g = extrude(s, 6, 40);
      g.rotateX(Math.PI / 2);
      return geometryData(g);
    },
    setup: (part) => {
      const x0 = part.bbox.min[0];
      const zc = (part.bbox.min[2] + part.bbox.max[2]) / 2;
      const inHole = (c) => [-12, 12].some((z) => Math.hypot(c[0] - (x0 + 15), c[2] - (zc + z)) < 5.4);
      return {
        fixtures: [{ name: 'Bolt holes', patches: [patch(part, (c, n) => inHole(c) && Math.abs(n[1]) < 0.5)] }],
        loads: [{ name: 'Tip load', type: 'force', magnitude: 250, dir: [0, -1, 0], patches: [patch(part, (c, n, b) => c[0] > b.max[0] - EPS && n[0] > 0.9)] }],
      };
    },
  },
  {
    id: 'wrench',
    name: 'Open-end wrench',
    note: '13 mm Cr-V spanner gripping a nut, 500 N pushed at the end of the handle',
    material: 'steel-alloy',
    make: () => {
      const R = 17, w = 6.5, hw = 7, L = 150;
      const xj = -Math.sqrt(R * R - w * w);
      const pts = [];
      const arc = (cx, cy, r, a0, a1, n) => {
        for (let i = 0; i <= n; i++) {
          const a = a0 + ((a1 - a0) * i) / n;
          pts.push(new THREE.Vector2(cx + r * Math.cos(a), cy + r * Math.sin(a)));
        }
      };
      const tip = Math.atan2(w, xj); // upper jaw tip angle (~157 deg)
      const neck = Math.asin(hw / R); // where the handle meets the head (~24 deg)
      arc(0, 0, R, 2 * Math.PI - tip, 2 * Math.PI - neck, 28); // lower head arc
      arc(L, 0, hw, -Math.PI / 2, Math.PI / 2, 18); // rounded handle end
      arc(0, 0, R, neck, tip, 28); // upper head arc
      pts.push(new THREE.Vector2(2, w), new THREE.Vector2(2, -w)); // jaw opening
      const s = new THREE.Shape(pts);
      const g = extrude(s, 5, 48);
      g.rotateX(Math.PI / 2);
      return geometryData(g);
    },
    setup: (part) => {
      const x0 = part.bbox.min[0];
      const zc = (part.bbox.min[2] + part.bbox.max[2]) / 2;
      return {
        fixtures: [{
          name: 'Nut flats',
          patches: [patch(part, (c, n) => c[0] < x0 + 18.5 && Math.abs(Math.abs(c[2] - zc) - 6.5) < 0.3 && Math.abs(n[2]) > 0.9)],
        }],
        loads: [{
          name: 'Hand force', type: 'force', magnitude: 500, dir: [0, 0, 1],
          patches: [patch(part, (c, n, b) => c[0] > b.max[0] - 25 && n[2] < -0.7)],
        }],
      };
    },
  },
  {
    id: 'hook',
    name: 'Crane hook',
    note: '12 mm steel rod bent into a J, hung from the top, 1.5 kN load in the throat',
    material: 'steel-1020',
    make: () => {
      const pts = [];
      for (let y = 90; y > 30; y -= 6) pts.push(new THREE.Vector3(0, y, 0));
      for (let a = 180; a <= 400; a += 8) {
        const r = THREE.MathUtils.degToRad(a);
        pts.push(new THREE.Vector3(22 + 22 * Math.cos(r), 30 + 22 * Math.sin(r), 0));
      }
      const path = new THREE.CatmullRomCurve3(pts);
      const circle = new THREE.Shape();
      circle.absarc(0, 0, 6, 0, Math.PI * 2, false);
      const g = new THREE.ExtrudeGeometry(circle, { steps: 160, bevelEnabled: false, extrudePath: path, curveSegments: 20 });
      return geometryData(g);
    },
    setup: (part) => {
      const b = part.bbox;
      const cx = b.min[0] + 6 + 22; // hook bend centre (x)
      return {
        fixtures: [{ name: 'Top eye', patches: [patch(part, (c, n) => c[1] > b.max[1] - 0.5 && n[1] > 0.9)] }],
        loads: [{
          name: 'Hanging load', type: 'force', magnitude: 1500, dir: [0, -1, 0],
          patches: [patch(part, (c, n) => c[1] < b.min[1] + 16 && Math.abs(c[0] - cx) < 7 && n[1] > 0.6)],
        }],
      };
    },
  },
  {
    id: 'wing',
    name: 'Wing (NACA 2412)',
    note: '100 mm chord, 300 mm span at 6° - see the flow and tip vortices, then apply the wind load',
    material: 'al-6061',
    airflow: { yaw: 90, pitch: 6, speed: 30 },
    make: () => {
      const m = 0.02, p = 0.4, t = 0.12, c = 100, n = 60;
      const up = [], lo = [];
      for (let i = 0; i <= n; i++) {
        const x = (1 - Math.cos((i / n) * Math.PI)) / 2;
        const yt = 5 * t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4);
        const yc = x < p ? (m / (p * p)) * (2 * p * x - x * x) : (m / (1 - p) ** 2) * (1 - 2 * p + 2 * p * x - x * x);
        const dy = x < p ? ((2 * m) / (p * p)) * (p - x) : ((2 * m) / (1 - p) ** 2) * (p - x);
        const th = Math.atan(dy);
        up.push(new THREE.Vector2((x - yt * Math.sin(th)) * c, (yc + yt * Math.cos(th)) * c));
        lo.push(new THREE.Vector2((x + yt * Math.sin(th)) * c, (yc - yt * Math.cos(th)) * c));
      }
      const outline = [...lo, ...up.reverse().slice(1, -1)];
      const s = new THREE.Shape(outline);
      return geometryData(extrude(s, 300));
    },
    setup: (part) => ({
      fixtures: [{ name: 'Wing root', patches: [patch(part, (c, n, b) => c[2] < b.min[2] + EPS && n[2] < -0.9)] }],
      loads: [{ name: 'Lift (distributed)', type: 'pressure', magnitude: 0.004, dir: [0, 1, 0], patches: [patch(part, (c, n) => n[1] < -0.3)] }],
    }),
  },
  {
    id: 'ahmed',
    name: 'Ahmed body (car)',
    note: 'Standard car aerodynamics benchmark: rounded nose, 25° rear slant',
    material: 'al-6061',
    airflow: { yaw: 90, pitch: 0, speed: 40 },
    make: () => ahmedBody(),
    setup: (part) => ({
      fixtures: [{ name: 'Underside', patches: [patch(part, (c, n, b) => c[1] < b.min[1] + EPS && n[1] < -0.9)] }],
      loads: [],
    }),
  },
];
