// Airflow validation cases: simple shapes with published drag, lift and skin-friction data.
//
// Units: parts are in millimetres, speeds in m/s, forces in newtons. The air is the ISA standard
// atmosphere at sea level, 15 C (density 1.225 kg/m3, dynamic viscosity 1.789e-5 Pa s). For each case
// the Reynolds number uses the named reference length L (Re = U L / nu) and the coefficients the named
// reference area A (Cd = F / (0.5 rho U^2 A)).
import * as THREE from 'three';
import { ahmedBody } from '../core/samples.js';

export const AIR = { density: 1.225, viscosity: 1.789e-5 };

/** Kinematic viscosity of air (m^2/s) at a given density (the viscosity barely changes with it). */
export const airNu = (density = AIR.density) => AIR.viscosity / density;

function positionsOf(geometry) {
  const g = geometry.index ? geometry.toNonIndexed() : geometry;
  const positions = Float32Array.from(g.getAttribute('position').array);
  if (g !== geometry) g.dispose();
  geometry.dispose();
  return positions;
}

/** NACA 4-digit section (chord c, cosine spacing) as a closed outline, leading edge at x = 0. */
export function nacaOutline(code, c, n = 80) {
  const m = Number(code[0]) / 100, p = Number(code[1]) / 10, t = Number(code.slice(2)) / 100;
  const up = [], lo = [];
  for (let i = 0; i <= n; i++) {
    const x = (1 - Math.cos((i / n) * Math.PI)) / 2;
    // closed trailing edge (-0.1036)
    const yt = 5 * t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4);
    let yc = 0, dy = 0;
    if (m > 0) {
      yc = x < p ? (m / (p * p)) * (2 * p * x - x * x) : (m / (1 - p) ** 2) * (1 - 2 * p + 2 * p * x - x * x);
      dy = x < p ? ((2 * m) / (p * p)) * (p - x) : ((2 * m) / (1 - p) ** 2) * (p - x);
    }
    const th = Math.atan(dy);
    up.push(new THREE.Vector2((x - yt * Math.sin(th)) * c, (yc + yt * Math.cos(th)) * c));
    lo.push(new THREE.Vector2((x + yt * Math.sin(th)) * c, (yc - yt * Math.cos(th)) * c));
  }
  return [...lo, ...up.reverse().slice(1, -1)];
}

/** A wing section extruded along z (span), rotated nose-up by `alphaDeg` about its quarter chord. */
function wingSection(code, chord, span, alphaDeg) {
  const g = new THREE.ExtrudeGeometry(new THREE.Shape(nacaOutline(code, chord)), { depth: span, bevelEnabled: false, curveSegments: 1, steps: 1 });
  g.translate(-0.25 * chord, 0, -span / 2);
  g.rotateZ(THREE.MathUtils.degToRad(-alphaDeg)); // nose up for a wind along +x (trailing edge down)
  return positionsOf(g);
}

const sphere = (d) => positionsOf(new THREE.SphereGeometry(d / 2, 96, 64));
const box = (x, y, z) => positionsOf(new THREE.BoxGeometry(x, y, z));

/**
 * Validation cases. make() returns part positions (mm, triangles). The wind blows along +x unless
 * `wind` says otherwise; lattice y is up. `ref` holds the published values the result is judged by.
 * `lref` / `aref` are in metres and square metres. `tunnel` holds options for the wind tunnel
 * (ground plane, periodic span) where the case needs them.
 */
export const CASES = [
  {
    id: 'plate',
    name: 'Square plate across the flow',
    note: '100 x 100 mm plate, 5 mm thick, face-on at 20 m/s',
    make: () => box(5, 100, 100),
    speed: 20, lref: 0.1, aref: 0.01, across: 40,
    ref: { cd: 1.17, cdRange: [1.1, 1.2], source: 'Hoerner, Fluid-Dynamic Drag (1965), ch. 3: square plate normal to the flow, Re > 1e3' },
  },
  {
    id: 'cube',
    name: 'Cube, face-on',
    note: '100 mm cube at 20 m/s',
    make: () => box(100, 100, 100),
    speed: 20, lref: 0.1, aref: 0.01, across: 40,
    ref: { cd: 1.05, cdRange: [1.0, 1.1], source: 'Hoerner (1965), ch. 3: cube face-on, Re > 1e4' },
  },
  {
    id: 'sphere',
    name: 'Sphere (subcritical)',
    note: '100 mm sphere at 20 m/s (Re 1.4e5; the drag crisis starts near 3.5e5)',
    make: () => sphere(100),
    speed: 20, lref: 0.1, aref: Math.PI * 0.05 ** 2, across: 40,
    ref: { cd: 0.47, cdRange: [0.4, 0.5], source: 'Achenbach, J. Fluid Mech. 54 (1972): Cd 0.4-0.5 for Re 1e4-2e5' },
  },
  {
    id: 'ahmed25',
    name: 'Ahmed body, 25° slant',
    note: '1044 mm long, 50 mm above a moving ground, 40 m/s (Re 2.9e6 on the length); also the app sample',
    make: () => ahmedBody(25).positions,
    speed: 40, lref: 1.044, aref: 0.389 * 0.288, across: 100, acrossAxis: 'length',
    tunnel: { ground: 50 },
    ref: { cd: 0.285, cdRange: [0.25, 0.33], source: 'Ahmed, Ramm & Faltin, SAE 840300 (1984); Lienhart & Becker, SAE 2003-01-0656' },
  },
  {
    id: 'ahmed35',
    name: 'Ahmed body, 35° slant',
    note: '1044 mm long, 50 mm above a moving ground, 40 m/s',
    make: () => ahmedBody(35).positions,
    speed: 40, lref: 1.044, aref: 0.389 * 0.288, across: 100, acrossAxis: 'length',
    tunnel: { ground: 50 },
    ref: { cd: 0.257, cdRange: [0.22, 0.30], source: 'Ahmed, Ramm & Faltin, SAE 840300 (1984)' },
  },
  {
    id: 'naca0012',
    name: 'NACA 0012 section, lift slope',
    note: '300 mm chord at 0°, 4° and 8°, 50 m/s (Re 1.0e6), spanwise periodic (2D section)',
    make: (alpha = 4) => wingSection('0012', 300, 400, alpha),
    alphas: [0, 4, 8],
    speed: 50, lref: 0.3, aref: 0.3, across: 120, acrossAxis: 'length', spanCells: 8,
    // 2D section: margins in chords (open boundaries 4 chords above and below)
    tunnel: { periodicSpan: true, margins: { up: 3, down: 6, side: 4, ofLength: true } },
    ref: {
      clSlope: 0.105, clSlopeRange: [0.095, 0.116],
      source: 'Abbott & von Doenhoff, Theory of Wing Sections (1959): 0.105/deg at Re 1e6-3e6; thin-airfoil theory 2π/rad = 0.110/deg',
    },
    perSpan: true,
  },
  {
    id: 'plate-laminar',
    name: 'Flat plate along the flow, laminar skin friction',
    note: '100 mm plate, 2 mm thick, at 2.2 m/s (Re_L 1.5e4), spanwise periodic',
    make: () => box(100, 2, 400),
    speed: 2.2, lref: 0.1, aref: 2 * 0.1, across: 200, acrossAxis: 'length', spanCells: 4,
    tunnel: { periodicSpan: true, margins: { up: 0.5, down: 0.5, side: 0.5, ofLength: true } },
    friction: true,
    ref: { cfLaminar: true, source: 'Blasius: Cf = 1.328 / sqrt(Re_L), both sides' },
    perSpan: true,
  },
  {
    id: 'plate-turbulent',
    name: 'Flat plate along the flow, turbulent skin friction',
    note: '3 m plate, 30 mm thick, at 50 m/s (Re_L 1.0e7), spanwise periodic',
    make: () => box(3000, 30, 8000),
    speed: 50, lref: 3, aref: 2 * 3, across: 200, acrossAxis: 'length', spanCells: 4,
    tunnel: { periodicSpan: true, margins: { up: 0.5, down: 0.5, side: 0.5, ofLength: true } },
    friction: true,
    ref: { cfTurbulent: true, source: 'Prandtl-Schlichting: Cf = 0.455 / (log10 Re_L)^2.58, both sides, turbulent from the leading edge' },
    perSpan: true,
  },
  {
    id: 'wing',
    name: 'Wing sample (NACA 2412, aspect ratio 3, 6°)',
    note: 'The app sample: 100 mm chord, 300 mm span at 30 m/s',
    make: () => {
      const g = new THREE.ExtrudeGeometry(new THREE.Shape(nacaOutline('2412', 100, 60)), { depth: 300, bevelEnabled: false, curveSegments: 1, steps: 1 });
      g.translate(-25, 0, -150);
      g.rotateZ(THREE.MathUtils.degToRad(-6)); // nose up
      return positionsOf(g);
    },
    speed: 30, lref: 0.1, aref: 0.1 * 0.3, across: 40, acrossAxis: 'length',
    ref: {
      cl: 0.53, clRange: [0.45, 0.6],
      source: 'Finite-wing lift: a = a0 / (1 + a0 / (pi AR)) with a0 = 0.105/deg, AR 3, zero-lift angle -2.1° (Abbott & von Doenhoff)',
    },
  },

];

/** Skin-friction reference for a case at Reynolds number re (per side). */
export function frictionReference(c, re) {
  if (c.ref.cfLaminar) return 1.328 / Math.sqrt(re);
  if (c.ref.cfTurbulent) return 0.455 / Math.log10(re) ** 2.58;
  return NaN;
}

export const caseById = (id) => CASES.find((c) => c.id === id);
