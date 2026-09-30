// Writes a STEP AP214 B-rep cube (6 planar faces, millimetres) for the native import tests:
//   node scripts/make-step-cube.mjs [size_mm] > native/tests/data/cube.stp
const s = Number(process.argv[2] || 20);
const V = [[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0], [0, 0, s], [s, 0, s], [s, s, s], [0, s, s]];
// corners of each face, counter-clockwise seen from outside, and the outward normal
const FACES = [
  [[0, 3, 2, 1], [0, 0, -1]], [[4, 5, 6, 7], [0, 0, 1]], [[0, 1, 5, 4], [0, -1, 0]],
  [[2, 3, 7, 6], [0, 1, 0]], [[1, 2, 6, 5], [1, 0, 0]], [[3, 0, 4, 7], [-1, 0, 0]],
];
const out = [];
let id = 0;
const add = (text) => { out.push(`#${++id}=${text};`); return id; };
const f = (x) => (Number.isInteger(x) ? `${x}.` : String(x));
const pt = (p) => add(`CARTESIAN_POINT('',(${p.map(f).join(',')}))`);
const dir = (d) => add(`DIRECTION('',(${d.map(f).join(',')}))`);

// check the winding: Newell normal of each loop must point outward
for (const [loop, n] of FACES) {
  const nn = [0, 0, 0];
  for (let i = 0; i < 4; i++) {
    const a = V[loop[i]], b = V[loop[(i + 1) % 4]];
    nn[0] += (a[1] - b[1]) * (a[2] + b[2]); nn[1] += (a[2] - b[2]) * (a[0] + b[0]); nn[2] += (a[0] - b[0]) * (a[1] + b[1]);
  }
  if (nn[0] * n[0] + nn[1] * n[1] + nn[2] * n[2] <= 0) throw new Error(`face ${loop} is wound inward`);
}

const appCtx = add("APPLICATION_CONTEXT('automotive design')");
add(`APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#${appCtx})`);
const prodCtx = add(`PRODUCT_CONTEXT('',#${appCtx},'mechanical')`);
const product = add(`PRODUCT('cube','cube','',(#${prodCtx}))`);
const formation = add(`PRODUCT_DEFINITION_FORMATION('','',#${product})`);
const defCtx = add(`PRODUCT_DEFINITION_CONTEXT('part definition',#${appCtx},'design')`);
const definition = add(`PRODUCT_DEFINITION('design','',#${formation},#${defCtx})`);
const shape = add(`PRODUCT_DEFINITION_SHAPE('','',#${definition})`);
const mm = add('(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))');
const rad = add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))");
const sr = add('(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())');
const tol = add(`UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#${mm},'distance_accuracy_value','confusion accuracy')`);
const geoCtx = add(`(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#${tol}))GLOBAL_UNIT_ASSIGNED_CONTEXT((#${mm},#${rad},#${sr}))REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY'))`);
const origin = add(`AXIS2_PLACEMENT_3D('',#${pt([0, 0, 0])},#${dir([0, 0, 1])},#${dir([1, 0, 0])})`);

const verts = V.map((p) => add(`VERTEX_POINT('',#${pt(p)})`));
const edges = new Map(); // "a-b" (a < b) -> edge id, running from a to b
const edgeOf = (a, b) => {
  const k = a < b ? `${a}-${b}` : `${b}-${a}`;
  if (!edges.has(k)) {
    const [p, q] = a < b ? [a, b] : [b, a];
    const d = V[q].map((x, i) => (x - V[p][i]) / s);
    const vec = add(`VECTOR('',#${dir(d)},${f(s)})`);
    const line = add(`LINE('',#${pt(V[p])},#${vec})`);
    edges.set(k, { id: add(`EDGE_CURVE('',#${verts[p]},#${verts[q]},#${line},.T.)`), start: p });
  }
  return edges.get(k);
};
const faces = FACES.map(([loop, n]) => {
  const oriented = loop.map((a, i) => {
    const b = loop[(i + 1) % 4], e = edgeOf(a, b);
    return add(`ORIENTED_EDGE('',*,*,#${e.id},${e.start === a ? '.T.' : '.F.'})`);
  });
  const edgeLoop = add(`EDGE_LOOP('',(${oriented.map((o) => `#${o}`).join(',')}))`);
  const bound = add(`FACE_OUTER_BOUND('',#${edgeLoop},.T.)`);
  const ref = n[0] ? [0, 1, 0] : [1, 0, 0];
  const axis = add(`AXIS2_PLACEMENT_3D('',#${pt(V[loop[0]])},#${dir(n)},#${dir(ref)})`);
  const plane = add(`PLANE('',#${axis})`);
  return add(`ADVANCED_FACE('',(#${bound}),#${plane},.T.)`);
});
const shell = add(`CLOSED_SHELL('',(${faces.map((x) => `#${x}`).join(',')}))`);
const solid = add(`MANIFOLD_SOLID_BREP('cube',#${shell})`);
const rep = add(`ADVANCED_BREP_SHAPE_REPRESENTATION('',(#${origin},#${solid}),#${geoCtx})`);
add(`SHAPE_DEFINITION_REPRESENTATION(#${shape},#${rep})`);

process.stdout.write(`ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Parts Sim test cube, ${s} mm'),'2;1');
FILE_NAME('cube.stp','2026-09-30T00:00:00',(''),(''),'Parts Sim make-step-cube.mjs','Parts Sim','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));
ENDSEC;
DATA;
${out.join('\n')}
ENDSEC;
END-ISO-10303-21;
`);
