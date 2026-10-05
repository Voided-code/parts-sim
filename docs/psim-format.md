# The .psim file format

A `.psim` file holds one part, how it is set up, and the results of the studies run on it, in as few
bytes as practical. Either Parts Sim app (web, Electron or native) and the website open it and show the
results at once, without solving again. If the geometry and setup are inside, **Re-run** recomputes them.

This page is the specification, format version 1. The JavaScript reader and writer are in
[src/core/psim.js](../src/core/psim.js); `node scripts/psim.mjs` inspects, verifies and converts files.

- Suggested MIME type: `application/x-parts-sim`. File extension: `.psim`.
- Nothing in a file is ever run or evaluated. A file is data only.
- No user name, path or machine detail is written. The file does contain the customer's geometry when the geometry
  is included, so the save dialog says so.

## Container

All numbers are little-endian. The same content always produces the same bytes from the same implementation
(see *Determinism*).

```
offset  size  field
0       9     magic: 89 50 53 49 4D 0D 0A 1A 0A   (\x89 "PSIM" CR LF ^Z LF)
9       1     0
10      2     format version that wrote the file (u16, 1)
12      2     oldest reader version that can read it (u16, 1)
14      2     flags (u16, 0)
16      4     section count N (u32, at most 64)
20      4     offset of the section table (u32, 32)
24      4     CRC-32 of bytes 0..23 followed by the section table
28      4     0
32      24*N  section table
...           section data, in table order, with no gaps
end-8   4     CRC-32 of every byte before the trailer
end-4   4     total file length (u32)
```

The magic follows PNG: the high bit catches 7-bit transfers, `CR LF` and `LF` catch text-mode conversions,
`^Z` stops a DOS `type`.

Each section table entry is 24 bytes:

```
0   4  id: four ASCII letters
4   2  codec (u16): 0 stored, 1 deflate-raw (RFC 1951); 2 and 3 are reserved for zstd and brotli
6   2  flags (u16): bit 0 = lossy section (its INFO bounds apply)
8   4  offset of the stored bytes from the start of the file (u32)
12  4  stored length (u32)
16  4  uncompressed length (u32)
20  4  CRC-32 of the uncompressed bytes
```

Rules a reader enforces, in this order, before it allocates anything large:

1. The file is at least 48 bytes, the magic matches, `minReader` is not newer than the reader (otherwise it says
   "this file needs a newer Parts Sim (format N)"), and the trailer's length equals the real length.
2. The trailer CRC-32 over the whole file and the header CRC-32 match. A mismatch is reported as a damaged file.
3. The first section starts right after the table, every next one starts where the previous one ends, and the last
   ends at the trailer. Overlaps, gaps and out-of-range values are rejected.
4. Section limits: uncompressed length at most 512 MiB per section and 1 GiB for the file, and at most 1100 times the
   stored length plus 64 KiB (the best deflate can do is about 1032 to 1). Decompression stops at the declared length
   and the result must be exactly that long, with a matching CRC-32.
5. Unknown section ids and unknown codecs of optional sections are skipped. Unknown codecs of a needed section
   are reported by name.

Sections are compressed one by one, so a reader can fetch INFO and the thumbnail without decompressing the rest.

### Determinism

Sections are always written in the order below, JSON objects with their keys in a fixed order, and arrays in a fixed
byte layout. The deflate bytes depend on the zlib build, so two implementations can write different bytes for the same
content; hash the *uncompressed* sections (the CRC-32 values in the table) to compare content across apps. The
writer takes `created` as an input, so tests can fix it.

## Sections

| id | content | codec |
|---|---|---|
| `INFO` | JSON: what the file is and what is in it | deflate |
| `THMB` | a JPEG preview, a few KB | stored |
| `GEOM` | the part (see below) | deflate |
| `CADS` | the original CAD file's bytes (off by default) | deflate |
| `SETP` | JSON: material, fixtures, loads, study settings, thermal and airflow setup | deflate |
| `RFEA` | structural and thermal results (arrays container) | deflate |
| `RAIR` | airflow results (arrays container) | deflate |
| `VIEW` | JSON: camera, colour scale, active tab and plot | deflate |

Only the sections a file needs are present. Sections appear in the order of the table above.

### INFO

A JSON object with these keys (the order of keys is not significant, and extra keys are allowed and ignored):

```json
{
  "format": 1,
  "app": { "name": "Parts Sim", "version": "1.1.0", "kind": "web" },
  "created": "2026-10-04T12:00:00Z",
  "name": "Ahmed body",
  "notes": "",
  "units": "mm",
  "contains": { "geometry": "quantised16", "setup": true, "results": ["airflow"], "cad": false },
  "part": { "vertices": 31204, "triangles": 62404, "bbox": { "min": [0, 0, 0], "max": [1044, 288, 389] } },
  "bounds": [
    { "field": "geometry.position", "absolute": 0.0079, "unit": "mm" },
    { "field": "airflow.velocity", "absolute": 0.0012, "unit": "m/s" }
  ]
}
```

- `notStored` (optional): the ids of results the app had but a file cannot hold. Version 1 knows `dynamic` (the linear dynamic study: its modal basis is large, so its settings are stored and Re-run recomputes it). The info panel says so.
- `kind` is `web`, `electron` or `native`. There is no user name, folder name or machine name.
- `bounds` lists the worst-case error of every lossy field: the largest difference between a stored and an original
  value. Fields that are not listed are exact.

### GEOM

The part as the apps hold it after import: welded vertices, indexed triangles, face ids. Results and patches refer to
vertex and triangle numbers of exactly this mesh, so it is stored as built, not re-created from the source file.

```
0    1   mode: 0 exact (float32 positions), 1 quantised
1    1   bits per axis when quantised (16), else 0
2    1   flags: bit 0 = faces come from the CAD B-rep (face ids are stored)
3    1   0
4    4   vertex count nV (u32)
8    4   triangle count nT (u32)
12   4   face count (u32)
16   4   smooth-face angle in degrees (f32, 0 for B-rep faces)
20   48  bounding box minimum then maximum (6 x f64)
68   ..  data planes
```

All integers below are first mapped to unsigned with zig-zag (`(v << 1) ^ (v >> 31)`), then split into byte
planes (all low bytes, then all next bytes, and so on). Byte planes compress far better than interleaved values.

1. **Positions**, three axes in turn (x for all vertices, then y, then z).
   - Exact: the float32 bit patterns, byte planes of 4.
   - Quantised: `q = round((v - min) / (max - min) * 65535)` as 16 bits (an axis with `max = min` stores 0), the
     difference to the previous vertex's `q` (the first is stored as is), zig-zag mapped as 16 bits, 2 byte planes.
     Error: at most `(max - min) / 131070` per axis, listed in INFO.
2. **Triangle corners**, the three corner slots in turn. For slot `k`, the difference of corner `k` to corner `k` of the
   previous triangle (the first is stored as is), zig-zag mapped, 4 byte planes.
3. **Face ids**, only when flag bit 0 is set: one value per triangle, difference to the previous, zig-zag, 4 byte planes.
   Without CAD faces the faces are made again from the smooth-face angle (region growing across edges bent less
   than the angle), and the face count must match.

The reader checks that every corner index is below `nV`, that no triangle uses one vertex twice, and that the
counts fit the section length, before it builds a part.

### SETP

A JSON object. Triangle sets (`patches`) are lists of runs `[start, count, start, count, ...]` over sorted
triangle numbers.

```json
{
  "units": "mm",
  "material": { "id": "steel-1020", "name": "...", "E": 205, "nu": 0.29, "yield": 350, "uts": 420, "density": 7870,
                "k": 51.9, "cp": 486, "alpha": 11.7, "fatigue": { "Se": 200, "Ne": 1e6, "endurance": true, "metal": true }, "cost": 0.8 },
  "structural": {
    "study": "static",
    "resolution": 56,
    "gravity": false,
    "fixtures": [ { "name": "Fixed 1", "patches": [ { "tris": [0, 12], "clip": null, "label": "Face 3" } ] } ],
    "loads": [ { "name": "Force 1", "type": "force", "magnitude": 800, "dir": [0, -1, 0], "patches": [ ... ] } ],
    "options": { "nonlinear": { }, "modal": { }, "buckling": { }, "fatigue": { }, "drop": { }, "dynamic": { }, "optimize": { } }
  },
  "thermal": { "mode": "steady", "items": [ { "name": "...", "type": "temp", "value": 100, "ambient": 20, "patches": [ ... ] } ],
               "ambient": { "enabled": true, "h": 10, "t": 20 }, "duration": 600, "steps": 60, "initial": 20, "resolution": 56 },
  "airflow": { "yaw": 90, "pitch": 0, "speed": 30, "density": 1.225, "cells": 1000000, "ground": null,
               "boundaryLayer": "auto", "engine": "auto" }
}
```

- `type` of a load is `force` (newtons), `pressure` (megapascals) or `wind`. A wind load has no patches; its
  per-triangle forces are in the `RFEA` arrays as `load.<index>.forces`.
- The material is stored whole (library or custom), so the file does not depend on the library.
- Keys that do not apply to the file's contents are left out. A reader uses defaults for missing keys.

### Arrays container (`RFEA`, `RAIR`)

Results are numbers in arrays. A result section starts with a JSON header and then the array bytes:

```
0       4   header length H (u32)
4       H   UTF-8 JSON
4+H     ..  array data, in the order of the header's "arrays" list
```

The header is `{ "meta": { ... }, "arrays": [ ... ] }`. `meta` holds every scalar and small list of the results (JSON
cannot hold infinity or not-a-number, so these are written as `{"$num":"inf"}`, `{"$num":"-inf"}` and
`{"$num":"nan"}`). Each array entry is:

```json
{ "name": "static.vm", "enc": "q16", "n": 31204, "dims": [63, 40, 30], "min": 0, "max": 1.2e8, "err": 458.2, "bytes": 51234 }
```

- `name`: `<study>.<field>`, for example `static.vm`, `thermal.T`, `airflow.cp`, `airflow.avg.ux`.
- `enc`:
  - `f32`: float32 bit patterns in 4 byte planes. Exact.
  - `i32`: signed 32-bit integers, zig-zag, 4 byte planes. Exact.
  - `u8`: bytes. Exact.
  - `q16` and `q8`: a lossy quantisation. `min` and `max` are the smallest and largest finite value. A finite value
    `v` becomes `c = 1 + round((v - min) / (max - min) * (L - 2))` with `L = 65536` (`q16`) or `256` (`q8`), and
    `c = 0` means "no value" (NaN or a solid cell). When `max = min`, every finite value is `c = 1`. The error is at most
    `(max - min) / (2 * (L - 2))`. The codes are then predicted and written as zig-zag residuals in byte planes.
- `err`: written for `q16` and `q8`: the largest error of the array, `(max - min) / (2 * (L - 2))`. Readers may ignore it.
- `stride`: optional, 2, 3 or 4, for `q16` and `q8` arrays without `dims` whose values come in groups (a displacement or a mode shape has three values per vertex: x y z). The prediction is then the value `stride` places back (the same component of the previous vertex) instead of the previous value, and `n` must be a multiple of `stride`. Without it (stride 1), the prediction is the previous value. On a displacement field this makes the array about half the size.
- `base`, with `min` and `max` equal to the base array's: for a series of arrays that change little from one to the next (the steps of a break test, the frames of a drop test). `base` names an earlier array of the same `enc`, `n`, `min` and `max`. The codes are first differenced against the base's codes (`(code - base code) mod L`), then predicted as usual (`stride`, `dims`), and the reader adds the base's codes back. The writer gives the whole series one shared `min` and `max`. On a break test this makes the steps 40-50% smaller.
- `dims`: present for 3-D grids (`nx * ny * nz = n`, x fastest). The prediction is the 3-D Lorenzo predictor
  `p = a[x-1] + a[y-1] + a[z-1] - a[x-1,y-1] - a[x-1,z-1] - a[y-1,z-1] + a[x-1,y-1,z-1]` (missing neighbours count as
  0). Without `dims` the prediction is the previous value. The residual `code - p` is taken modulo `L`, as a signed
  value, then zig-zag mapped.
- `bytes`: the length of the array's data. A reader checks that the arrays fill the section exactly.

Reading a quantised array returns `v = min + (c - 1) / (L - 2) * (max - min)` and `NaN` for `c = 0`.

#### `RFEA` contents

`meta.results` lists the studies present. A per-vertex array has one value per vertex of `GEOM` (`nV`), a
displacement or shape has three (`3 nV`, x y z per vertex).

| study | meta | arrays |
|---|---|---|
| `static` | maxima, load factors, flags, material, units, resolution, total force | `vm`, `p1`, `p3`, `fos` (q16, per vertex), `u` (q16, 3 nV) |
| `break` | per step: load factor, maximum displacement, cracked and detached voxel counts | per step `vm` and `u`, `cracked` and `detached` voxel ids |
| `nonlinear` | per step: load factor, displacement, maxima | per step `vm`, `pe`, `u`; the unloaded step |
| `modal` | frequencies, effective mass | per mode `shape` |
| `buckling` | factors | per mode `shape`, and `vm` |
| `fatigue` | options | `vm`, `p1`, `p3` (the damage and life are recomputed) |
| `drop` | history, peak force, contact time | `vmMax`, `tPeak`, per frame `u` and `vm` |
| `topology` | history | `density` (q8, voxel grid), the extracted shape |
| `sizing` | the table of rows | none |
| `thermal` | range, history | `T`, `flux`, and per frame `T` when transient |

`dynamic` (the modal basis for frequency sweeps) is not stored in version 1: it is large and is solved again with Re-run.

#### `RAIR` contents

`meta` holds the force results (drag, lift, side force and friction in newtons, Cd, Cl, frontal area, the 95 %
confidence half-widths, whether it is averaged), the reference values used (air density, speed, dynamic pressure,
reference area), both Reynolds numbers with the length they use, the wall model, the steps and averaging samples, the
engine, and the coarse grid (`dims`, `factor`). Arrays: `airflow.cp` (surface pressure coefficient per vertex, q16),
and the coarse averaged flow on the grid, `airflow.avg.rho`, `airflow.avg.ux`, `airflow.avg.uy`, `airflow.avg.uz`
(q16 with `dims`; solid cells are `c = 0`). Slice, streamlines and particles are drawn from these, as in a live run.

### VIEW

A JSON object: `{ "tab": "airflow", "study": "static", "camera": { "position": [...], "target": [...] },
"plot": "vm", "scalePct": 100, "level": "applied", "step": 0, "mode": 0, "flowDisplay": { "cp": true, "slice": false } }`.
Missing keys mean the app's defaults.

### THMB

A JPEG at most 320 pixels wide, normally 5 to 15 KB.

### CADS

The bytes of the original CAD file (STEP, IGES, SolidWorks), as imported, with its file name in INFO
(`contains.cadName`, without any folder). It is off by default; the save dialog shows its size.

## Opening a file

1. Read the header, the table and the trailer; check them (the rules above).
2. Show INFO and the thumbnail.
3. Decompress the sections the app needs: GEOM first, then SETP, the results and VIEW.
4. Build the part from GEOM, restore the setup from SETP, and give each study its stored results. The results view
   says **Loaded from file, not computed here**, with the file's app version, the size of each section and the error
   bounds. The results stay on screen until the user changes the setup or presses Re-run.
5. **Re-run** needs GEOM and SETP. It solves again and offers the difference from the stored result.

A file made by a newer app than the reader is still opened when `minReader` allows it; unknown keys and sections are skipped.

## Safety

Files come from strangers. A reader:

- never evaluates anything from a file, and reads JSON only with a parser that builds plain data;
- checks every count, offset and length against the file size before it allocates;
- caps each section, the file and the compression ratio, and the number of vertices and triangles
  (50 million each) and of array elements (500 million);
- refuses a file whose arrays do not fill their section or whose indices point outside the mesh;
- reports a clear message for every failure, and never hangs: all loops are bounded by the checked sizes.

The test suite (`npm test`, `test/psim.test.js`) holds round trips for every section, lossy and exact modes, a fixture per
format version, truncated and bit-flipped files, oversized headers, decompression bombs and a fuzz run.

## Sizes

The measured sizes against binary STL and raw results are in [website-and-psim.md](website-and-psim.md).

### Study result schemas (RFEA)

These are the schemas of the structural studies that `src/ui/studies.js` writes (`exportResult`) and reads (`importResult`).
`nV` is the part's vertex count, in the vertex order of `GEOM`. Every `meta` has `nVert` (= `nV`); a reader refuses the
study when it differs from the loaded part, and when an array does not have the stated length. The material is stored
whole in `meta.material` (an object as in `SETP`); `units` is the model's unit (`mm`, `cm`, `m`, `in`), `diag` the
bounding-box diagonal in model units. Values in `meta` may be infinite (written as `{"$num":"inf"}`). A key that is
`null` means "absent". Only what a redraw needs is stored: damage and life (fatigue), the work arrays of the solver, the
unused principal stress of nonlinear steps and the modal basis of dynamic are not. Linear dynamic (`dynamic`) is not
stored in version 1.

Lossy fields use `q16` (relative to the range of that array, at most `1/131068` = 7.6e-6 of `max - min`) except the
topology density, which uses `q8` (at most `1/508` = 0.002 of the density range 0..1). Everything else is exact.

| study | arrays (`enc`, count) | `meta` keys |
|---|---|---|
| `nonlinear` | `nonlinear.step.<i>.u` (q16, 3 nV), `nonlinear.step.<i>.vm` (q16, nV, pascal), `nonlinear.step.<i>.pe` (q16, nV, plastic strain, 1 = 100 %) for each load step `i` from 0; the same three as `nonlinear.unloaded.u/.vm/.pe` after spring-back. `u` is in model units. | `units`, `material`, `totalF` (newton, size of all loads), `reason` (`reached`, `collapse`, `rupture`, `crack`, `large deformation`, `time limit`, `step limit`), `engine`, `gpuNote`, `plastic`, `opts` (`large`, `plastic`, `mode`, `steps`), `steps` (a list of `{lam, D, maxVM, maxPE, maxDisp, iterations}`: load factor, load-weighted displacement in model units, maxima in pascal, strain and model units), `unloaded` (the same object, or `null`) |
| `modal` | `modal.shape.<i>` (q16, 3 nV) for mode `i` from 0. Each shape is scaled so its largest displacement is 1. | `units`, `material`, `diag`, `free` (no fixtures), `converged`, `iterations`, `totalMass` (kg), `engine`, `gpuNote`, `modes` (a list of `{freq, eff}`: hertz and the three effective-mass fractions x y z) |
| `buckling` | `buckling.shape.<i>` (q16, 3 nV), scaled to a largest displacement of 1. | `units`, `material`, `diag`, `totalF`, `maxVM` (pascal, largest stress under the applied loads), `maxFactor` (largest factor searched, may be infinite), `converged`, `iterations`, `engine`, `gpuNote`, `factors` (list of load factors, infinite when none) |
| `fatigue` | `fatigue.vm`, `fatigue.p1`, `fatigue.p3` (q16, nV, pascal: von Mises and the largest and smallest principal stress under the applied loads) | `units`, `material`, `totalF`, `engine`, `gpuNote`. The options (`SETP`/view: `loading`, `R`, `cycles`, `finish`, `scale`) are applied again to the stresses, so life, damage and safety factor are recomputed on loading. |
| `drop` | `drop.vmMax` (q16, nV, pascal, peak over the impact), `drop.times` (f32, `nTimes`, seconds) and `drop.forces` (f32, `nTimes`, newton) for the force chart, and per stored frame `i` from 0: `drop.frame.<i>.u` (q16, 3 nV, model units) and `drop.frame.<i>.vm` (q16, nV, pascal). All frames of the run are kept. | `units`, `material`, `height` (m), `speed` (m/s), `mass` (kg), `peak` (pascal, maximum of `vmMax`), `peakAt` (vertex of the maximum), `peakForce`, `contactTime`, `duration`, `steps`, `dt` (s), `rebounded`, `autoScale` (shape exaggeration that makes the largest movement 5 % of the part), `nTimes`, `peakFrame` (index of the first frame at or after the peak-stress time of `peakAt`), `frameTimes` (seconds, one per frame), `frameForces` (newton), `engine`, `gpuNote` |
| `optimize`, topology | `optimize.density` (q8, `nx*ny*nz`, `dims` = the voxel grid, x fastest) | `goal` = `topology`, `units`, `material`, `toMeters`, `resolution` (voxels on the longest side: the grid is rebuilt by the voxelizer at this resolution and must give `dims`), `dims`, `history` (a list of `{it, compliance, volume}`), `keptFraction`, `engine`. The shown shape is an iso-surface of the density at the view's `level` and is rebuilt on loading. |
| `optimize`, sizing | none | `goal` = `sizing`, `units`, `material`, `engine`, `gpuNote`, `base` (`{fos, peak, exact, disp}` of the current design), `rows` (a list of `{id, name, feasible, scale, fos, disp, mass, cost, exact}`; infeasible rows only have `id`, `name`, `feasible`) |

Study options (`opts`) and view settings (`plot`, `step`, `mode`, `frame`, `level`...) are plain JSON in `SETP` and
`VIEW`; the readers use defaults for missing keys and ignore values of the wrong type.

### Airflow result schema (RAIR)

The section is an arrays container. `meta` keys (all numbers are plain JSON numbers, or `{"$num":...}` for not-a-number
and infinity; `NaN` is written for a confidence half-width that does not exist yet):

| key | meaning |
|---|---|
| `schema` | 1 |
| `results` | object: `drag`, `lift`, `side`, `frictionDrag` (N), `dragCI`, `liftCI`, `sideCI` (N, 95 % half-widths, `NaN` until averaged), `cd`, `cl`, `cdCI`, `clCI` (dimensionless), `frontalArea` (m², already in SI), `force` (`[x, y, z]` newtons in world axes), `averaged` (bool) |
| `reynolds` | on the part's larger size across the wind; `reynoldsLength`: on its length along the wind (decides the wall model); `simReynolds`: the value the grid simulates |
| `nuAir`, `nuLat`, `uLat` | air viscosity (m²/s), lattice viscosity, lattice free-stream speed (0.08) |
| `wallModel` | bool, turbulent wall model on |
| `groundGap` | gap under the part in cells, `-1` without a road |
| `q` | dynamic pressure `0.5 rho U²` (Pa) |
| `frontal` | frontal area in cells² |
| `steps`, `samples` | lattice steps run, averaging samples taken |
| `converged`, `developing` | bools |
| `mlups`, `engine` | speed of the run, `"WebGPU"` or `"CPU"` |
| `dims` | tunnel grid `[nx, ny, nz]` (cells); `h`: cell size in model units; `toMeters`: metres per model unit |
| `factor`, `fieldDims`, `fieldSamples` | coarse view grid: cells per coarse cell side, its size `[ceil(nx/factor), ceil(ny/factor), ceil(nz/factor)]`, number of samples in the stored average |
| `nVert` | vertices of the part (the length of `airflow.cp`) |
| `settings` | `dir` (`[x, y, z]` unit vector the air travels), `yaw`, `pitch` (degrees, derived from `dir`), `speed` (m/s), `airDensity` (kg/m³), `ground` (clearance in model units, or `null`), `boundaryLayer` (`auto`, `turbulent`, `laminar`), `cells` (requested), `engine` (requested) |

Arrays, all `q16`:

| name | elements | dims | content |
|---|---|---|---|
| `airflow.cp` | `nVert` | none | time-averaged pressure coefficient per part vertex, `Cp = surfaceRho * 1/(3 * 0.5 * uLat²)`; `NaN` (code 0) where the vertex has no wall cell |
| `airflow.avg.rho` | `fieldDims[0] * fieldDims[1] * fieldDims[2]` | `fieldDims` | time-averaged `rho - 1` per coarse cell (lattice units), x fastest |
| `airflow.avg.ux`, `.uy`, `.uz` | same | `fieldDims` | time-averaged velocity components in the tunnel frame (lattice units, `uLat` = free stream), x along the wind |

A coarse cell that holds no fluid (solid, or the ground layer) is `NaN` in all four arrays, so its code is `0`. On reading,
such a cell is the solid marker `rho - 1 = -2` with velocity 0. Only the average is stored; particles move in it too. The
tunnel frame is planned again on load: `planTunnel(part, dir, any, {margins: "app", ground, h})` with the stored `h` gives the
same `dims` and origin (the reader checks `dims`), and the fine solid cells for the streamlines and particles come from
voxelising the part on that grid. Lattice position `p` maps to the model as `x = e1 * (origin0 + h p0) + e2 * (origin1 + h p1) + e3 * (origin2 + h p2)`
with `e1 = dir`, `e2` = the world up vector (`(0,1,0)`, or `(1,0,0)` when `|dir.y| >= 0.9`) made orthogonal to `e1` and normalised, `e3 = e1 x e2`.
