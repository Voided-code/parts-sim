# How accurate is the airflow study?

Parts Sim's airflow study is a lattice-Boltzmann large-eddy simulation. This page shows what it gets
right, what it gets wrong, and by how much, against published wind-tunnel data. Every number here comes
from the validation cases in [src/cfd/validation.js](../src/cfd/validation.js), run with the app's own
engine and settings, and you can rerun them yourself (see *Reproducing* at the end).

## The short version

**Status (30 September 2026, before 1.0): not yet good enough.** Bluff bodies in free air are close:
the square plate, the cube and the sphere land within 10% of the published drag. Bodies near a
moving road, streamlined wings and skin friction are not: the Ahmed car body's drag comes out 45–70%
high, the wing sample's lift is far too low at the app's default settings, 2D wing sections give 50%
too much lift, and flat-plate skin friction is 50–90% too low. The table below has every number, and
*What limits the accuracy* says why. This page is updated with each change to the engine.

## What the engine does

- **Lattice Boltzmann, D3Q19**, with recursive regularised collision and a Smagorinsky subgrid model
  (large-eddy simulation). The air has its real viscosity (ISA, sea level, 15 °C): the grid does not
  slow the flow down to a lower Reynolds number unless the cells are so large that the lattice
  viscosity hits its floor, and the app says so when that happens.
- **Walls** sit where the part's surface really is, between cells (interpolated bounce-back), and
  thin walls with no cell inside them still block the flow.
- **Turbulent boundary layers** get a wall model (Reichardt's law of the wall, after Malaspinas and
  Sagaut, 2014) from a Reynolds number of 5 × 10⁵ along the part, or when you pick it. Below that the
  boundary layer is resolved by the grid.
- **A moving road** under the part (the Ahmed sample, any car) moves with the wind, as in a wind tunnel
  with a rolling belt.
- **Forces** come from the momentum the air exchanges with the part's surface (pressure and skin
  friction together), averaged once the flow has developed, with a 95% confidence interval from batch
  means. The study stops by itself when the interval is within about 1%.

The web app (WebGPU or the CPU) and the native app (Vulkan, Direct3D 12, Metal or all CPU cores) run
the same scheme; their GPU and CPU engines agree to about 10⁻⁵ in the tests.

## Results

Native app on an Apple M5 (Metal), the app's default tunnel and settings, forces averaged once the
flow has developed (± is the 95% confidence interval). Coefficients use each case's reference area.

| Case | Published | Parts Sim | Off by | Notes |
| --- | --- | --- | --- | --- |
| Square plate across the flow, Re 1.4×10⁵ | Cd 1.17 (1.10–1.20), Hoerner | Cd 1.124 ± 0.022 | −4% | in band |
| Cube, face-on, Re 1.4×10⁵ | Cd 1.05 (1.00–1.10), Hoerner | Cd 1.152 ± 0.019 | +10% | tunnel blockage: 1.06–1.09 in a tunnel 1.5–2.2× wider |
| Sphere, Re 1.4×10⁵ | Cd 0.47 (0.40–0.50), Achenbach | Cd 0.473 ± 0.007 | +1% | in band |
| Ahmed body 25°, 50 mm over a moving road, Re 2.9×10⁶ | Cd 0.285 (0.25–0.33), Ahmed; Lienhart & Becker | Cd 0.412 ± 0.006, Cl −0.43 | +45% | 5 cells under the body; 0.36 with the underbody left to plain walls |
| Ahmed body 35° | Cd 0.257 (0.22–0.30), Ahmed | Cd 0.430 ± 0.008 | +67% | as above |
| Wing sample, NACA 2412, aspect ratio 3, 6°, Re 2×10⁵ | Cl 0.53 (0.45–0.60), finite-wing theory | Cl 0.027 ± 0.015 | −95% | laminar setting (the default below Re 5×10⁵); 0.22 with the wall model |
| NACA 0012 section, Re 10⁶, 60 cells per chord | lift slope 0.105/° (0.095–0.116), Abbott & von Doenhoff | 0.157/° | +50% | Cl −0.06 at 0° (should be 0) |
| Flat plate, laminar skin friction, Re_L 1.5×10⁴ | Cf 0.0108, Blasius | Cf 0.0054 | −50% | |
| Flat plate, turbulent skin friction, Re_L 10⁷ | Cf 0.0030, Prandtl–Schlichting | Cf 0.00034 | −89% | wall model on |

The web app runs the same engine (its WebGPU and CPU solvers agree with the native ones), so its
numbers are the same to within the statistics.

## What limits the accuracy

- **Tunnel size (blockage).** The app sizes the wind tunnel from the part's cross-section, with the
  sides 1.35 blockage sizes away. That puts a bluff body's frontal area at about 7% of the tunnel's
  cross-section, which pushes drag up by roughly 10% (the cube). A larger tunnel costs cells: at the
  same cell count, cells get coarser.
- **Boundary layers the grid cannot resolve.** At a few million cells a car's or a wing's boundary
  layer is thinner than one cell. Below Re 5×10⁵ the app leaves it to the grid, and plain no-slip walls
  on so coarse a grid make streamlined parts separate far too early (the wing sample). Above it the
  wall model takes over, which keeps the flow attached, but its skin friction comes out far too low
  (the turbulent flat plate), and it over-predicts the lift of 2D sections.
- **Gaps under the part.** A 50 mm gap under a car body is about five cells at the default grid.
  The flow through it sets the lift and part of the drag, and five cells cannot resolve it.
- **Where the pressure is read.** The wall model reads the pressure one to two cells off the wall,
  which misses the sharp suction and stagnation peaks around a thin leading edge.

Being worked on for 1.0: a wall treatment that holds the right skin friction (the flat plates are the
test), the boundary-layer setting chosen from what the grid resolves rather than a fixed Reynolds
number, a larger default tunnel or a blockage correction, and finer grids near the part.

## Reproducing

```sh
node scripts/export-cases.mjs test-artifacts/cases        # the cases' meshes and settings
cd native && cmake --build build-release --target bench
build-release/bench --json cases.json cases ../test-artifacts/cases   # every case, to converged forces
```

Options: a list of case ids (`cube,sphere`), `scale=1.5` for a finer grid, `margins=2` for a larger
tunnel, `wall=on|off`, `engine=cpu`. In the web app, open `bench.html?suite=quick` (or `full`, which adds
a viscosity and a grid-refinement variant of every case).
