# Parts Sim (native)

The native C++ edition of Parts Sim. It has the same studies, samples and results as the Electron app in the
repository root. The UI is built with Qt Widgets and the viewport draws through Qt RHI: Metal on macOS, Direct3D on
Windows, Vulkan or OpenGL on Linux. CAD import uses OpenCascade, and GPU compute uses wgpu-native with the same WGSL
shaders the web app runs.

Compared with the Electron app:

- **Faster solvers.** The CPU solvers are about 10× faster and use every core (the JavaScript ones ran on one).
- **Identical results.** The validation tests give the same numbers as the JavaScript suite.
- **Smaller install, no browser engine.** The app is not an Electron wrapper.
- **Native STEP / IGES reading.** OpenCascade runs as native code, not WebAssembly.

## Status

| Area | Native app |
|---|---|
| Import: SolidWorks (`.SLDPRT` / `.SLDASM`), STEP, IGES, STL, OBJ, PLY, glTF / GLB, 3MF | ✓ |
| Part tab: material library and custom materials, units, size, orientation, face detection | ✓ |
| Linear static with load levels and bend-to-break animation, break test | ✓ (CPU or GPU) |
| Nonlinear static: large deflection, J2 plasticity, collapse, spring-back | ✓ |
| Frequency, buckling (LOBPCG) | ✓ |
| Fatigue (S-N, Marin, Goodman) | ✓ |
| Drop test (explicit dynamics, rigid floor) | ✓ (CPU, multithreaded) |
| Linear dynamic: harmonic sweep, shock, sine burst, earthquake | ✓ |
| Optimization: topology (SIMP) with STL export, material and size | ✓ |
| Thermal: steady and transient conduction, convection, heat flux | ✓ |
| Airflow: lattice-Boltzmann flow, particles, streamlines, slices, surface pressure, wind load for the bend test | ✓ (GPU or CPU) |

Not ported yet: the GPU version of the drop test, where the multithreaded CPU solver is used instead.

## Building

You need CMake 3.24+, Ninja, a C++20 compiler and these libraries:

| Library | Tested with | Needed for |
|---|---|---|
| Qt (`qtbase`, `qtshadertools`) | 6.11 (6.7 is the minimum) | the desktop app (engine and tests build without it) |
| Open CASCADE Technology | 7.9 | STEP / IGES import and the CAD-built samples |
| wgpu-native | 29 | GPU solvers (optional: without it everything runs on the CPU) |
| zlib | the system zlib | SolidWorks and 3MF files |

### macOS (Homebrew)

```sh
brew install cmake ninja qtbase qtshadertools opencascade wgpu-native
cmake -S native -B native/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build native/build
open "native/build/Parts Sim.app"
```

To build a self-contained `.dmg` (Qt, OpenCascade and wgpu are copied into the app):

```sh
cd native/build && cpack
```

### Windows and Linux

Install the same libraries with vcpkg, your package manager, or the Qt online installer. Point CMake at them with
`-DCMAKE_PREFIX_PATH=...` and build as above. `cpack` makes a ZIP and an NSIS installer on Windows and a `.tar.gz`
on Linux.

### Options

| CMake option | Default | Effect |
|---|---|---|
| `PARTS_SIM_APP` | ON | build the Qt app (OFF: engine and tests only) |
| `PARTS_SIM_STEP` | ON | OpenCascade import; turned off automatically when it is not found |
| `PARTS_SIM_GPU` | ON | wgpu-native GPU solvers; turned off automatically when it is not found |

Environment variables at run time:

| Variable | Effect |
|---|---|
| `PARTS_SIM_THREADS` | number of solver threads (default: all cores) |
| `PARTS_SIM_NO_GPU` | set to run everything on the CPU |
| `PARTS_SIM_SAMPLE=beam` | open a sample at start-up |

## Tests

```sh
cd native/build
ctest --output-on-failure
```

| Suite | What it checks |
|---|---|
| `test_fea` | element, cantilever deflection and stress against beam theory, solver speed |
| `test_model` | voxelization, thin walls, STL / OBJ / STEP / SolidWorks import |
| `test_gpu` | GPU multigrid-CG against the CPU solver |
| `test_studies` | frequency, free-free modes, buckling, elastica, plastic collapse, heat conduction, fin, drop impact, modal superposition, fatigue, topology |
| `test_lbm` | lattice-Boltzmann stability and drag, GPU against CPU, wall links, pressure forces |

The app has a small automation hook used for screenshot checks:

```sh
PARTS_SIM_SCRIPT="sample:bracket;tab:structural;study:modal;run;idle;shot:/tmp/modal.png;quit" \
  "native/build/Parts Sim.app/Contents/MacOS/Parts Sim"
```

## Source layout

| Path | Contents |
|---|---|
| `src/util` | thread pool and parallel loops, zip / inflate, XML, JSON |
| `src/core` | meshes and parts, BVH, voxelizer, materials, importers, SolidWorks reader, OpenCascade tessellation |
| `src/fea` | hex8 element, multigrid solver, structural model, static / break, eigen (LOBPCG), nonlinear, explicit dynamics, fatigue, modal response, topology, thermal, study runners |
| `src/cfd` | lattice-Boltzmann (CPU and GPU), wall links, forces, airflow controller |
| `src/gpu` | wgpu-native context and the GPU multigrid solver |
| `src/app` | Qt app: main window, viewport, panels, studies, samples |
| `shaders` | viewport shaders (GLSL, compiled by Qt's `qsb`) and the compute shaders (WGSL, embedded at build time) |
| `tests` | validation suites and test files |
