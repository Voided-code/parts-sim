# Third-party notices

Parts Sim is MIT-licensed. It depends on the following components, which keep their own licences.

| Component | Use | Licence |
|---|---|---|
| [three.js](https://github.com/mrdoob/three.js) | 3D rendering; STL, OBJ, PLY, 3MF and glTF loaders | MIT |
| [three-mesh-bvh](https://github.com/gkjohnson/three-mesh-bvh) | Fast picking and wall-distance ray casts | MIT |
| [occt-import-js](https://github.com/kovacsv/occt-import-js) | STEP, IGES and BREP import (WebAssembly) | LGPL-2.1 |
| [Open CASCADE Technology](https://dev.opencascade.org/) (inside occt-import-js) | CAD kernel used for tessellation | LGPL-2.1 with the Open CASCADE exception |
| [fflate](https://github.com/101arrowz/fflate) (bundled with three.js) | 3MF unzip, SolidWorks block inflation | MIT |
| [Electron](https://www.electronjs.org/) (desktop app) | Desktop shell, including Chromium and Node.js | MIT (Chromium: BSD-style and others, listed in `LICENSES.chromium.html` inside the app) |

### Native app (`native/`)

| Component | Use | Licence |
|---|---|---|
| [Qt 6](https://www.qt.io/) (Core, Gui, Widgets, Shader Tools) | Windows, widgets, 3D viewport (Qt RHI) | LGPL-3.0 |
| [Open CASCADE Technology](https://dev.opencascade.org/) | STEP / IGES import, CAD kernel for the samples | LGPL-2.1 with the Open CASCADE exception |
| [wgpu-native](https://github.com/gfx-rs/wgpu-native) | GPU compute (Metal, Direct3D 12, Vulkan) | MIT or Apache-2.0 |
| [zlib](https://zlib.net/) | SolidWorks block inflation, 3MF unzip | zlib |

## SolidWorks format

The SolidWorks reader (`src/core/solidworks.js`) is an independent JavaScript implementation. Its format facts come from:

- [sldprt-export](https://github.com/XRTC5/sldprt-export) (MIT, © 2026 XRTC5): the block container, strip tables and assembly directory. It was used as the reference implementation during testing.
- The [cadmpeg](https://github.com/cadmpeg/cadmpeg) project's `docs/formats/sldprt.md` (CC BY 4.0): the container layout.

The native app's reader (`native/src/core/solidworks.cpp`) is a C++ port of the same code. No SolidWorks code, headers or libraries are used. SolidWorks is a trademark of Dassault Systèmes.

## LGPL components

occt-import-js and Open CASCADE are shipped unmodified as a separate WebAssembly module (`occt-import-js.wasm`) loaded at runtime in a Web Worker. You may replace that module with your own build of occt-import-js: rebuild the library, swap in the `occt-import-js` npm package, and run `npm run build`.

The full licence texts ship with the package in `node_modules/occt-import-js/dist/license.occt-import-js.txt` and `license.occt.txt`. Their sources are available from the links above.

In the native app, Qt and Open CASCADE are unmodified shared libraries that the app links dynamically. The packaged app keeps them as separate files: `Contents/Frameworks` on macOS, next to the executable on Windows and Linux. You may replace them with your own builds of the same versions. The full licence texts come with each library's distribution, and their sources are available from the links above.

## Website

The website's web app (`site/vendor/coi-serviceworker.js`) includes [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) v0.1.7, MIT, © 2021 Guido Zuidhof and contributors. It lets a static host without custom headers serve the app with the cross-origin isolation its helper threads need. The licence text is in `site/vendor/coi-serviceworker.LICENSE`.
