import * as THREE from 'three';

// SolidWorks-style rainbow: blue (low) -> cyan -> green -> yellow -> red (high).
const RAINBOW = [
  [0.0, [0, 0, 255]],
  [0.25, [0, 255, 255]],
  [0.5, [0, 255, 0]],
  [0.75, [255, 255, 0]],
  [1.0, [255, 0, 0]],
];

// Colour-blind-safe sequential alternative (ColorBrewer OrRd): pale = low, deep red = high.
const HEAT = ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000', '#7f0000'].map((hex, i, a) => [
  i / (a.length - 1),
  [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16)),
]);

export const NO_DATA_RGB = [150, 150, 150];

export function rainbow(t, out = [0, 0, 0]) {
  return ramp(RAINBOW, t, out);
}

export function colormap(name, t, out = [0, 0, 0]) {
  return ramp(name === 'heat' ? HEAT : RAINBOW, t, out);
}

function ramp(stops, t, out) {
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1];
      const [t1, c1] = stops[i];
      const s = (t - t0) / (t1 - t0);
      out[0] = c0[0] + s * (c1[0] - c0[0]);
      out[1] = c0[1] + s * (c1[1] - c0[1]);
      out[2] = c0[2] + s * (c1[2] - c0[2]);
      return out;
    }
  }
  const last = stops[stops.length - 1][1];
  out[0] = last[0]; out[1] = last[1]; out[2] = last[2];
  return out;
}

/**
 * 2-row texture: row 0 is the color ramp (sampled with uv.x = normalized value, uv.y = 0.25),
 * row 1 is the "no data" grey (uv.y = 0.75).
 * @param {number} bands 0 for a smooth ramp, otherwise the number of discrete contour bands
 * @param {boolean} reverse  red at the low end (used for factor of safety)
 */
export function makeColormapTexture(bands = 0, reverse = false, name = 'rainbow') {
  const w = bands > 0 ? bands : 256;
  const data = new Uint8Array(w * 2 * 4);
  const c = [0, 0, 0];
  for (let i = 0; i < w; i++) {
    const t = bands > 0 ? (i + 0.5) / w : i / (w - 1);
    colormap(name, reverse ? 1 - t : t, c);
    data.set([c[0], c[1], c[2], 255], 4 * i);
    data.set([...NO_DATA_RGB, 255], 4 * (w + i));
  }
  const tex = new THREE.DataTexture(data, w, 2, THREE.RGBAFormat);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.magFilter = bands > 0 ? THREE.NearestFilter : THREE.LinearFilter;
  tex.minFilter = bands > 0 ? THREE.NearestFilter : THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

export function cssGradient(reverse = false, bands = 0, name = 'rainbow') {
  const stops = [];
  const n = bands > 0 ? bands : 16;
  const c = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const t0 = i / n, t1 = (i + 1) / n;
    const t = bands > 0 ? (i + 0.5) / n : t0;
    colormap(name, reverse ? 1 - t : t, c);
    const col = `rgb(${c.map(Math.round).join(',')})`;
    if (bands > 0) stops.push(`${col} ${(t0 * 100).toFixed(2)}%`, `${col} ${(t1 * 100).toFixed(2)}%`);
    else stops.push(`${col} ${(t0 * 100).toFixed(2)}%`);
  }
  if (!bands) {
    colormap(name, reverse ? 0 : 1, c);
    stops.push(`rgb(${c.map(Math.round).join(',')}) 100%`);
  }
  // bottom = low values
  return `linear-gradient(to top, ${stops.join(', ')})`;
}
