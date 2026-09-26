export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/** Tiny element builder: h('div.card', {onclick}, child, 'text') */
export function h(tag, attrs = {}, ...children) {
  const [name, ...classes] = tag.split('.');
  const el = document.createElement(name || 'div');
  if (classes.length) el.className = classes.join(' ');
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k in el && k !== 'list') el[k] = v;
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) el.append(c instanceof Node ? c : String(c));
  return el;
}

const nf = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 3 });
const nf4 = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 4 });

export function num(v, digits = 3) {
  if (!Number.isFinite(v)) return '–';
  if (v !== 0 && (Math.abs(v) >= 1e6 || Math.abs(v) < 1e-3)) return v.toExponential(digits - 1);
  return (digits >= 4 ? nf4 : nf).format(v);
}

export function stress(pa) {
  if (!Number.isFinite(pa)) return '–';
  const a = Math.abs(pa);
  if (a >= 1e9) return `${num(pa / 1e9)} GPa`;
  if (a >= 1e5) return `${num(pa / 1e6)} MPa`;
  if (a >= 100) return `${num(pa / 1e3)} kPa`;
  return `${num(pa)} Pa`;
}

/** Stress formatter with one fixed unit chosen from the largest magnitude (for legends and probes). */
export function stressFormatter(maxAbs) {
  const [div, unit] = maxAbs >= 1e9 ? [1e9, 'GPa'] : maxAbs >= 1e5 ? [1e6, 'MPa'] : maxAbs >= 100 ? [1e3, 'kPa'] : [1, 'Pa'];
  return (pa) => (Number.isFinite(pa) ? `${num(pa / div)} ${unit}` : '–');
}

export function force(n) {
  if (!Number.isFinite(n)) return '–';
  const a = Math.abs(n);
  if (a >= 1e6) return `${num(n / 1e6)} MN`;
  if (a >= 1e4) return `${num(n / 1e3)} kN`;
  return `${num(n)} N`;
}

export function length(v, unit) {
  return `${num(v)} ${unit}`;
}

export function nextFrame() {
  return new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
}
