import { cssGradient } from '../viewer/colormap.js';
import { h } from './dom.js';

/**
 * Render one or more color legends into `el`.
 * spec: { title, sub, min, max, format(v) -> string, bands, reverse, colormap, markers: [{value, label}] }
 */
export function renderLegend(el, specs) {
  el.replaceChildren();
  const list = (Array.isArray(specs) ? specs : [specs]).filter(Boolean);
  el.hidden = list.length === 0;
  list.forEach((s, i) => {
    const ticks = h('div.legend-ticks');
    const H = list.length > 1 ? 150 : 220;
    const n = 6;
    // marker lines (yield, tensile strength) win over the regular ticks they would overlap
    const marks = [];
    for (const m of s.markers || []) {
      if (!(m.value >= s.min && m.value <= s.max)) continue;
      const y = H - ((m.value - s.min) / (s.max - s.min || 1)) * H;
      if (marks.some((o) => Math.abs(o.y - y) < 14)) continue;
      marks.push({ y, label: m.label });
    }
    for (let k = 0; k <= n; k++) {
      const v = s.min + ((s.max - s.min) * k) / n;
      const y = H - (k / n) * H;
      if (marks.some((m) => Math.abs(m.y - y) < 14)) continue;
      ticks.append(h('span', { style: { top: `${y}px` } }, s.format(v)));
    }
    for (const m of marks) ticks.append(h('span.yield', { style: { top: `${m.y}px`, left: '0' } }, m.label));
    const bar = h('div.legend-bar', { style: { background: cssGradient(s.reverse, s.bands, s.colormap), height: `${H}px` } });
    ticks.style.height = `${H}px`;
    el.append(
      h('div', { style: { marginTop: i ? '12px' : '0' } },
        h('div.legend-title', {}, s.title),
        s.sub ? h('div.legend-sub', {}, s.sub) : null,
        h('div.legend-body', {}, bar, ticks),
      ),
    );
  });
}
