// Minimal canvas line chart (break test, load-displacement, frequency response, time histories).
// One series (no legend - the title names it), 2px line, recessive grid, hover tooltip, and a
// ringed marker on the point currently shown in the viewport.

function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const f = raw / mag;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * mag;
}

export class LineChart {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {HTMLElement} tip
   * @param {object} o
   * @param {boolean} [o.integerX]  x is a step counter (ticks on whole numbers)
   * @param {boolean} [o.logY]      logarithmic y axis (frequency response)
   * @param {(p: {x: number, y: number}, i: number) => string} [o.tipText]
   */
  constructor(canvas, tip, { xLabel, formatX, formatY, xTicks = 4, onPick, integerX = true, logY = false, tipText = null, yMin = 0 } = {}) {
    this.canvas = canvas;
    this.tip = tip;
    this.opts = { xLabel, formatX, formatY, xTicks, onPick, integerX, logY, tipText, yMin };
    this.points = [];
    this.current = -1;
    this.hover = -1;
    canvas.addEventListener('pointermove', (e) => this.onMove(e));
    canvas.addEventListener('pointerleave', () => { this.hover = -1; this.tip.hidden = true; this.draw(); });
    canvas.addEventListener('click', () => this.hover >= 0 && this.opts.onPick?.(this.hover));
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  set(points, current = points.length - 1) {
    this.points = points;
    this.current = current;
    this.draw();
  }

  layout() {
    const css = getComputedStyle(this.canvas);
    const w = this.canvas.clientWidth, hgt = this.canvas.clientHeight || 150;
    const pad = { l: 50, r: 10, t: 8, b: 30 };
    const xs = this.points.map((p) => p.x), ys = this.points.map((p) => p.y).filter(Number.isFinite);
    const xMin = Math.min(...xs), xMax = Math.max(...xs, xMin + (this.opts.integerX ? 1 : 1e-12));
    const log = this.opts.logY;
    let yMin = this.opts.yMin, yMax = Math.max(...ys, 1e-12) * 1.1;
    if (log) {
      const pos = ys.filter((y) => y > 0);
      yMax = Math.max(...pos, 1e-30) * 1.5;
      yMin = Math.max(Math.min(...pos, yMax) / 1.5, yMax * 1e-6);
    } else if (ys.some((y) => y < 0)) yMin = Math.min(...ys) * 1.1;
    const Y = log
      ? (y) => hgt - pad.b - ((Math.log10(Math.max(y, yMin)) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin) || 1)) * (hgt - pad.t - pad.b)
      : (y) => hgt - pad.b - ((y - yMin) / (yMax - yMin || 1)) * (hgt - pad.t - pad.b);
    return {
      w, hgt, pad, css, xMin, xMax, yMin, yMax,
      X: (x) => pad.l + ((x - xMin) / (xMax - xMin)) * (w - pad.l - pad.r),
      Y,
    };
  }

  draw() {
    const c = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const L = this.layout();
    c.width = Math.round(L.w * dpr);
    c.height = Math.round(L.hgt * dpr);
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, L.w, L.hgt);
    if (!this.points.length) return;
    const root = getComputedStyle(document.documentElement);
    const grid = root.getPropertyValue('--chart-grid').trim();
    const ink2 = root.getPropertyValue('--text-2').trim();
    const muted = root.getPropertyValue('--muted').trim();
    const line = root.getPropertyValue('--chart-line').trim();
    const surface = root.getPropertyValue('--panel').trim();
    g.font = '10px ui-monospace, Menlo, monospace';
    g.lineWidth = 1;
    // grid + y ticks
    const yTicks = [];
    if (this.opts.logY) {
      for (let e = Math.ceil(Math.log10(L.yMin)); e <= Math.floor(Math.log10(L.yMax)); e++) yTicks.push(10 ** e);
      if (yTicks.length > 5) for (let i = yTicks.length - 2; i > 0; i -= 2) yTicks.splice(i, 1);
    } else {
      for (let k = 0; k <= 3; k++) yTicks.push(L.yMin + ((L.yMax - L.yMin) * k) / 3.3);
    }
    for (const yv of yTicks) {
      const y = L.Y(yv);
      g.strokeStyle = grid;
      g.beginPath();
      g.moveTo(L.pad.l, y + 0.5);
      g.lineTo(L.w - L.pad.r, y + 0.5);
      g.stroke();
      g.fillStyle = muted;
      g.textAlign = 'right';
      g.textBaseline = 'middle';
      g.fillText(this.opts.formatY(yv), L.pad.l - 5, y);
    }
    // x ticks
    g.textAlign = 'center';
    g.textBaseline = 'top';
    const span = L.xMax - L.xMin;
    const every = this.opts.integerX ? Math.max(1, Math.ceil(span / this.opts.xTicks)) : niceStep(span, this.opts.xTicks);
    const x0 = this.opts.integerX ? L.xMin : Math.ceil(L.xMin / every) * every;
    for (let xv = x0; xv <= L.xMax + every * 1e-9; xv += every) g.fillText(this.opts.formatX(xv), L.X(xv), L.hgt - L.pad.b + 4);
    g.fillStyle = ink2;
    g.fillText(this.opts.xLabel, (L.pad.l + L.w - L.pad.r) / 2, L.hgt - 12);
    g.strokeStyle = line;
    g.lineWidth = 2;
    g.lineJoin = 'round';
    g.beginPath();
    let started = false;
    this.points.forEach((p) => {
      if (!Number.isFinite(p.y)) { started = false; return; }
      if (started) g.lineTo(L.X(p.x), L.Y(p.y));
      else { g.moveTo(L.X(p.x), L.Y(p.y)); started = true; }
    });
    g.stroke();
    const mark = (i, r) => {
      const p = this.points[i];
      if (!p || !Number.isFinite(p.y)) return;
      g.beginPath();
      g.arc(L.X(p.x), L.Y(p.y), r + 2, 0, Math.PI * 2);
      g.fillStyle = surface;
      g.fill();
      g.beginPath();
      g.arc(L.X(p.x), L.Y(p.y), r, 0, Math.PI * 2);
      g.fillStyle = line;
      g.fill();
    };
    if (this.hover >= 0 && this.hover !== this.current) mark(this.hover, 3);
    mark(this.current, 4);
  }

  onMove(e) {
    if (!this.points.length) return;
    const L = this.layout();
    const r = this.canvas.getBoundingClientRect();
    const mx = e.clientX - r.left;
    let best = -1, bd = Infinity;
    this.points.forEach((p, i) => {
      const d = Math.abs(L.X(p.x) - mx);
      if (d < bd) { bd = d; best = i; }
    });
    this.hover = best;
    const p = this.points[best];
    this.tip.hidden = false;
    this.tip.textContent = this.opts.tipText ? this.opts.tipText(p, best) : `step ${best + 1} · ${this.opts.formatY(p.y)}`;
    const tx = Math.min(L.w - 150, Math.max(0, L.X(p.x) - 60));
    this.tip.style.left = `${tx + 8}px`;
    this.tip.style.top = `${Math.max(0, L.Y(Number.isFinite(p.y) ? p.y : L.yMin) - 16)}px`;
    this.draw();
  }
}
