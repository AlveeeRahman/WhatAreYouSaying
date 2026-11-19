// Hand-drawn SVG helpers for the WAYS figures. Adapted from OpenCarMind's docs/images/src (same author, MIT):
// a seeded wobble makes every render identical. Fonts: Kalam (titles) and Patrick Hand (text) from Google Fonts.
const C = {
  page: '#FBF4E4', card: '#FFFCF4', ink: '#2B2620', soft: '#6B6158',
  orange: '#D9780F', blue: '#2F5FD0', purple: '#7A48B5', green: '#2E8A4E', black: '#2B2620',
  red: '#C8372D', amber: '#F59A2E'
};
const HAND = "'Patrick Hand', 'Comic Sans MS', 'Marker Felt', cursive";
const TITLE = "Kalam, 'Comic Sans MS', 'Marker Felt', cursive";
const NS = 'http://www.w3.org/2000/svg';
const svg = document.getElementById('fig');
const W = +svg.getAttribute('width'), H = +svg.getAttribute('height');

let seed = 20261112;
function rnd() {                       // mulberry32: the same strokes on every render
  seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
const jit = a => (rnd() * 2 - 1) * a;

function el(name, attrs, parent = svg) {
  const e = document.createElementNS(NS, name);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  parent.appendChild(e);
  return e;
}

// A smooth path through points (Catmull-Rom turned into cubic Beziers).
function smooth(p, closed) {
  const n = p.length, g = i => p[closed ? (i + n) % n : Math.max(0, Math.min(n - 1, i))];
  let d = `M${p[0][0].toFixed(1)},${p[0][1].toFixed(1)}`;
  const last = closed ? n : n - 1;
  for (let i = 0; i < last; i++) {
    const a = g(i - 1), b = g(i), c = g(i + 1), e = g(i + 2);
    d += ` C${(b[0] + (c[0] - a[0]) / 6).toFixed(1)},${(b[1] + (c[1] - a[1]) / 6).toFixed(1)}` +
         ` ${(c[0] - (e[0] - b[0]) / 6).toFixed(1)},${(c[1] - (e[1] - b[1]) / 6).toFixed(1)} ${c[0].toFixed(1)},${c[1].toFixed(1)}`;
  }
  return d;
}

// Points around a rounded rectangle, nudged by a slow wobble so the edge looks drawn by hand.
function roundRectPts(x, y, w, h, r, amp) {
  const pts = [], step = 16;
  const ph1 = rnd() * 6.28, ph2 = rnd() * 6.28;
  const side = (x0, y0, x1, y1) => {
    const L = Math.hypot(x1 - x0, y1 - y0), n = Math.max(2, Math.round(L / step));
    const nx = -(y1 - y0) / L, ny = (x1 - x0) / L;
    for (let i = 0; i < n; i++) {
      const t = i / n, s = pts.length;
      const o = amp * (0.7 * Math.sin(s * 0.35 + ph1) + 0.3 * Math.sin(s * 1.1 + ph2));
      pts.push([x0 + (x1 - x0) * t + nx * o, y0 + (y1 - y0) * t + ny * o]);
    }
  };
  const arc = (cx, cy, a0) => {
    for (let i = 0; i < 3; i++) {
      const a = a0 + (i / 3) * Math.PI / 2;
      pts.push([cx + r * Math.cos(a) + jit(amp * 0.4), cy + r * Math.sin(a) + jit(amp * 0.4)]);
    }
  };
  side(x + r, y, x + w - r, y);          arc(x + w - r, y + r, -Math.PI / 2);
  side(x + w, y + r, x + w, y + h - r);  arc(x + w - r, y + h - r, 0);
  side(x + w - r, y + h, x + r, y + h);  arc(x + r, y + h - r, Math.PI / 2);
  side(x, y + h - r, x, y + r);          arc(x + r, y + r, Math.PI);
  return pts;
}

function roughRect(x, y, w, h, r, color, o = {}) {
  const sw = o.sw || 2.4, passes = o.dash ? 1 : 2;
  if (o.fill) el('rect', { x, y, width: w, height: h, rx: r, fill: o.fill });
  for (let k = 0; k < passes; k++) {
    const pts = roundRectPts(x + jit(1), y + jit(1), w + jit(1.5), h + jit(1.5), r, o.amp ?? 1.3);
    const extra = Math.round(pts.length * 0.04);   // overshoot the start a little, like a pen closing a loop
    const open = pts.concat(pts.slice(0, extra + 1)).map(([a, b], i) => i > pts.length ? [a + jit(1.2), b + jit(1.2)] : [a, b]);
    el('path', { d: smooth(open, false), fill: 'none', stroke: color, 'stroke-width': k ? sw * 0.7 : sw,
      'stroke-linecap': 'round', 'stroke-linejoin': 'round', opacity: k ? 0.55 : 1,
      ...(o.dash ? { 'stroke-dasharray': o.dash } : {}) });
  }
}

// A hand-drawn stroke through waypoints, with an optional open arrowhead at the end.
function roughLine(way, color, o = {}) {
  const pts = [];
  for (let s = 0; s < way.length - 1; s++) {
    const [x0, y0] = way[s], [x1, y1] = way[s + 1];
    const L = Math.hypot(x1 - x0, y1 - y0), n = Math.max(3, Math.round(L / 18));
    const bow = (o.bow ?? 0.012) * L * (rnd() < 0.5 ? -1 : 1), nx = -(y1 - y0) / L, ny = (x1 - x0) / L;
    for (let i = s ? 1 : 0; i <= n; i++) {
      const t = i / n, b = bow * Math.sin(Math.PI * t), e = (i === 0 || i === n) ? 0 : 0.6;
      pts.push([x0 + (x1 - x0) * t + nx * b + jit(e), y0 + (y1 - y0) * t + ny * b + jit(e)]);
    }
  }
  const sw = o.sw || 2.4;
  el('path', { d: smooth(pts, false), fill: 'none', stroke: color, 'stroke-width': sw, 'stroke-linecap': 'round',
    ...(o.dash ? { 'stroke-dasharray': o.dash } : {}) });
  if (o.arrow !== false) {
    const [tx, ty] = pts[pts.length - 1], [px, py] = pts[pts.length - 3];
    const a = Math.atan2(ty - py, tx - px), L = 14;
    for (const s of [-1, 1]) {
      const b = a + Math.PI + s * (0.48 + jit(0.06));
      el('path', { d: `M${tx},${ty} Q${tx + Math.cos(b) * L * 0.5 + jit(1)},${ty + Math.sin(b) * L * 0.5 + jit(1)} ${tx + Math.cos(b) * L},${ty + Math.sin(b) * L}`,
        fill: 'none', stroke: color, 'stroke-width': sw, 'stroke-linecap': 'round' });
    }
  }
}

function text(x, y, s, o = {}) {
  const t = el('text', { x, y, 'font-family': o.font || HAND, 'font-size': o.size || 20, fill: o.color || C.ink,
    'font-weight': o.weight || 400, 'text-anchor': o.anchor || 'start' }, o.parent || svg);
  t.textContent = s;
  return t;
}

// Word-wrap s into lines no wider than maxW. Returns the baseline after the last line.
function wrap(x, y, s, maxW, lh, o = {}) {
  const words = s.split(' ');
  let line = '', t = text(x, y, '', o);
  for (const w of words) {
    const tryLine = line ? line + ' ' + w : w;
    t.textContent = tryLine;
    if (t.getComputedTextLength() > maxW && line) {
      t.textContent = line; y += lh; line = w; t = text(x, y, w, o);
    } else line = tryLine;
  }
  t.textContent = line;
  return y;
}

function numCircle(cx, cy, n, color) {
  el('circle', { cx, cy, r: 20, fill: C.page });
  const pts = [];
  for (let i = 0; i < 22; i++) {
    const a = i / 20 * Math.PI * 2 - 1.2, r = 20 + jit(0.9);
    pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  el('path', { d: smooth(pts, false), fill: 'none', stroke: color, 'stroke-width': 2.6, 'stroke-linecap': 'round' });
  text(cx, cy + 8, String(n), { font: TITLE, weight: 700, size: 22, color, anchor: 'middle' });
}

function card(n, x, y, w, h, color, title, lane, o = {}) {
  roughRect(x, y, w, h, 18, color, { fill: C.card, sw: 2.6 });
  if (n) numCircle(x + 6, y + 6, n, color);
  const ty = wrap(x + 36, y + 40, title, w - 56, 30, { font: TITLE, weight: 700, size: o.tsize || 25, color });
  text(x + 36, ty + 24, lane, { size: 18, color, weight: 400 }).setAttribute('opacity', '0.8');
  return ty + 24;                         // baseline of the lane line
}

function pill(x, y, label, color) {
  const t = text(0, 0, label, { size: 19, color });
  const w = t.getComputedTextLength() + 30;
  t.setAttribute('x', x + 15); t.setAttribute('y', y + 22);
  roughRect(x, y, w, 32, 16, color, { sw: 2, amp: 0.8 });
  return x + w;
}

function star(cx, cy, R, color) {
  const pts = [];
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + i * Math.PI / 5, r = (i % 2 ? R * 0.45 : R) + jit(0.8);
    pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  const d = 'M' + pts.map(p => p.map(v => v.toFixed(1)).join(',')).join(' L') + ' Z';
  el('path', { d, fill: color, stroke: '#B8650C', 'stroke-width': 2, 'stroke-linejoin': 'round' });
}


function overflowCheck() {
  let over = [];
  for (const t of svg.querySelectorAll('text')) { const b = t.getBBox(); if (b.x + b.width > W - 8 || b.y + b.height > H) over.push(t.textContent); }
  document.body.dataset.overflow = over.join(' | ');
}
function whenFonts(fn) {
  const want = ["700 25px Kalam", "400 20px 'Patrick Hand'"];
  Promise.all(want.map(f => document.fonts.load(f))).then(() => {
    fn(); overflowCheck();
    document.body.dataset.fonts = want.map(f => document.fonts.check(f) ? 'ok' : 'MISSING').join(',');
  });
}
