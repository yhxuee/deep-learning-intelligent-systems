// Interactive figures for "Deep Learning and Intelligent Systems".
// Dependency-free: any element <div class="dlis-widget" data-widget="NAME"></div> becomes a live explorer.
// Labels follow <html lang>: English, Traditional Chinese (zh-TW) or Simplified Chinese (zh-CN).

const LANG = (document.documentElement.lang || '').toLowerCase();
const ZH = LANG.startsWith('zh');
const T = (en, tw, cn) => (LANG === 'zh-cn' ? cn ?? tw : ZH ? tw : en);
const NS = 'http://www.w3.org/2000/svg';
const fmt = (v, d = 3) => (Number.isFinite(v) ? (Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(d)) : '∞');

// ------------------------------------------------------------------ helpers
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'on') for (const [ev, fn] of Object.entries(v)) e.addEventListener(ev, fn);
    else if (v !== undefined && v !== null) e.setAttribute(k, v);
  }
  for (const c of kids.flat()) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}
function svgEl(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
let uid = 0;
function slider(label, { min, max, step, value, show = (v) => v }, onInput) {
  const id = `dlis-s${++uid}`;
  const out = h('output', { for: id });
  const input = h('input', { type: 'range', id, min, max, step, value });
  const sync = () => { out.textContent = show(+input.value); };
  input.addEventListener('input', () => { sync(); onInput(); });
  sync();
  return { el: h('div', { class: 'dlis-ctl' }, h('label', { for: id }, label), input, out), get: () => +input.value, set: (v) => { input.value = v; sync(); } };
}
function select(label, options, onChange) {
  const id = `dlis-s${++uid}`;
  const sel = h('select', { id, on: { change: onChange } }, options.map(([v, t]) => h('option', { value: v }, t)));
  return { el: h('div', { class: 'dlis-ctl' }, h('label', { for: id }, label), sel), get: () => sel.value };
}
function button(label, onClick) { return h('button', { type: 'button', class: 'dlis-btn', on: { click: onClick } }, label); }
function checkbox(label, checked, onChange) {
  const id = `dlis-s${++uid}`;
  const cb = h('input', { type: 'checkbox', id, on: { change: onChange } });
  cb.checked = checked;
  return { el: h('div', { class: 'dlis-ctl dlis-check' }, cb, h('label', { for: id }, label)), get: () => cb.checked };
}
function rng(seed) {
  let a = seed >>> 0;
  const uni = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const normal = () => Math.sqrt(-2 * Math.log(1 - uni())) * Math.cos(2 * Math.PI * uni());
  return { uni, normal };
}
function niceTicks(a, b, n = 5) {
  const span = b - a; if (!(span > 0)) return [a];
  const raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const ticks = []; for (let t = Math.ceil(a / step) * step; t <= b + 1e-9 * span; t += step) ticks.push(+t.toFixed(10));
  return ticks;
}
function softmax(s) { const m = Math.max(...s); const e = s.map((x) => Math.exp(x - m)); const z = e.reduce((p, q) => p + q, 0); return e.map((x) => x / z); }
const entropy = (p) => -p.reduce((acc, x) => acc + (x > 0 ? x * Math.log(x) : 0), 0);

// Line plot: series [{pts:[[x,y]], cls, dots, label}], optional hlines/vlines/bands, logY.
function linePlot(svg, { x, y, series, xlabel, ylabel, hlines = [], vlines = [], logY = false, w = 640, hgt = 300, ariaLabel }) {
  svg.replaceChildren();
  svg.setAttribute('viewBox', `0 0 ${w} ${hgt}`);
  svg.setAttribute('role', 'img');
  if (ariaLabel) svg.setAttribute('aria-label', ariaLabel);
  const m = { l: 56, r: 14, t: 12, b: 42 };
  const ty = (v) => (logY ? Math.log10(Math.max(v, 1e-300)) : v);
  const [x0, x1] = x, [y0, y1] = logY ? [Math.log10(y[0]), Math.log10(y[1])] : y;
  const sx = (v) => m.l + ((v - x0) / (x1 - x0)) * (w - m.l - m.r);
  const sy = (v) => hgt - m.b - ((ty(v) - y0) / (y1 - y0)) * (hgt - m.t - m.b);
  const clipId = `dlis-clip${++uid}`;
  const defs = svgEl('defs', {}, svg);
  svgEl('rect', { x: m.l, y: m.t, width: w - m.l - m.r, height: hgt - m.t - m.b }, svgEl('clipPath', { id: clipId }, defs));
  const grid = svgEl('g', { class: 'dlis-grid' }, svg);
  for (const t of niceTicks(x0, x1)) {
    svgEl('line', { x1: sx(t), x2: sx(t), y1: m.t, y2: hgt - m.b }, grid);
    svgEl('text', { x: sx(t), y: hgt - m.b + 16, 'text-anchor': 'middle', class: 'dlis-tick' }, svg).textContent = +t.toPrecision(6);
  }
  const yt = logY ? niceTicks(Math.ceil(y0), Math.floor(y1), 5).filter(Number.isInteger) : niceTicks(y0, y1);
  for (const t of yt) {
    const py = hgt - m.b - ((t - y0) / (y1 - y0)) * (hgt - m.t - m.b);
    svgEl('line', { x1: m.l, x2: w - m.r, y1: py, y2: py }, grid);
    svgEl('text', { x: m.l - 6, y: py + 4, 'text-anchor': 'end', class: 'dlis-tick' }, svg).textContent = logY ? `1e${t}` : +t.toPrecision(6);
  }
  svgEl('line', { x1: m.l, x2: m.l, y1: m.t, y2: hgt - m.b, class: 'dlis-axis' }, svg);
  svgEl('line', { x1: m.l, x2: w - m.r, y1: hgt - m.b, y2: hgt - m.b, class: 'dlis-axis' }, svg);
  if (xlabel) svgEl('text', { x: (m.l + w - m.r) / 2, y: hgt - 6, 'text-anchor': 'middle', class: 'dlis-axlabel' }, svg).textContent = xlabel;
  if (ylabel) svgEl('text', { x: 14, y: (m.t + hgt - m.b) / 2, 'text-anchor': 'middle', transform: `rotate(-90 14 ${(m.t + hgt - m.b) / 2})`, class: 'dlis-axlabel' }, svg).textContent = ylabel;
  const g = svgEl('g', { 'clip-path': `url(#${clipId})` }, svg);
  for (const { y: v, label } of hlines) {
    svgEl('line', { x1: m.l, x2: w - m.r, y1: sy(v), y2: sy(v), class: 'dlis-ref' }, g);
    if (label) svgEl('text', { x: w - m.r - 4, y: sy(v) - 4, 'text-anchor': 'end', class: 'dlis-tick' }, g).textContent = label;
  }
  for (const { x: v, label } of vlines) {
    svgEl('line', { x1: sx(v), x2: sx(v), y1: m.t, y2: hgt - m.b, class: 'dlis-ref' }, g);
    if (label) svgEl('text', { x: sx(v) + 4, y: m.t + 12, class: 'dlis-tick' }, g).textContent = label;
  }
  for (const s of series) {
    const pts = s.pts.filter(([, v]) => Number.isFinite(v));
    if (!pts.length) continue;
    svgEl('path', { d: pts.map(([a, b], i) => `${i ? 'L' : 'M'}${sx(a).toFixed(1)},${sy(b).toFixed(1)}`).join(''), class: `dlis-line ${s.cls}` }, g);
    if (s.dots) for (const [a, b] of pts) svgEl('circle', { cx: sx(a), cy: sy(b), r: 4, class: `dlis-dot ${s.cls}` }, g);
  }
  return { sx, sy };
}
function legend(items) {
  return h('div', { class: 'dlis-legend' }, items.map(([cls, label]) => h('span', {}, h('i', { class: `dlis-swatch ${cls}` }), label)));
}
function frame(root, title, note) {
  root.replaceChildren();
  root.classList.add('dlis-ready');
  const svg = svgEl('svg', { class: 'dlis-svg' });
  const controls = h('div', { class: 'dlis-controls' });
  const readout = h('div', { class: 'dlis-readout', 'aria-live': 'polite' });
  root.append(h('div', { class: 'dlis-title' }, h('span', { class: 'dlis-badge' }, T('Interactive', '互動', '交互')), title), controls, svg, readout);
  if (note) root.append(h('p', { class: 'dlis-note' }, note));
  return { svg, controls, readout };
}
const kv = (pairs) => pairs.map(([k, v]) => h('span', {}, h('b', {}, k), ' ', v));

// ------------------------------------------------------------------ Unit 03
const widgets = {};

widgets['gd-parabola'] = (root) => {
  const { svg, controls, readout } = frame(root, T('Gradient descent on a parabola, L(w) = (a/2)·w²', '拋物線上的梯度下降，L(w) = (a/2)·w²', '抛物线上的梯度下降，L(w) = (a/2)·w²'),
    T('Try η just below and just above 2/a. §3.2.3 derives the boundary.', '試著把 η 調到略低於、略高於 2/a。§3.2.3 推導了這個界線。', '试着把 η 调到略低于、略高于 2/a。§3.2.3 推导了这个界线。'));
  const a = slider(T('curvature a', '曲率 a', '曲率 a'), { min: 0.5, max: 8, step: 0.5, value: 4 }, draw);
  const eta = slider(T('learning rate η', '學習率 η', '学习率 η'), { min: 0.01, max: 0.8, step: 0.01, value: 0.1 }, draw);
  controls.append(a.el, eta.el);
  function draw() {
    const f = 1 - eta.get() * a.get();
    const pts = Array.from({ length: 16 }, (_, k) => [k, f ** k]);
    linePlot(svg, { x: [0, 15], y: [-3, 3], series: [{ pts, cls: 's1', dots: true }], hlines: [{ y: 0 }], xlabel: T('iteration k', '迭代次數 k', '迭代次数 k'), ylabel: 'w_k',
      ariaLabel: T(`Iterates for factor ${fmt(f)}`, `每步乘以 ${fmt(f)} 的迭代值`, `每步乘以 ${fmt(f)} 的迭代值`) });
    const lim = 2 / a.get();
    const state = Math.abs(f) < 1e-9 ? T('exact minimum in one step', '一步到達最小值', '一步到达最小值') : f > 0 && f < 1 ? T('monotone convergence', '單調收斂', '单调收敛') : f < 0 && f > -1 ? T('converges while oscillating', '振盪中收斂', '振荡中收敛') : Math.abs(f + 1) < 1e-9 ? T('oscillates forever', '永遠振盪', '永远振荡') : T('diverges', '發散', '发散');
    readout.replaceChildren(...kv([[T('factor 1 − ηa =', '每步因子 1 − ηa =', '每步因子 1 − ηa ='), fmt(f)], [T('stable iff η <', '穩定條件 η <', '稳定条件 η <'), fmt(lim)], [T('behaviour:', '行為：', '行为：'), state]]));
  }
  draw();
};

widgets['gd-2d'] = (root) => {
  const { svg, controls, readout } = frame(root, T('Ill-conditioning and momentum on L = ½(θ₁² + κθ₂²)', '病態條件與動量：L = ½(θ₁² + κθ₂²)', '病态条件与动量：L = ½(θ₁² + κθ₂²)'),
    T('η is set as a multiple of 2/λmax. Compare the paths and step counts with §3.5.4 and §3.5.7.', 'η 以 2/λmax 的倍數設定。請與 §3.5.4、§3.5.7 的推導比較路徑與步數。', 'η 以 2/λmax 的倍数设定。请与 §3.5.4、§3.5.7 的推导比较路径与步数。'));
  const kappa = slider(T('condition number κ', '條件數 κ', '条件数 κ'), { min: 1, max: 100, step: 1, value: 10 }, draw);
  const frac = slider(T('η × λmax', 'η × λmax', 'η × λmax'), { min: 0.05, max: 2.1, step: 0.01, value: 1.82 }, draw);
  const beta = slider(T('momentum β', '動量 β', '动量 β'), { min: 0, max: 0.98, step: 0.01, value: 0 }, draw);
  const best = button(T('use optimal GD step', '套用最佳 GD 步長', '应用最优 GD 步长'), () => { const k = kappa.get(); frac.set(2 * k / (1 + k)); beta.set(0); draw(); });
  const bestM = button(T('use optimal momentum', '套用最佳動量', '应用最优动量'), () => { const k = kappa.get(); const r = (Math.sqrt(k) - 1) / (Math.sqrt(k) + 1); frac.set(k * 4 / (Math.sqrt(k) + 1) ** 2); beta.set(Math.min(0.98, r * r)); draw(); });
  controls.append(kappa.el, frac.el, beta.el, h('div', { class: 'dlis-ctl' }, best, bestM));
  function run(steps) {
    const k = kappa.get(), eta = frac.get() / k, b = beta.get();
    let x = [1, 1], prev = [1, 1]; const path = [[1, 1]]; let hit = -1;
    for (let i = 1; i <= steps; i++) {
      const nx = [x[0] - eta * x[0] + b * (x[0] - prev[0]), x[1] - eta * k * x[1] + b * (x[1] - prev[1])];
      prev = x; x = nx; path.push(x);
      if (hit < 0 && Math.hypot(x[0], x[1]) < 1e-3 * Math.SQRT2) hit = i;
      if (!Number.isFinite(x[0] + x[1]) || Math.abs(x[0]) + Math.abs(x[1]) > 1e6) break;
    }
    return { path, hit };
  }
  function draw() {
    const k = kappa.get();
    svg.replaceChildren();
    const W = 640, H = 360, sx = (v) => W / 2 + v * 140 * (H / 360) * 1.1, sy = (v) => H / 2 - v * 140;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    for (const c of [0.02, 0.1, 0.3, 0.6, 1.0, 2.0]) {
      const pts = Array.from({ length: 81 }, (_, i) => { const t = (i / 80) * 2 * Math.PI; return [Math.sqrt(2 * c) * Math.cos(t), Math.sqrt((2 * c) / k) * Math.sin(t)]; });
      svgEl('path', { d: pts.map(([a, b], i) => `${i ? 'L' : 'M'}${sx(a)},${sy(b)}`).join('') + 'Z', class: 'dlis-contour' }, svg);
    }
    const { path, hit } = run(2000);
    const shown = path.slice(0, 61).filter(([a, b]) => Math.abs(a) < 3 && Math.abs(b) < 2);
    svgEl('path', { d: shown.map(([a, b], i) => `${i ? 'L' : 'M'}${sx(a).toFixed(1)},${sy(b).toFixed(1)}`).join(''), class: 'dlis-line s2' }, svg);
    for (const [a, b] of shown) svgEl('circle', { cx: sx(a), cy: sy(b), r: 3, class: 'dlis-dot s2' }, svg);
    svgEl('text', { x: sx(0) + 6, y: sy(0) - 6, class: 'dlis-tick' }, svg).textContent = '★';
    svg.setAttribute('aria-label', T('Optimization path on elliptical contours', '橢圓等高線上的最佳化路徑', '椭圆等高线上的优化路径'));
    const eta = frac.get() / k;
    const gdRate = Math.max(Math.abs(1 - eta), Math.abs(1 - eta * k));
    readout.replaceChildren(...kv([
      [T('GD factor max|1−ηλ| =', 'GD 每步因子 max|1−ηλ| =', 'GD 每步因子 max|1−ηλ| ='), fmt(gdRate)],
      [T('best GD factor (κ−1)/(κ+1) =', '最佳 GD 因子 (κ−1)/(κ+1) =', '最优 GD 因子 (κ−1)/(κ+1) ='), fmt((k - 1) / (k + 1))],
      [T('best momentum factor =', '最佳動量因子 =', '最优动量因子 ='), fmt((Math.sqrt(k) - 1) / (Math.sqrt(k) + 1))],
      [T('steps to 10⁻³:', '誤差降到 10⁻³ 的步數：', '误差降到 10⁻³ 的步数：'), hit > 0 ? hit : T('not within 2000 (or diverged)', '2000 步內未達成（或發散）', '2000 步内未达成（或发散）')],
    ]));
  }
  draw();
};

widgets['backprop-neuron'] = (root) => {
  const { svg, controls, readout } = frame(root, T('Forward and backward through one neuron', '單一神經元的前向與反向傳播', '单一神经元的前向与反向传播'),
    T('Every adjoint is computed with the rule of §3.3.3; the last row checks ∂L/∂w₁ with a central difference.', '每個伴隨值都用 §3.3.3 的規則計算；最後一列以中央差分檢查 ∂L/∂w₁。', '每个伴随值都用 §3.3.3 的规则计算；最后一行以中心差分检查 ∂L/∂w₁。'));
  svg.remove();
  const p = {};
  for (const [k, v, lo, hi] of [['x1', 1, -2, 2], ['x2', 2, -2, 2], ['w1', 0.5, -2, 2], ['w2', -0.25, -2, 2], ['b', 0.1, -2, 2], ['t', 0, -1, 1]]) {
    p[k] = slider(k.replace(/(\d)/, (d) => '₀₁₂₃'[d]), { min: lo, max: hi, step: 0.05, value: v, show: (x) => x.toFixed(2) }, draw);
    controls.append(p[k].el);
  }
  const table = h('table', { class: 'dlis-table' });
  root.insertBefore(table, readout);
  function f(w1) { const z = w1 * p.x1.get() + p.w2.get() * p.x2.get() + p.b.get(); const a = Math.tanh(z); return 0.5 * (a - p.t.get()) ** 2; }
  function draw() {
    const x1 = p.x1.get(), x2 = p.x2.get(), w1 = p.w1.get(), w2 = p.w2.get(), b = p.b.get(), t = p.t.get();
    const z = w1 * x1 + w2 * x2 + b, a = Math.tanh(z), d = a - t, L = 0.5 * d * d;
    const abar = d, zbar = abar * (1 - a * a);
    const rows = [
      [T('forward', '前向', '前向'), 'z = w₁x₁ + w₂x₂ + b', fmt(z, 6)], ['', 'a = tanh z', fmt(a, 6)], ['', 'L = ½(a − t)²', fmt(L, 6)],
      [T('backward', '反向', '反向'), 'L̄ = 1', '1'], ['', 'ā = L̄·(a − t)', fmt(abar, 6)], ['', 'z̄ = ā·(1 − a²)', fmt(zbar, 6)],
      ['', 'w̄₁ = z̄·x₁', fmt(zbar * x1, 6)], ['', 'w̄₂ = z̄·x₂', fmt(zbar * x2, 6)], ['', 'b̄ = z̄', fmt(zbar, 6)],
      [T('check', '檢查', '检查'), T('central difference for w̄₁', 'w̄₁ 的中央差分', 'w̄₁ 的中心差分'), fmt((f(w1 + 1e-6) - f(w1 - 1e-6)) / 2e-6, 6)],
    ];
    table.replaceChildren(h('tbody', {}, rows.map((r) => h('tr', {}, r.map((c, i) => h(i === 2 ? 'td' : i === 0 ? 'th' : 'td', i === 2 ? { class: 'num' } : {}, c))))));
    readout.replaceChildren(...kv([[T('Default values reproduce the numbers of §3.3.3:', '預設值重現 §3.3.3 的數字：', '默认值复现 §3.3.3 的数字：'), 'w̄₁ = 0.098678, w̄₂ = 0.197356']]));
  }
  draw();
};

// ------------------------------------------------------------------ Unit 04
function bars(svg, labels, values, { yMax = 1, ariaLabel, highlight = -1, valueFmt = (v) => v.toFixed(3) } = {}) {
  svg.replaceChildren();
  const W = 640, H = 240, m = { l: 40, r: 10, t: 14, b: 36 }, n = values.length;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('role', 'img'); if (ariaLabel) svg.setAttribute('aria-label', ariaLabel);
  const bw = (W - m.l - m.r) / n;
  for (const t of niceTicks(0, yMax, 4)) { const y = H - m.b - (t / yMax) * (H - m.t - m.b); svgEl('line', { x1: m.l, x2: W - m.r, y1: y, y2: y, class: 'dlis-gridline' }, svg); svgEl('text', { x: m.l - 6, y: y + 4, 'text-anchor': 'end', class: 'dlis-tick' }, svg).textContent = t; }
  values.forEach((v, i) => {
    const bh = (Math.min(v, yMax) / yMax) * (H - m.t - m.b), x = m.l + i * bw + bw * 0.18;
    svgEl('rect', { x, y: H - m.b - bh, width: bw * 0.64, height: Math.max(bh, 0.5), rx: 4, class: `dlis-bar ${i === highlight ? 's2' : 's1'}` }, svg);
    svgEl('text', { x: x + bw * 0.32, y: H - m.b - bh - 4, 'text-anchor': 'middle', class: 'dlis-tick' }, svg).textContent = valueFmt(v);
    svgEl('text', { x: x + bw * 0.32, y: H - m.b + 18, 'text-anchor': 'middle', class: 'dlis-axlabel' }, svg).textContent = labels[i];
  });
}

widgets['softmax-temperature'] = (root) => {
  const { svg, controls, readout } = frame(root, T('Softmax and temperature', 'Softmax 與溫度', 'Softmax 与温度'),
    T('Lower τ sharpens toward argmax; higher τ flattens toward uniform (§4.2.6, §4.5.2).', 'τ 越低越接近 argmax，越高越接近均勻分布（§4.2.6、§4.5.2）。', 'τ 越低越接近 argmax，越高越接近均匀分布（§4.2.6、§4.5.2）。'));
  const s = [2, 1, 0, -1].map((v, i) => slider(`s${'₁₂₃₄'[i]}`, { min: -3, max: 3, step: 0.1, value: v, show: (x) => x.toFixed(1) }, draw));
  const tau = slider(T('temperature τ', '溫度 τ', '温度 τ'), { min: 0.05, max: 5, step: 0.05, value: 1, show: (x) => x.toFixed(2) }, draw);
  controls.append(...s.map((x) => x.el), tau.el);
  function draw() {
    const sc = s.map((x) => x.get()), p = softmax(sc.map((v) => v / tau.get()));
    bars(svg, ['1', '2', '3', '4'], p, { ariaLabel: T('softmax probabilities', 'softmax 機率', 'softmax 概率') });
    const lse = tau.get() * Math.log(sc.reduce((acc, v) => acc + Math.exp(v / tau.get()), 0));
    readout.replaceChildren(...kv([[T('entropy H(p) =', '熵 H(p) =', '熵 H(p) ='), `${fmt(entropy(p))} / ln 4 = ${fmt(Math.log(4))}`], ['max s =', fmt(Math.max(...sc))], ['τ·log Σ e^{s/τ} =', fmt(lse)]]));
  }
  draw();
};

widgets['soft-lookup'] = (root) => {
  const { svg, controls, readout } = frame(root, T('Soft lookup with queries, keys and values', '以查詢、鍵、值做軟查找', '以查询、键、值做软查找'),
    T('The construction of §4.2.7: the query reads the query slot, keys read key slots, values read the value slot. Shuffle: weights follow content, not position.', '§4.2.7 的構造：查詢讀取「查詢欄」、鍵讀取「鍵欄」、值讀取「值欄」。打亂順序後，權重仍跟著內容走，而不是位置。', '§4.2.7 的构造：查询读取“查询槽”、键读取“键槽”、值读取“值槽”。打乱顺序后，权重仍跟着内容走，而不是位置。'));
  let pairs = [['a', 3], ['b', 7], ['c', 1]];
  const beta = slider(T('score scale β', '分數尺度 β', '分数尺度 β'), { min: 0, max: 10, step: 0.1, value: 4, show: (x) => x.toFixed(1) }, draw);
  const q = select(T('query', '查詢', '查询'), [['b', '?b'], ['a', '?a'], ['c', '?c']], draw);
  controls.append(beta.el, q.el, h('div', { class: 'dlis-ctl' }, button(T('shuffle pairs', '打亂鍵值對', '打乱键值对'), () => { pairs = pairs.map((x) => [Math.random(), x]).sort((u, v) => u[0] - v[0]).map((x) => x[1]); draw(); })));
  function draw() {
    const qk = q.get();
    const scores = [...pairs.map(([k]) => (k === qk ? beta.get() : 0)), 0];
    const w = softmax(scores), vals = [...pairs.map(([, v]) => v), 0];
    const y = w.reduce((acc, wi, i) => acc + wi * vals[i], 0);
    const hit = pairs.findIndex(([k]) => k === qk);
    bars(svg, [...pairs.map(([k, v]) => `${k}→${v}`), `?${qk}`], w, { highlight: hit, ariaLabel: T('attention weights', '注意力權重', '注意力权重') });
    readout.replaceChildren(...kv([[T('output y = Σ wⱼvⱼ =', '輸出 y = Σ wⱼvⱼ =', '输出 y = Σ wⱼvⱼ ='), fmt(y)], [T('hard-lookup answer:', '硬查找答案：', '硬查找答案：'), pairs[hit][1]], [T('weight on match =', '匹配位置的權重 =', '匹配位置的权重 ='), fmt(w[hit])]]));
  }
  draw();
};

widgets['attention-scaling'] = (root) => {
  const { svg, controls, readout } = frame(root, T('Why divide by √d_k? Attention weights for random queries and keys', '為什麼要除以 √d_k？隨機查詢與鍵的注意力權重', '为什么要除以 √d_k？随机查询与键的注意力权重'),
    T('Each row is one query’s softmax over 8 keys (q, k ~ N(0, 1)). Without scaling, rows collapse toward one-hot as d_k grows (§4.2.8).', '每一列是一個查詢對 8 個鍵的 softmax（q、k ~ N(0, 1)）。不縮放時，d_k 越大，每列越接近 one-hot（§4.2.8）。', '每一行是一个查询对 8 个键的 softmax（q、k ~ N(0, 1)）。不缩放时，d_k 越大，每行越接近 one-hot（§4.2.8）。'));
  const dims = [2, 4, 8, 16, 32, 64, 128, 256, 512];
  const dk = slider('d_k', { min: 0, max: dims.length - 1, step: 1, value: 4, show: (i) => dims[i] }, draw);
  const scaled = checkbox(T('divide scores by √d_k', '分數除以 √d_k', '分数除以 √d_k'), true, draw);
  const causal = checkbox(T('causal mask', '因果遮罩', '因果遮罩'), false, draw);
  let seed = 7;
  controls.append(dk.el, scaled.el, causal.el, h('div', { class: 'dlis-ctl' }, button(T('new random vectors', '重新抽樣', '重新采样'), () => { seed++; draw(); })));
  function draw() {
    const d = dims[dk.get()], r = rng(seed), N = 8;
    const Q = Array.from({ length: N }, () => Array.from({ length: d }, r.normal));
    const K = Array.from({ length: N }, () => Array.from({ length: d }, r.normal));
    const A = Q.map((q, i) => softmax(K.map((k, j) => (causal.get() && j > i ? -Infinity : q.reduce((acc, v, m) => acc + v * k[m], 0) / (scaled.get() ? Math.sqrt(d) : 1)))));
    svg.replaceChildren(); const W = 640, H = 330, c = 34, x0 = 40, y0 = 20;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', T('8 by 8 attention heat map', '8×8 注意力熱圖', '8×8 注意力热图'));
    A.forEach((row, i) => row.forEach((v, j) => {
      svgEl('rect', { x: x0 + j * c, y: y0 + i * c, width: c - 2, height: c - 2, rx: 3, class: 'dlis-cell', style: `fill-opacity:${(0.06 + 0.94 * v).toFixed(3)}` }, svg);
      if (v > 0.5) svgEl('text', { x: x0 + j * c + c / 2 - 1, y: y0 + i * c + c / 2 + 4, 'text-anchor': 'middle', class: 'dlis-celltext' }, svg).textContent = v.toFixed(2);
    }));
    svgEl('text', { x: x0, y: y0 + N * c + 18, class: 'dlis-axlabel' }, svg).textContent = T('columns: keys · rows: queries', '欄：鍵 · 列：查詢', '列：键 · 行：查询');
    const meanMax = A.reduce((acc, r) => acc + Math.max(...r), 0) / N, meanH = A.reduce((acc, r) => acc + entropy(r), 0) / N;
    const tx = x0 + N * c + 30;
    [[T('mean largest weight', '平均最大權重', '平均最大权重'), fmt(meanMax)], [T('mean entropy (nats)', '平均熵（nats）', '平均熵（nats）'), `${fmt(meanH)} / ${fmt(Math.log(8))}`], [T('score std ≈', '分數標準差 ≈', '分数标准差 ≈'), fmt(scaled.get() ? 1 : Math.sqrt(d))]].forEach(([k, v], i) => {
      svgEl('text', { x: tx, y: y0 + 30 + i * 46, class: 'dlis-axlabel' }, svg).textContent = k;
      svgEl('text', { x: tx, y: y0 + 52 + i * 46, class: 'dlis-big' }, svg).textContent = v;
    });
    readout.replaceChildren(...kv([[T('Each row sums to 1:', '每列總和為 1：', '每行总和为 1：'), A.every((row) => Math.abs(row.reduce((p, q) => p + q, 0) - 1) < 1e-9) ? '✓' : '✗']]));
  }
  draw();
};

// ------------------------------------------------------------------ Unit 05
widgets['ema-filter'] = (root) => {
  const { svg, controls, readout } = frame(root, T('The moving average on a noisy sensor', '雜訊感測器上的指數移動平均', '噪声传感器上的指数移动平均'),
    T('A genuine step from 20 to 23 °C at t = 40 and a glitch at t = 70. No single α is both quiet and fast (§5.2.6).', '在 t = 40 有真正的 20→23 °C 階躍，在 t = 70 有一次突波。沒有任何單一 α 能同時安靜又快速（§5.2.6）。', '在 t = 40 有真正的 20→23 °C 阶跃，在 t = 70 有一次尖峰。没有任何单一 α 能同时安静又快速（§5.2.6）。'));
  const alpha = slider('α', { min: 0.3, max: 0.995, step: 0.005, value: 0.9, show: (x) => x.toFixed(3) }, draw);
  let seed = 3;
  controls.append(alpha.el, h('div', { class: 'dlis-ctl' }, button(T('new noise', '重新產生雜訊', '重新产生噪声'), () => { seed++; draw(); })));
  function draw() {
    const r = rng(seed), a = alpha.get();
    const x = Array.from({ length: 120 }, (_, t) => (t < 40 ? 20 : 23) + 0.8 * r.normal() + (t === 70 ? 6 : 0));
    let hcur = 20; const hs = x.map((v) => (hcur = a * hcur + (1 - a) * v));
    linePlot(svg, { x: [0, 119], y: [17, 30], series: [{ pts: x.map((v, t) => [t, v]), cls: 's0' }, { pts: hs.map((v, t) => [t, v]), cls: 's1' }],
      xlabel: T('time step t (s)', '時間步 t（秒）', '时间步 t（秒）'), ylabel: '°C', vlines: [{ x: 40 }], ariaLabel: T('sensor readings and moving average', '感測讀數與移動平均', '传感器读数与移动平均') });
    readout.replaceChildren(legend([['s0', T('reading xₜ', '讀數 xₜ', '读数 xₜ')], ['s1', T('EMA hₜ', 'EMA hₜ', 'EMA hₜ')]]), ...kv([
      [T('half-life', '半衰期', '半衰期'), `${fmt(Math.log(0.5) / Math.log(a), 2)} ${T('steps', '步', '步')}`],
      [T('95% response', '95% 響應時間', '95% 响应时间'), `${fmt(Math.log(0.05) / Math.log(a), 1)} ${T('steps', '步', '步')}`],
      [T('noise std factor √((1−α)/(1+α))', '雜訊標準差因子 √((1−α)/(1+α))', '噪声标准差因子 √((1−α)/(1+α))'), fmt(Math.sqrt((1 - a) / (1 + a)))],
      [T('time constant τ = −1/ln α', '時間常數 τ = −1/ln α', '时间常数 τ = −1/ln α'), `${fmt(-1 / Math.log(a), 2)} s`],
    ]));
  }
  draw();
};

widgets['discretization'] = (root) => {
  const { svg, controls, readout } = frame(root, T('Exact versus forward-Euler discretization of τ·dh/dt = x − h', 'τ·dh/dt = x − h 的精確離散化與前向歐拉法', 'τ·dh/dt = x − h 的精确离散化与前向欧拉法'),
    T('Step response with x = 1. Euler leaves the stable band when Δt > 2τ (§5.2.8).', '輸入 x = 1 的階躍響應。當 Δt > 2τ 時，歐拉法離開穩定區（§5.2.8）。', '输入 x = 1 的阶跃响应。当 Δt > 2τ 时，欧拉法离开稳定区（§5.2.8）。'));
  const r = slider('Δt / τ', { min: 0.05, max: 3, step: 0.05, value: 1, show: (x) => x.toFixed(2) }, draw);
  controls.append(r.el);
  function draw() {
    const ae = Math.exp(-r.get()), ap = 1 - r.get();
    let he = 0, hp = 0; const pe = [[0, 0]], pp = [[0, 0]];
    for (let t = 1; t <= 20; t++) { he = ae * he + (1 - ae); hp = ap * hp + (1 - ap); pe.push([t, he]); pp.push([t, hp]); }
    linePlot(svg, { x: [0, 20], y: [-2, 3], series: [{ pts: pe, cls: 's1', dots: true }, { pts: pp, cls: 's2', dots: true }], hlines: [{ y: 1, label: T('target', '目標', '目标') }],
      xlabel: T('step', '步', '步'), ylabel: 'hₜ', ariaLabel: T('step responses', '階躍響應', '阶跃响应') });
    readout.replaceChildren(legend([['s1', T('exact (zero-order hold)', '精確（零階保持）', '精确（零阶保持）')], ['s2', T('forward Euler', '前向歐拉法', '前向欧拉法')]]), ...kv([
      [T('exact α = e^(−Δt/τ) =', '精確 α = e^(−Δt/τ) =', '精确 α = e^(−Δt/τ) ='), fmt(ae)], [T('Euler α = 1 − Δt/τ =', '歐拉 α = 1 − Δt/τ =', '欧拉 α = 1 − Δt/τ ='), fmt(ap)],
      [T('Euler is', '歐拉法', '欧拉法'), Math.abs(ap) < 1 ? T('stable', '穩定', '稳定') : T('UNSTABLE', '不穩定', '不稳定')],
    ]));
  }
  draw();
};

widgets['eigen-modes'] = (root) => {
  const { svg, controls, readout } = frame(root, T('A 2×2 memory: eigenvalues, stability and transient growth', '2×2 記憶：特徵值、穩定性與暫態增長', '2×2 记忆：特征值、稳定性与瞬态增长'),
    T('The plot shows ‖Aᵗ‖₂ on a log scale; the inset shows the eigenvalues and the unit circle (§5.3.3–§5.3.5).', '圖中以對數尺度畫出 ‖Aᵗ‖₂；右上小圖是特徵值與單位圓（§5.3.3–§5.3.5）。', '图中以对数尺度画出 ‖Aᵗ‖₂；右上小图是特征值与单位圆（§5.3.3–§5.3.5）。'));
  const e = {};
  for (const [k, v, lo, hi] of [['a11', 0.9, -1.2, 1.2], ['a12', 5, -6, 6], ['a21', 0, -6, 6], ['a22', 0.9, -1.2, 1.2]]) { e[k] = slider(k, { min: lo, max: hi, step: 0.01, value: v, show: (x) => x.toFixed(2) }, draw); controls.append(e[k].el); }
  const presets = [[T('non-normal', '非正規', '非正规'), [0.9, 5, 0, 0.9]], [T('rotation', '旋轉', '旋转'), [0.9 * Math.cos(Math.PI / 6), -0.9 * Math.sin(Math.PI / 6), 0.9 * Math.sin(Math.PI / 6), 0.9 * Math.cos(Math.PI / 6)]], [T('two modes', '兩個模態', '两个模态'), [0.5, 0, 0, 0.95]], [T('unstable', '不穩定', '不稳定'), [1.05, 0.2, 0, 0.7]]];
  controls.append(h('div', { class: 'dlis-ctl' }, presets.map(([label, vals]) => button(label, () => { ['a11', 'a12', 'a21', 'a22'].forEach((k, i) => e[k].set(vals[i])); draw(); }))));
  function norm2(m) { const [a, b, c, d] = m; const s1 = a * a + b * b + c * c + d * d, det = a * d - b * c; return Math.sqrt((s1 + Math.sqrt(Math.max(s1 * s1 - 4 * det * det, 0))) / 2); }
  function draw() {
    const A = [e.a11.get(), e.a12.get(), e.a21.get(), e.a22.get()];
    const tr = A[0] + A[3], det = A[0] * A[3] - A[1] * A[2], disc = tr * tr - 4 * det;
    const eig = disc >= 0 ? [[(tr + Math.sqrt(disc)) / 2, 0], [(tr - Math.sqrt(disc)) / 2, 0]] : [[tr / 2, Math.sqrt(-disc) / 2], [tr / 2, -Math.sqrt(-disc) / 2]];
    const rho = Math.max(...eig.map(([re, im]) => Math.hypot(re, im)));
    let P = [1, 0, 0, 1]; const pts = [];
    for (let t = 0; t <= 60; t++) { pts.push([t, norm2(P)]); P = [A[0] * P[0] + A[1] * P[2], A[0] * P[1] + A[1] * P[3], A[2] * P[0] + A[3] * P[2], A[2] * P[1] + A[3] * P[3]]; }
    const peak = pts.reduce((m, p) => (p[1] > m[1] ? p : m), [0, 0]);
    linePlot(svg, { x: [0, 60], y: [1e-4, 1e4], logY: true, series: [{ pts, cls: 's2' }, { pts: pts.map(([t]) => [t, rho ** t]), cls: 's1' }], hlines: [{ y: 1 }],
      xlabel: 't', ylabel: '‖Aᵗ‖₂', ariaLabel: T('norm of matrix powers', '矩陣冪次的範數', '矩阵幂次的范数') });
    const cx = 560, cy = 70, R = 40;
    svgEl('circle', { cx, cy, r: R, class: 'dlis-unit' }, svg);
    svgEl('line', { x1: cx - R - 8, x2: cx + R + 8, y1: cy, y2: cy, class: 'dlis-axis' }, svg);
    svgEl('line', { x1: cx, x2: cx, y1: cy - R - 8, y2: cy + R + 8, class: 'dlis-axis' }, svg);
    for (const [re, im] of eig) svgEl('circle', { cx: cx + Math.max(-1.6, Math.min(1.6, re)) * R, cy: cy - Math.max(-1.6, Math.min(1.6, im)) * R, r: 4.5, class: 'dlis-dot s2' }, svg);
    const isNormal = Math.abs(A[1] - A[2]) < 1e-9 || (Math.abs(A[0] - A[3]) < 1e-9 && Math.abs(A[1] + A[2]) < 1e-9);
    readout.replaceChildren(legend([['s2', '‖Aᵗ‖₂'], ['s1', 'ρ(A)ᵗ']]), ...kv([
      [T('eigenvalues', '特徵值', '特征值'), eig.map(([re, im]) => (Math.abs(im) < 1e-9 ? fmt(re) : `${fmt(re)} ${im > 0 ? '+' : '−'} ${fmt(Math.abs(im))}i`)).join(', ')],
      ['ρ(A) =', fmt(rho)], [T('peak ‖Aᵗ‖₂', '‖Aᵗ‖₂ 峰值', '‖Aᵗ‖₂ 峰值'), `${fmt(peak[1], 2)} (t = ${peak[0]})`],
      [T('verdict:', '判讀：', '判读：'), rho >= 1 ? T('not asymptotically stable', '非漸近穩定', '非渐近稳定') : peak[1] > 1.05 ? T('stable, but grows before it decays (non-normal)', '穩定，但先增長再衰減（非正規）', '稳定，但先增长再衰减（非正规）') : T('stable, decays', '穩定並衰減', '稳定并衰减')],
      [T('normal matrix?', '正規矩陣？', '正规矩阵？'), isNormal ? T('yes', '是', '是') : T('no', '否', '否')],
    ]));
  }
  draw();
};

widgets['linear-attention-capacity'] = (root) => {
  const { svg, controls, readout } = frame(root, T('How many pairs fit in a linear-attention memory?', '線性注意力記憶能放幾組鍵值對？', '线性注意力记忆能放几组键值对？'),
    T('Hebbian storage S = Σ vⱼkⱼᵀ with random unit keys; the curve is the first-order estimate √((n−1)/d_k) of §5.5.3.', '以隨機單位鍵做 Hebbian 儲存 S = Σ vⱼkⱼᵀ；曲線是 §5.5.3 的一階估計 √((n−1)/d_k)。', '以随机单位键做 Hebbian 存储 S = Σ vⱼkⱼᵀ；曲线是 §5.5.3 的一阶估计 √((n−1)/d_k)。'));
  const dk = slider('d_k', { min: 4, max: 64, step: 4, value: 16 }, draw);
  let seed = 11;
  controls.append(dk.el, h('div', { class: 'dlis-ctl' }, button(T('resample', '重新抽樣', '重新采样'), () => { seed++; draw(); })));
  function trial(n, d, r) {
    const K = Array.from({ length: n }, () => { const k = Array.from({ length: d }, r.normal); const z = Math.hypot(...k); return k.map((v) => v / z); });
    const V = Array.from({ length: n }, () => Array.from({ length: 8 }, r.normal));
    let err = 0;
    for (let i = 0; i < n; i++) {
      const out = Array(8).fill(0);
      for (let j = 0; j < n; j++) { const c = K[j].reduce((acc, v, m) => acc + v * K[i][m], 0); for (let m = 0; m < 8; m++) out[m] += c * V[j][m]; }
      err += Math.hypot(...out.map((v, m) => v - V[i][m])) / Math.hypot(...V[i]);
    }
    return err / n;
  }
  function draw() {
    const d = dk.get(), r = rng(seed), ns = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128].filter((n) => n <= 8 * d);
    const meas = ns.map((n) => [n / d, Array.from({ length: 6 }, () => trial(n, d, r)).reduce((p, q) => p + q, 0) / 6]);
    const theory = Array.from({ length: 81 }, (_, i) => { const n = 1 + (i / 80) * (Math.min(128, 8 * d) - 1); return [n / d, Math.sqrt((n - 1) / d)]; });
    linePlot(svg, { x: [0, Math.min(128, 8 * d) / d], y: [0, 3.5], series: [{ pts: theory, cls: 's2' }, { pts: meas, cls: 's1 dlis-nolines', dots: true }], vlines: [{ x: 1, label: 'n = d_k' }],
      xlabel: T('stored pairs per key dimension n/d_k', '每個鍵維度儲存的對數 n/d_k', '每个键维度存储的对数 n/d_k'), ylabel: T('relative error', '相對誤差', '相对误差'), ariaLabel: T('retrieval error versus load', '檢索誤差與負載', '检索误差与负载') });
    const at1 = meas.find(([x]) => Math.abs(x - 1) < 1e-9);
    readout.replaceChildren(legend([['s1', T('measured', '實測', '实测')], ['s2', T('first-order estimate', '一階估計', '一阶估计')]]), ...kv([[T('error at n = d_k:', 'n = d_k 時的誤差：', 'n = d_k 时的误差：'), at1 ? fmt(at1[1]) : '—']]));
  }
  draw();
};

// ------------------------------------------------------------------ mount
function mount() {
  for (const el of document.querySelectorAll('.dlis-widget[data-widget]:not(.dlis-ready)')) {
    const make = widgets[el.dataset.widget];
    if (!make) { el.textContent = `Unknown widget: ${el.dataset.widget}`; continue; }
    try { make(el); } catch (err) { el.textContent = T('This interactive figure failed to load.', '互動圖載入失敗。', '交互图加载失败。'); console.error(err); }
  }
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount); else mount();
