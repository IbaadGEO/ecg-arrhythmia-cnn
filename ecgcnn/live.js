// Live canvas diagrams for explorer.py (st.components.v2). Every number drawn comes from the Python side.
const C = {
  bg: "#161c22", grid: "rgba(214,222,230,0.06)", gridMajor: "rgba(214,222,230,0.12)", text: "#e8e4da",
  muted: "#8b949a", teal: "#56c7b8", gold: "#f2c94c", sky: "#56b4e9",
  cls: ["#e8e4da", "#56b4e9", "#e69f00", "#cc79a7"], status: ["#8b949a", "#e69f00", "#d55e00", "#56b4e9"],
};
const MONO = '"JetBrains Mono", monospace', BODY = "Manrope, sans-serif", HEAD = '"Barlow Semi Condensed", sans-serif';

const clamp = (u, a = 0, b = 1) => Math.max(a, Math.min(b, u));
const ease = (u) => { u = clamp(u); return u < 0.5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2; };
const phase = (t, a, b) => clamp((t - a) / (b - a));
const lerp = (a, b, u) => a + (b - a) * u;
const NARROW = 560;  // below this width (phones) side panels move under the diagram, which grows by scene.extra(w)

function txt(ctx, s, x, y, { font = `12px ${BODY}`, color = C.text, align = "left", base = "middle" } = {}) {
  ctx.font = font; ctx.fillStyle = color; ctx.textAlign = align; ctx.textBaseline = base; ctx.fillText(s, x, y);
}
function fit(ctx, s, max, size, family) {  // largest font up to `size` px at which `s` fits in `max` px
  ctx.font = `${size}px ${family}`; const w = ctx.measureText(s).width;
  return `${w > max ? Math.max(8, Math.floor(size * max / w)) : size}px ${family}`;
}
function line(ctx, pts, color, lw = 1.2) {
  if (pts.length < 2) return;
  ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.lineJoin = "round"; ctx.stroke();
}
function frame(ctx, x, y, w, h) {
  ctx.strokeStyle = C.gridMajor; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
}
function paper(ctx, x, y, w, h, step) {  // ECG-paper grid, every 5th line stronger
  for (let k = 0, gx = x; gx <= x + w; gx += step, k++) {
    ctx.fillStyle = k % 5 ? C.grid : C.gridMajor; ctx.fillRect(gx, y, 1, h);
  }
  for (let k = 0, gy = y + h; gy >= y; gy -= step, k++) {
    ctx.fillStyle = k % 5 ? C.grid : C.gridMajor; ctx.fillRect(x, gy, w, 1);
  }
}
function extent(a) { let lo = Infinity, hi = -Infinity; for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; } return [lo, hi]; }
const CIVIDIS = [[0, 32, 76], [65, 77, 107], [124, 123, 120], [188, 175, 111], [255, 233, 69]];
function cividis(u) {
  u = clamp(u) * 4; const i = Math.min(3, Math.floor(u)), f = u - i, a = CIVIDIS[i], b = CIVIDIS[i + 1];
  return [lerp(a[0], b[0], f), lerp(a[1], b[1], f), lerp(a[2], b[2], f)];
}
function heatImage(vals, rows, cols) {  // vals 0..1, row-major
  const c = document.createElement("canvas"); c.width = cols; c.height = rows;
  const g = c.getContext("2d"), img = g.createImageData(cols, rows);
  for (let i = 0; i < rows * cols; i++) {
    const [r, gg, b] = cividis(vals[i]); img.data.set([r, gg, b, 255], i * 4);
  }
  g.putImageData(img, 0, 0); return c;
}
function badge(ctx, s, x, y, fg, edge) {
  ctx.font = `600 13px ${HEAD}`; const w = ctx.measureText(s).width + 10;
  ctx.fillStyle = C.bg; ctx.fillRect(x - w / 2, y - 9, w, 18);
  if (edge) { ctx.strokeStyle = edge; ctx.lineWidth = 2; ctx.strokeRect(x - w / 2, y - 9, w, 18); }
  txt(ctx, s, x, y + 1, { font: `600 13px ${HEAD}`, color: fg, align: "center" });
}
function probBars(ctx, x, y, w, rowH, names, probs, grow, highlight) {
  names.forEach((n, k) => {
    const yy = y + k * rowH;
    txt(ctx, n, x, yy + rowH / 2, { font: `11px ${BODY}`, color: k === highlight ? C.text : C.muted });
    const bx = x + 104, bw = w - 104 - 46;
    ctx.fillStyle = "rgba(214,222,230,0.08)"; ctx.fillRect(bx, yy + 4, bw, rowH - 8);
    ctx.fillStyle = k === highlight ? C.teal : "rgba(86,199,184,0.45)"; ctx.fillRect(bx, yy + 4, bw * probs[k] * grow, rowH - 8);
    txt(ctx, `${(probs[k] * grow * 100).toFixed(0)}%`, x + w, yy + rowH / 2, { font: `11px ${MONO}`, align: "right" });
  });
}
function gauge(ctx, x, y, w, value, thr, label) {
  txt(ctx, label, x, y, { font: `11px ${BODY}`, color: C.muted });
  ctx.fillStyle = "rgba(214,222,230,0.08)"; ctx.fillRect(x, y + 10, w, 8);
  ctx.fillStyle = value >= thr ? C.status[2] : C.teal; ctx.fillRect(x, y + 10, w * value, 8);
  ctx.fillStyle = C.gold; ctx.fillRect(x + w * thr - 1, y + 6, 2, 16);
  txt(ctx, `${(value * 100).toFixed(0)}%`, x + w, y, { font: `11px ${MONO}`, align: "right" });
  txt(ctx, `alarm at ${(thr * 100).toFixed(0)}%`, x + w * thr, y + 30, { font: `10px ${MONO}`, color: C.gold, align: "center" });
}

// ---- scenes: each returns {duration, draw(ctx, w, h, t)} ----------------------------------------------------------
const SCENES = {
  // bedside-monitor playback: the trace sweeps in real time, each beat gets the model's call once its input exists
  monitor(d) {
    const VIEW = 6, n = d.sig.length;
    return {
      duration: d.dur,
      extra: (w) => (w < NARROW ? 275 : 0),
      draw(ctx, w, h, t) {
        const nar = w < NARROW, px = 52, pw = nar ? w - px - 4 : Math.max(260, w - 250 - px), ph = h - 40, py = 6;
        const lo = Math.max(0, t - VIEW * 0.85), hi = lo + VIEW;
        const X = (s) => px + ((s - lo) / VIEW) * pw, Y = (v) => py + 26 + (1 - (v - d.lo) / (d.hi - d.lo)) * (ph - 60);
        paper(ctx, px, py, pw, ph, pw / (VIEW / 0.04));  // 0.04 s small squares, 0.2 s large
        frame(ctx, px, py, pw, ph);
        const a = Math.max(0, Math.floor(lo * d.fs)), b = Math.min(n, Math.floor(Math.min(t, hi) * d.fs));
        const pts = []; for (let i = a; i < b; i++) pts.push([X(i / d.fs), Y(d.sig[i])]);
        line(ctx, pts, C.teal, 1.6);
        if (pts.length) { const [cx, cy] = pts[pts.length - 1]; ctx.fillStyle = C.gold; ctx.beginPath(); ctx.arc(cx, cy, 3.5, 0, 7); ctx.fill(); }
        txt(ctx, "MODEL", px - 8, py + 14, { font: `600 10px ${HEAD}`, color: C.muted, align: "right" });
        txt(ctx, "EXPERTS", px - 8, py + ph - 12, { font: `600 10px ${HEAD}`, color: C.muted, align: "right" });
        let latest = -1; const tally = [0, 0, 0, 0], seen = { all: 0, abn: 0, alarmed: 0 };
        d.beats.forEach((bt, k) => {
          if (bt.ready > t) return;
          latest = k; tally[bt.status]++; seen.all++; if (bt.ref) seen.abn++; if (bt.alarm) seen.alarmed++;
          if (bt.t < lo || bt.t > hi) return;
          const x = X(bt.t);
          badge(ctx, d.classes[bt.pred], x, py + 14, C.cls[bt.pred], bt.alarm ? (bt.status ? C.status[bt.status] : C.text) : null);
          txt(ctx, d.classes[bt.ref], x, py + ph - 12, { font: `12px ${HEAD}`, color: C.text, align: "center" });
          if (bt.status >= 2) txt(ctx, bt.status === 2 ? "missed" : "false alarm", x, py + ph - 26, { font: `10px ${BODY}`, color: C.status[bt.status], align: "center" });
        });
        // readout panel
        const rx = nar ? 0 : px + pw + 18, rw = w - rx, ry = nar ? h - 6 : py;
        txt(ctx, "LATEST BEAT", rx, ry + 10, { font: `600 12px ${HEAD}`, color: C.text });
        if (latest >= 0) {
          const bt = d.beats[latest], grow = ease((t - bt.ready) / 0.35);
          probBars(ctx, rx, ry + 22, rw, 20, d.names, bt.probs, grow, bt.pred);
          gauge(ctx, rx, ry + 118, rw - 4, bt.pabn * grow, d.thr, "chance not ordinary");
        } else txt(ctx, "waiting for the first full window…", rx, ry + 40, { font: `11px ${BODY}`, color: C.muted });
        txt(ctx, "SO FAR", rx, ry + 178, { font: `600 12px ${HEAD}`, color: C.text });
        const rows = [["beats read", `${seen.all}`, C.text], ["correct", `${tally[0]}`, C.text],
          ["alarm, wrong type", `${tally[1]}`, C.status[1]], ["missed", `${tally[2]}`, C.status[2]], ["false alarms", `${tally[3]}`, C.status[3]]];
        rows.forEach(([l, v, c], k) => {
          txt(ctx, l, rx, ry + 198 + k * 17, { font: `11px ${BODY}`, color: c });
          txt(ctx, v, w - 4, ry + 198 + k * 17, { font: `11px ${MONO}`, align: "right" });
        });
        txt(ctx, `${t.toFixed(1)} s`, px + pw - 6, py + ph + 16, { font: `11px ${MONO}`, color: C.muted, align: "right" });
      },
    };
  },

  // cutting the window out of the recording, then rescaling it
  input(d) {
    return {
      duration: 6,
      draw(ctx, w, h, t) {
        const nar = w < NARROW, pad = 10, pw = w - 2 * pad, ph = h - 50, top = 8;
        const [rlo, rhi] = extent(d.raw), [zlo, zhi] = extent(d.x);
        const nRaw = d.raw.length, ws = d.win[0], we = d.win[1];
        const cut = ease(phase(t, 1.4, 3.0)), resc = ease(phase(t, 3.4, 4.8)), fade = ease(phase(t, 0.2, 1.0));
        paper(ctx, pad, top, pw, ph, pw / 70); frame(ctx, pad, top, pw, ph);
        // context trace fades out as the window is cut out
        const ctxPts = []; for (let i = 0; i < nRaw; i += 1) ctxPts.push([pad + (i / (nRaw - 1)) * pw, top + 8 + (1 - (d.raw[i] - rlo) / (rhi - rlo)) * (ph - 16)]);
        ctx.globalAlpha = 1 - 0.85 * cut; line(ctx, ctxPts, C.muted, 1); ctx.globalAlpha = 1;
        const bx0 = pad + (ws / (nRaw - 1)) * pw, bx1 = pad + (we / (nRaw - 1)) * pw;
        ctx.fillStyle = `rgba(242,201,76,${0.12 * fade * (1 - cut)})`; ctx.fillRect(bx0, top, bx1 - bx0, ph);
        // the window's own samples move from their place in the recording to fill the panel, then rescale
        const pts = [];
        for (let k = 0; k < d.x.length; k++) {
          const i = ws + (k / (d.x.length - 1)) * (we - ws);
          const x0 = pad + (i / (nRaw - 1)) * pw, x1 = pad + (k / (d.x.length - 1)) * pw;
          const rawV = d.raw[Math.round(i)];
          const yRaw = top + 8 + (1 - (rawV - rlo) / (rhi - rlo)) * (ph - 16);
          const yZ = top + 8 + (1 - (d.x[k] - zlo) / (zhi - zlo)) * (ph - 16);
          pts.push([lerp(x0, x1, cut), lerp(yRaw, yZ, resc)]);
        }
        line(ctx, pts, C.teal, 1.6);
        const msg = t < 1.4 ? "1 · a 1.4 s slice centred on this beat's main spike"
          : t < 3.2 ? `2 · cut out: ${d.x.length} measurements` : "3 · rescaled: average 0, typical spread 1, so only the shape is left";
        txt(ctx, msg, pad, h - 26, { font: fit(ctx, msg, w - 2 * pad, 12, BODY), color: C.text });
        txt(ctx, resc > 0.5 ? "units: standard deviations" : "units: millivolts", w - pad, nar ? h - 6 : h - 26, { font: `11px ${MONO}`, color: C.muted, align: "right" });
      },
    };
  },

  // one detector sliding along the beat; its output is drawn as it goes
  conv(d) {
    const n = d.x.length;
    return {
      duration: 7,
      extra: (w) => (w < NARROW ? 175 : 0),
      draw(ctx, w, h, t) {
        const nar = w < NARROW, iw = nar ? Math.min(w, 320) : 210, pw = nar ? w : w - iw - 16, top = 8, ph = h - 40, k = Math.min(n - 1, Math.floor(phase(t, 0.3, 6.6) * n));
        const [xlo, xhi] = extent(d.x), amax = Math.max(1e-6, extent(d.act)[1]);
        const X = (i) => (i / (n - 1)) * pw, Y = (v) => top + 6 + (1 - (v - xlo) / (xhi - xlo)) * (ph * 0.62);
        const A = (v) => top + ph - 4 - (v / amax) * (ph * 0.3);
        paper(ctx, 0, top, pw, ph, pw / 70); frame(ctx, 0, top, pw, ph);
        line(ctx, Array.from(d.x, (v, i) => [X(i), Y(v)]), C.teal, 1.3);
        // output so far
        ctx.beginPath(); ctx.moveTo(X(0), A(0));
        for (let i = 0; i <= k; i++) ctx.lineTo(X(i), A(d.act[i]));
        ctx.lineTo(X(k), A(0)); ctx.closePath(); ctx.fillStyle = "rgba(242,201,76,0.45)"; ctx.fill();
        txt(ctx, "detector output", 6, top + ph - 10 - ph * 0.3, { font: `10px ${HEAD}`, color: C.gold });
        // the 7-sample window under the detector
        const a = Math.max(0, k - 3), b = Math.min(n - 1, k + 3);
        ctx.strokeStyle = C.gold; ctx.lineWidth = 1.5; ctx.strokeRect(X(a) - 2, top + 2, X(b) - X(a) + 4, ph * 0.64);
        ctx.fillStyle = "rgba(242,201,76,0.6)"; ctx.fillRect(X(k), top + ph * 0.66, 1, ph * 0.34 - 4);
        // inset: samples x weights
        const ix = nar ? 0 : pw + 16, it = nar ? h : top, cw = (iw - 10) / 7;
        txt(ctx, "7 SAMPLES × 7 WEIGHTS", ix, it + 8, { font: `600 11px ${HEAD}` });
        const prods = []; let sum = 0;
        for (let j = 0; j < 7; j++) { const i = k - 3 + j, v = i >= 0 && i < n ? d.x[i] : 0; prods.push([v, d.w[j]]); sum += v * d.w[j]; }
        const wmax = Math.max(...d.w.map(Math.abs), 1e-6), vmax = Math.max(...prods.map((p) => Math.abs(p[0])), 1e-6);
        prods.forEach(([v, wt], j) => {
          const cx = ix + j * cw + cw / 2;
          ctx.fillStyle = C.teal; const hv = (v / vmax) * 22; ctx.fillRect(cx - 5, it + 50 - Math.max(hv, 0), 10, Math.abs(hv));
          ctx.fillStyle = wt >= 0 ? C.text : C.gold; const hw = (wt / wmax) * 22; ctx.fillRect(cx - 5, it + 104 - Math.max(hw, 0), 10, Math.abs(hw));
        });
        ctx.fillStyle = C.gridMajor; ctx.fillRect(ix, it + 50, iw - 10, 1); ctx.fillRect(ix, it + 104, iw - 10, 1);
        txt(ctx, "signal", ix, it + 22, { font: `10px ${BODY}`, color: C.teal });
        txt(ctx, "weights", ix, it + 76, { font: `10px ${BODY}`, color: C.muted });
        txt(ctx, "multiply and add", ix, it + 140, { font: `11px ${BODY}`, color: C.muted });
        txt(ctx, sum.toFixed(2), ix + iw - 10, it + 140, { font: `13px ${MONO}`, align: "right" });
        txt(ctx, d.bn ? "after learned scaling + ReLU" : "ReLU (negatives → 0)", ix, it + 162, { font: `11px ${BODY}`, color: C.muted });
        txt(ctx, d.act[k].toFixed(2), ix + iw - 10, it + 162, { font: `13px ${MONO}`, color: C.gold, align: "right" });
        txt(ctx, `${((k - n / 2) / d.fs).toFixed(2)} s from the main spike`, 0, h - 16, { font: `11px ${MONO}`, color: C.muted });
      },
    };
  },

  // the beat flowing through the four layers, then averaged into the summary numbers
  layers(d) {
    let imgs = null;
    return {
      duration: 8,
      draw(ctx, w, h, t) {
        imgs = imgs || d.blocks.map((b) => heatImage(b.v, b.rows, b.cols));
        const lw = 168, pw = w - lw - 8, bands = d.blocks.length + 2, gap = 8, bh = (h - 10 - gap * (bands - 1)) / bands;
        let y = 4;
        const [xlo, xhi] = extent(d.x), rev0 = ease(phase(t, 0.2, 1.2));
        txt(ctx, "INPUT", 0, y + bh / 2 - 7, { font: `600 12px ${HEAD}` });
        txt(ctx, `${d.x.length} numbers`, 0, y + bh / 2 + 8, { font: `10px ${MONO}`, color: C.muted });
        const m = Math.floor(rev0 * d.x.length);
        line(ctx, Array.from(d.x.slice(0, m), (v, i) => [lw + (i / (d.x.length - 1)) * pw, y + 2 + (1 - (v - xlo) / (xhi - xlo)) * (bh - 4)]), C.teal, 1.2);
        frame(ctx, lw, y, pw, bh); y += bh + gap;
        d.blocks.forEach((b, j) => {
          const u = ease(phase(t, 1.2 + j * 1.1, 2.2 + j * 1.1));
          txt(ctx, `LAYER ${j + 1}`, 0, y + bh / 2 - 7, { font: `600 12px ${HEAD}` });
          txt(ctx, `${b.total} × ${b.cols}${b.rows < b.total ? ` (${b.rows} shown)` : ""}`, 0, y + bh / 2 + 8, { font: `10px ${MONO}`, color: C.muted });
          if (u > 0) {
            ctx.save(); ctx.beginPath(); ctx.rect(lw, y, pw * u, bh); ctx.clip();
            ctx.imageSmoothingEnabled = false; ctx.drawImage(imgs[j], lw, y, pw, bh); ctx.restore();
            if (u < 1) { ctx.fillStyle = C.gold; ctx.fillRect(lw + pw * u - 1, y, 2, bh); }
          }
          frame(ctx, lw, y, pw, bh); y += bh + gap;
        });
        // averaging: the last layer's rows collapse into one number per detector
        const u = ease(phase(t, 6.0, 7.2));
        txt(ctx, "SUMMARY", 0, y + bh / 2 - 7, { font: `600 12px ${HEAD}` });
        txt(ctx, `${d.feats.length} averages`, 0, y + bh / 2 + 8, { font: `10px ${MONO}`, color: C.muted });
        const fmax = Math.max(...d.feats, 1e-6), cw = pw / d.feats.length;
        d.feats.forEach((f, i) => {
          const [r, g, bb] = cividis(f / fmax); ctx.fillStyle = `rgb(${r},${g},${bb})`;
          const hh = (bh - 4) * (f / fmax) * u; ctx.fillRect(lw + i * cw, y + bh - 2 - hh, Math.max(1, cw - 0.5), hh);
        });
        frame(ctx, lw, y, pw, bh);
      },
    };
  },

  // the four class scores built from their parts, then turned into percentages and the alarm
  decision(d) {
    const parts = [["shape", d.shape, C.teal], ["timing", d.timing, C.sky], ["starting point", d.bias, C.muted]].filter((p) => p[1]);
    return {
      duration: 7,
      extra: (w) => (w < NARROW ? 220 : 0),
      draw(ctx, w, h, t) {
        const nar = w < NARROW, half = nar ? w : Math.floor(w * 0.55), lw = 112, top = 26, rowH = Math.min(34, (h - top - 70) / 4);
        let span = 1e-6;
        d.names.forEach((_, k) => { let c = 0; for (const [, v] of parts) { c += v[k]; span = Math.max(span, Math.abs(c)); } });
        const zx = lw + (half - lw - 60) / 2, sc = (half - lw - 70) / 2 / span;
        txt(ctx, "SCORES", 0, 10, { font: `600 12px ${HEAD}` });
        ctx.fillStyle = C.gridMajor; ctx.fillRect(zx, top - 4, 1, rowH * 4 + 4);
        d.names.forEach((nm, k) => {
          const y = top + k * rowH;
          txt(ctx, nm, 0, y + rowH / 2, { font: `11px ${BODY}`, color: C.muted });
          let c = 0;
          parts.forEach(([, v, col], j) => {
            const u = ease(phase(t, 0.3 + j * 1.0, 1.2 + j * 1.0)), a = c, b = c + v[k] * u;
            ctx.fillStyle = col; ctx.globalAlpha = 0.85;
            ctx.fillRect(zx + Math.min(a, b) * sc, y + 6 + j * 3, Math.abs(b - a) * sc, rowH - 12 - j * 6);
            ctx.globalAlpha = 1; c += v[k] * u;
          });
          if (t > 0.3 + parts.length) {
            ctx.fillStyle = C.text; ctx.fillRect(zx + c * sc - 1, y + 3, 2, rowH - 6);
            txt(ctx, (c >= 0 ? "+" : "") + c.toFixed(2), half - 8, y + rowH / 2, { font: `11px ${MONO}`, align: "right" });
          }
        });
        let ly = top + rowH * 4 + 14, lx = 0;
        parts.forEach(([nm, , col]) => { ctx.fillStyle = col; ctx.fillRect(lx, ly - 4, 10, 8); txt(ctx, nm, lx + 14, ly, { font: `10px ${BODY}`, color: C.muted }); lx += ctx.measureText(nm).width + 34; });
        // softmax and alarm
        const rx = nar ? 0 : half + 24, rw = w - rx, oy = nar ? h : 0, u = ease(phase(t, 3.8, 5.0));
        txt(ctx, "PERCENTAGES (SOFTMAX)", rx, oy + 10, { font: `600 12px ${HEAD}`, color: u > 0 ? C.text : C.muted });
        probBars(ctx, rx, oy + top, rw, rowH, d.names, d.probs, u, d.probs.indexOf(Math.max(...d.probs)));
        const g = ease(phase(t, 5.2, 6.2));
        if (g > 0) gauge(ctx, rx, oy + top + rowH * 4 + 16, rw - 4, (1 - d.probs[0]) * g, d.thr, "100% − N group = chance not ordinary");
      },
    };
  },

  // an automatic sweep: a flattened stretch moving across the beat, or the gap before the beat stretching
  sweep(d) {
    const n = d.xs.length;
    return {
      duration: 7,
      extra: (w) => (w < NARROW ? 18 : 0),
      draw(ctx, w, h, t) {
        const nar = w < NARROW, k = Math.min(n - 1, Math.floor(phase(t, 0.2, 6.6) * n)), top = 8, th = (h - 60) * 0.45, by = top + th + 16, bh = h - by - 34;
        frame(ctx, 0, top, w, th);
        if (d.mode === "occlude") {
          const m = d.beat.length, [lo, hi] = extent(d.beat), X = (i) => (i / (m - 1)) * w, Y = (v) => top + 4 + (1 - (v - lo) / (hi - lo)) * (th - 8);
          const c = d.centres[k];
          line(ctx, Array.from(d.beat, (v, i) => [X(i), Y(Math.abs(i - c) <= d.hw ? 0 : v)]), C.teal, 1.2);
          ctx.fillStyle = "rgba(242,201,76,0.18)"; ctx.fillRect(X(c - d.hw), top, X(2 * d.hw) - X(0), th);
          txt(ctx, "flattened stretch", clamp(X(c), 50, w - 50), top + th - 8, { font: `10px ${BODY}`, color: C.gold, align: "center" });
        } else {
          const gap = d.xs[k], unit = (w - 80) / (2 * Math.max(...d.xs)), x0 = 40, x1 = x0 + gap * unit, x2 = x1 + d.post * unit;
          const spike = (x, col) => line(ctx, [[x - 14, top + th * 0.7], [x - 4, top + th * 0.7], [x, top + 10], [x + 4, top + th * 0.85], [x + 10, top + th * 0.7], [x + 30, top + th * 0.7]], col, 1.5);
          spike(x0, C.muted); spike(x1, C.teal); spike(x2, C.muted);
          ctx.fillStyle = C.gold; ctx.fillRect(x0, top + th - 14, x1 - x0, 2);
          txt(ctx, `gap before: ${gap.toFixed(2)}× average`, clamp((x0 + x1) / 2, 90, w - 90), top + th + 8, { font: `11px ${MONO}`, color: C.gold, align: "center" });
        }
        // the curve so far
        frame(ctx, 0, by, w, bh);
        const X = (i) => (i / (n - 1)) * w, Y = (v) => by + bh - v * bh;
        ctx.setLineDash([4, 4]); line(ctx, [[0, Y(d.thr)], [w, Y(d.thr)]], C.gold, 1); ctx.setLineDash([1, 3]);
        line(ctx, [[0, Y(d.base)], [w, Y(d.base)]], C.muted, 1); ctx.setLineDash([]);
        txt(ctx, "alarm", w - 4, Y(d.thr) - 8, { font: `10px ${MONO}`, color: C.gold, align: "right" });
        txt(ctx, "unchanged beat", 4, Y(d.base) - 8, { font: `10px ${MONO}`, color: C.muted });
        line(ctx, Array.from(d.pabn.slice(0, k + 1), (v, i) => [X(i), Y(v)]), C.teal, 1.8);
        ctx.fillStyle = d.pabn[k] >= d.thr ? C.status[2] : C.teal; ctx.beginPath(); ctx.arc(X(k), Y(d.pabn[k]), 4, 0, 7); ctx.fill();
        txt(ctx, `chance not ordinary: ${(d.pabn[k] * 100).toFixed(0)}%`, 0, h - 14, { font: `12px ${MONO}` });
        txt(ctx, d.xlabel, nar ? 0 : w, nar ? h + 6 : h - 14, { font: fit(ctx, d.xlabel, w, 11, BODY), color: C.muted, align: nar ? "left" : "right" });
      },
    };
  },
};

export default function (component) {
  const { data, parentElement } = component;
  let root = parentElement.querySelector(".live");
  if (root && root._stop) root._stop();
  if (!root) { root = document.createElement("div"); root.className = "live"; parentElement.appendChild(root); }
  root.innerHTML = '<div class="bar"><button class="play" type="button"></button><button class="again" type="button">Replay</button><span class="cap"></span></div><canvas role="img"></canvas>';
  const canvas = root.querySelector("canvas"), btn = root.querySelector(".play"), again = root.querySelector(".again");
  root.querySelector(".cap").textContent = data.caption || "";
  canvas.setAttribute("aria-label", data.alt || data.caption || "animated diagram");
  const H = data.height || 260, scene = SCENES[data.scene](data), hold = 2.5, ctx = canvas.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let t = reduce ? scene.duration : 0, last = null, raf = 0, w = 0, ch = H, playing = false;
  const draw = () => {
    ctx.clearRect(0, 0, w, ch); ctx.fillStyle = C.bg; ctx.fillRect(0, 0, w, ch);
    scene.draw(ctx, w, H, Math.min(t, scene.duration));
  };
  const size = () => {
    const dpr = window.devicePixelRatio || 1; w = Math.max(200, root.clientWidth - 16);
    ch = H + (scene.extra ? scene.extra(w) : 0);
    canvas.width = w * dpr; canvas.height = ch * dpr; canvas.style.width = `${w}px`; canvas.style.height = `${ch}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
  };
  const tick = (now) => {
    if (last !== null) { t += (now - last) / 1000; if (t > scene.duration + hold) t = 0; }
    last = now; draw(); raf = requestAnimationFrame(tick);
  };
  const setPlay = (p) => {
    playing = p; btn.textContent = p ? "Pause" : "Play"; btn.setAttribute("aria-pressed", String(p));
    cancelAnimationFrame(raf); last = null; if (p) raf = requestAnimationFrame(tick);
  };
  btn.onclick = () => setPlay(!playing);
  again.onclick = () => { t = 0; setPlay(true); };
  const ro = new ResizeObserver(size); ro.observe(root);
  size(); setPlay(!reduce);
  root._stop = () => { cancelAnimationFrame(raf); ro.disconnect(); };
  return root._stop;
}
