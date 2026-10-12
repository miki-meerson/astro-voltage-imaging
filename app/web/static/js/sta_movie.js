(() => {
  const NEURON_ID = JSON.parse(document.getElementById("page-data").textContent).neuronId;
  const root = document.getElementById("smv");
  if (!root) return;
  const $ = id => document.getElementById(id);
  const base = $("smv-base"), ovl = $("smv-ovl"), stage = $("smv-stage"), trace = $("smv-trace");
  const info = $("smv-info"), readout = $("smv-readout"), scrub = $("smv-scrub"), play = $("smv-play");
  const S = 3;  // on-screen pixels per movie pixel
  const COL = {positive: Polarity.color("positive"), negative: Polarity.color("negative"), neuron: "#f7d154", spike: "#ffffff"};  // cluster sign: positive = depolarizing
  let M = null, data = null, frame = 0, timer = null, win = null;

  const rel = f => M.relative_frames[f];
  const fmtRel = r => (r > 0 ? "+" : "") + r;
  function percentileWindow(vals, lo, hi) {
    const a = Float32Array.from(vals).sort(); return [a[Math.floor(lo * (a.length - 1))], a[Math.floor(hi * (a.length - 1))]];
  }
  let baseMap = null;
  function buildBaseMap() {  // per-pixel mean over the baseline frames (-58..-11), the same baseline the clusters were measured against
    const n = M.width * M.height, idx = [];
    M.relative_frames.forEach((r, i) => { if (r >= -58 && r <= -11) idx.push(i); });
    baseMap = new Float32Array(n);
    if (!idx.length) return;
    for (const f of idx) { const off = f * n; for (let i = 0; i < n; i++) baseMap[i] += data[off + i]; }
    for (let i = 0; i < n; i++) baseMap[i] /= idx.length;
  }
  const changeMode = () => $("smv-mode").value === "change";
  const pixel = (off, i) => changeMode() ? data[off + i] - baseMap[i] : data[off + i];
  function windowFor(vals) {  // raw: 1st..99.7th percentile; change: symmetric around 0 so that grey = no change
    if (changeMode()) { const a = Float32Array.from(vals, Math.abs).sort(); const m = Math.max(1, a[Math.floor(0.997 * (a.length - 1))]); return [-m, m]; }
    return percentileWindow(vals, 0.01, 0.997);
  }
  function globalWindow() {
    const n = M.width * M.height, step = Math.max(1, Math.floor(data.length / 60000)), sample = [];
    for (let i = 0; i < data.length; i += step) sample.push(pixel(Math.floor(i / n) * n, i % n));
    return windowFor(sample);
  }
  function drawBase() {
    const n = M.width * M.height, off = frame * n, ctx = base.getContext("2d"), img = ctx.createImageData(M.width, M.height);
    let lo, hi;
    if ($("smv-contrast").value === "frame") {
      const sample = []; const step = Math.max(1, Math.floor(n / 3000));
      for (let i = 0; i < n; i += step) sample.push(pixel(off, i));
      [lo, hi] = windowFor(sample);
    } else [lo, hi] = win;
    const span = Math.max(1, hi - lo);
    for (let i = 0; i < n; i++) { const v = Math.max(0, Math.min(255, ((pixel(off, i) - lo) / span) * 255)); img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = v; img.data[4 * i + 3] = 255; }
    ctx.putImageData(img, 0, 0);
  }
  function outline(ctx, pixels, color, alpha, width) {
    const set = new Set(pixels.map(p => p[0] * 100000 + p[1]));
    ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = width; ctx.beginPath();
    for (const [y, x] of pixels) {
      const X = x * S, Y = y * S;
      if (!set.has((y - 1) * 100000 + x)) { ctx.moveTo(X, Y); ctx.lineTo(X + S, Y); }
      if (!set.has((y + 1) * 100000 + x)) { ctx.moveTo(X, Y + S); ctx.lineTo(X + S, Y + S); }
      if (!set.has(y * 100000 + x - 1)) { ctx.moveTo(X, Y); ctx.lineTo(X, Y + S); }
      if (!set.has(y * 100000 + x + 1)) { ctx.moveTo(X + S, Y); ctx.lineTo(X + S, Y + S); }
    }
    ctx.stroke(); ctx.globalAlpha = 1;
  }
  function arrow(ctx, cy, cx, color, alpha, thick, label) {
    const len = 15, gap = 4, ang = -Math.PI / 4;  // arrow comes from the upper left and points at the cluster centre
    const hx = cx * S + S / 2 + Math.cos(ang) * gap, hy = cy * S + S / 2 + Math.sin(ang) * gap;  // arrow tip (near the cluster)
    const sx = hx + Math.cos(ang) * len, sy = hy + Math.sin(ang) * len;  // arrow tail
    ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = thick * 0.6;
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(hx, hy); ctx.stroke();
    const a1 = Math.atan2(hy - sy, hx - sx), h = 5;
    ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(hx - h * Math.cos(a1 - 0.45), hy - h * Math.sin(a1 - 0.45)); ctx.lineTo(hx - h * Math.cos(a1 + 0.45), hy - h * Math.sin(a1 + 0.45)); ctx.closePath(); ctx.fill();
    if (label) { ctx.font = "11px sans-serif"; ctx.textBaseline = "bottom"; ctx.fillText(label, Math.max(2, Math.min(M.width * S - 70, sx - 6)), Math.max(12, sy - 3)); }
    ctx.globalAlpha = 1;
  }
  function drawOverlay() {
    const ctx = ovl.getContext("2d"); ctx.clearRect(0, 0, ovl.width, ovl.height);
    const mode = $("smv-marks").value;
    if (mode === "none") return;
    const r = rel(frame), full = mode === "full";
    if (full && M.neuron.pixels.length) outline(ctx, M.neuron.pixels, COL.neuron, 0.9, 1.2);
    M.anatomy_clusters.slice(0, 4).forEach((c, i) => {
      const on = r >= c.first_rel && r <= c.last_rel, color = COL[c.sign] || COL.positive;
      if (full) {
        if (on) outline(ctx, c.pixels, color, 1, 2);
        arrow(ctx, c.centroid[0], c.centroid[1], color, on ? 1 : 0.4, on ? 3 : 1.5, `cluster ${i + 1} (${fmtRel(c.first_rel)}..${fmtRel(c.last_rel)})`);
      } else if (on) {  // arrows only, and only while the cluster is active: nothing is drawn over the signal itself
        arrow(ctx, c.centroid[0], c.centroid[1], color, 1, 2, null);
      }
    });
  }
  function drawTrace() {
    const dpr = window.devicePixelRatio || 1, w = trace.clientWidth, h = trace.clientHeight;
    if (!w) return;
    trace.width = Math.round(w * dpr); trace.height = Math.round(h * dpr);
    const ctx = trace.getContext("2d"); ctx.scale(dpr, dpr);
    const T = M.frames, X = f => 6 + (f / Math.max(1, T - 1)) * (w - 12), tr = M.neuron.trace;
    const lim = Math.max(1e-9, ...tr.map(Math.abs)), Y = v => h / 2 - (v / lim) * (h / 2 - 8);
    // clusters: shaded spans over their frames
    M.anatomy_clusters.slice(0, 4).forEach(c => {
      const f0 = M.relative_frames.indexOf(c.first_rel), f1 = M.relative_frames.indexOf(c.last_rel);
      if (f0 >= 0 && f1 >= 0) { ctx.globalAlpha = 0.28; ctx.fillStyle = COL[c.sign] || COL.positive; ctx.fillRect(X(f0), 2, Math.max(2, X(f1) - X(f0)), h - 4); ctx.globalAlpha = 1; }
    });
    ctx.strokeStyle = "#2a3644"; ctx.beginPath(); ctx.moveTo(6, h / 2); ctx.lineTo(w - 6, h / 2); ctx.stroke();
    const f0i = M.relative_frames.indexOf(0);
    if (f0i >= 0) { ctx.setLineDash([3, 3]); ctx.strokeStyle = COL.spike; ctx.beginPath(); ctx.moveTo(X(f0i), 2); ctx.lineTo(X(f0i), h - 2); ctx.stroke(); ctx.setLineDash([]); }
    if (tr.length) { ctx.strokeStyle = "#9bd0ff"; ctx.beginPath(); tr.forEach((v, i) => i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))); ctx.stroke(); }
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(frame), 0); ctx.lineTo(X(frame), h); ctx.stroke(); ctx.lineWidth = 1;
    ctx.fillStyle = "#9aa8b7"; ctx.font = "10px sans-serif"; ctx.textBaseline = "bottom";
    ctx.fillText(`${fmtRel(rel(0))}`, 6, h - 1); ctx.fillText(`${fmtRel(rel(T - 1))}`, w - 24, h - 1);
    ctx.fillStyle = COL.spike; if (f0i >= 0) ctx.fillText("spike", X(f0i) + 3, 30);
    if (M.neuron.trace_label) { ctx.fillStyle = "#9aa8b7"; ctx.textBaseline = "top"; ctx.fillText(`${M.neuron.trace_label} (as recorded); max ${(lim * 100).toFixed(0)}% dF/F`, 8, 2); ctx.textBaseline = "bottom"; }
  }
  // ---- change from baseline: cluster pixels (by sign) vs all other pixels, with random same-size patches as the reference band
  const groupsCv = $("smv-groups");
  let G = null;
  function buildGroups() {
    const n = M.width * M.height, T = M.frames, k = (M.hi - M.lo) / 65535;
    const inSet = { positive: new Uint8Array(n), negative: new Uint8Array(n) }, taken = new Uint8Array(n);
    M.anatomy_clusters.forEach(c => c.pixels.forEach(([y, x]) => { const i = y * M.width + x; if (inSet[c.sign]) inSet[c.sign][i] = 1; taken[i] = 1; }));
    M.neuron.pixels.forEach(([y, x]) => { taken[y * M.width + x] = 1; });  // the neuron's own body is not "other tissue"
    const others = []; for (let i = 0; i < n; i++) if (!taken[i]) others.push(i);
    const series = idx => {  // % change from baseline of the mean over idx: the movie is already relative change (dF/F), biological sign (up = depolarization)
      const out = new Float32Array(T);
      for (let f = 0; f < T; f++) { let s = 0; const off = f * n; for (const i of idx) s += data[off + i] - baseMap[i]; out[f] = (s / idx.length) * k * 100; }
      return {trace: out, pct: true};
    };
    G = {groups: [], other: others.length ? series(others) : null, others, series, n, isOther: Uint8Array.from(taken, v => 1 - v)};
    for (const sign of ["positive", "negative"]) {
      const idx = []; for (let i = 0; i < n; i++) if (inSet[sign][i]) idx.push(i);
      if (idx.length) G.groups.push({sign, idx, s: series(idx), band: null});
    }
  }
  function randomBand(g) {  // spatially compact random patches of "other" pixels with the same pixel count
    const o = G.others, m = Math.min(g.idx.length, Math.floor(o.length / 2)), W = M.width, trials = [];
    if (m < 1) return null;
    let seed = 12345; const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
    for (let t = 0; t < 120; t++) {
      const c = o[Math.floor(rnd() * o.length)], cy = Math.floor(c / W), cx = c % W;
      let r = Math.ceil(Math.sqrt(m)) + 2, cand;  // only look in a window around the centre (sorting all pixels 120 times froze the page)
      for (;;) {
        cand = [];
        for (let y = Math.max(0, cy - r); y <= Math.min(M.height - 1, cy + r); y++) for (let x = Math.max(0, cx - r); x <= Math.min(W - 1, cx + r); x++) if (G.isOther[y * W + x]) cand.push([(y - cy) ** 2 + (x - cx) ** 2, y * W + x]);
        if (cand.length >= m || r > Math.max(M.width, M.height)) break;
        r *= 2;
      }
      const order = cand.sort((a, b) => a[0] - b[0]).slice(0, m).map(a => a[1]);
      trials.push(G.series(order).trace);
    }
    const lo = new Float32Array(M.frames), hi = new Float32Array(M.frames), col = new Float32Array(trials.length);
    for (let f = 0; f < M.frames; f++) { trials.forEach((tr, j) => { col[j] = tr[f]; }); const s = Float32Array.from(col).sort(); lo[f] = s[Math.floor(0.05 * (s.length - 1))]; hi[f] = s[Math.floor(0.95 * (s.length - 1))]; }
    return {lo, hi, m};
  }
  function drawGroups() {
    const dpr = window.devicePixelRatio || 1, w = groupsCv.clientWidth, h = groupsCv.clientHeight;
    if (!w || !G) return;
    groupsCv.width = Math.round(w * dpr); groupsCv.height = Math.round(h * dpr);
    const ctx = groupsCv.getContext("2d"); ctx.scale(dpr, dpr);
    const note = $("smv-groups-note");
    if (!G.groups.length) { ctx.fillStyle = "#9aa8b7"; ctx.font = "12px sans-serif"; ctx.fillText("No significant anatomy cluster for this cell, so there is nothing to compare.", 10, h / 2); note.textContent = ""; return; }
    const T = M.frames, L = 40, X = f => L + (f / Math.max(1, T - 1)) * (w - L - 8);
    const all = []; G.groups.forEach(g => { all.push(...g.s.trace); if (g.band) all.push(...g.band.lo, ...g.band.hi); }); if (G.other) all.push(...G.other.trace);
    let lo = Math.min(...all, 0), hi = Math.max(...all, 0); const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
    const Y = v => 6 + (1 - (v - lo) / (hi - lo)) * (h - 24), unit = (G.other || G.groups[0].s).pct ? "%" : " (intensity units)";
    ctx.font = "10px sans-serif"; ctx.fillStyle = "#9aa8b7"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    const step = niceStep((hi - lo) / 3);
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) { ctx.strokeStyle = Math.abs(v) < step / 1e3 ? "#3b4a5b" : "#1c2630"; ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(w - 8, Y(v)); ctx.stroke(); ctx.fillText(v.toFixed(step < 1 ? 2 : 0), L - 4, Y(v)); }
    ctx.textAlign = "left";
    G.groups.forEach(g => {
      if (!g.band) return; ctx.fillStyle = COL[g.sign]; ctx.globalAlpha = 0.13; ctx.beginPath();
      for (let f = 0; f < T; f++) f ? ctx.lineTo(X(f), Y(g.band.hi[f])) : ctx.moveTo(X(f), Y(g.band.hi[f]));
      for (let f = T - 1; f >= 0; f--) ctx.lineTo(X(f), Y(g.band.lo[f])); ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
    });
    const line = (tr, color, wd) => { ctx.strokeStyle = color; ctx.lineWidth = wd; ctx.beginPath(); tr.forEach((v, f) => f ? ctx.lineTo(X(f), Y(v)) : ctx.moveTo(X(f), Y(v))); ctx.stroke(); ctx.lineWidth = 1; };
    if (G.other) line(G.other.trace, "#c5ced8", 1.4);
    G.groups.forEach(g => line(g.s.trace, COL[g.sign], 1.8));
    const f0i = M.relative_frames.indexOf(0);
    if (f0i >= 0) { ctx.setLineDash([3, 3]); ctx.strokeStyle = COL.spike; ctx.beginPath(); ctx.moveTo(X(f0i), 4); ctx.lineTo(X(f0i), h - 16); ctx.stroke(); ctx.setLineDash([]); }
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(frame), 4); ctx.lineTo(X(frame), h - 16); ctx.stroke(); ctx.lineWidth = 1;
    ctx.fillStyle = "#9aa8b7"; ctx.textBaseline = "bottom"; ctx.fillText(fmtRel(rel(0)), L, h - 1); ctx.fillText(fmtRel(rel(T - 1)), w - 26, h - 1);
    ctx.fillStyle = "#9aa8b7"; ctx.textBaseline = "top"; ctx.fillText("up = depolarization, down = hyperpolarization", L + 4, 0); ctx.textBaseline = "bottom";
    note.innerHTML = `Change from baseline${unit === "%" ? " (% dF/F; up = depolarization)" : unit}: ` +
      G.groups.map(g => `<span style="color:${COL[g.sign]}">${Polarity.of(g.sign)} cluster pixels (${g.idx.length} px)</span>`).join(", ") +
      `, <span style="color:#c5ced8">all other pixels (${G.others.length} px)</span>. Shaded: 5-95% of 120 random compact patches of other pixels with the same pixel count.`;
  }
  function niceStep(raw) { const p = 10 ** Math.floor(Math.log10(raw || 1)), r = raw / p; return (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * p; }
  const fromGroups = e => { const r = groupsCv.getBoundingClientRect(); show(((e.clientX - r.left - 40) / Math.max(1, r.width - 48)) * (M.frames - 1)); };
  let dragG = false;
  groupsCv.addEventListener("pointerdown", e => { stop(); dragG = true; groupsCv.setPointerCapture(e.pointerId); fromGroups(e); });
  groupsCv.addEventListener("pointermove", e => { if (dragG) fromGroups(e); });
  groupsCv.addEventListener("pointerup", () => { dragG = false; });
  function show(f) {
    frame = Math.max(0, Math.min(M.frames - 1, Math.round(f)));
    scrub.value = frame; drawBase(); drawOverlay(); drawTrace(); drawGroups();
    const tags = [];
    if (rel(frame) === 0) tags.push("SPIKE");
    info.textContent = `${M.width}x${M.height} px, ${M.n_peaks} spikes` + (tags.length ? "  |  " + tags.join(" + ") : "");
    info.style.color = tags.length ? COL.neuron : "";
    info.style.fontWeight = tags.length ? "700" : "";
    const sec = rel(frame) / Math.max(1e-9, M.fps);
    readout.textContent = `frame ${frame + 1}/${M.frames} | ${fmtRel(rel(frame))} frames (${sec > 0 ? "+" : ""}${sec.toFixed(3)} s)` +
      (M.anatomy_clusters.length ? ` | ${M.anatomy_clusters.length} anatomy cluster${M.anatomy_clusters.length > 1 ? "s" : ""}` : " | no significant anatomy cluster");
  }
  function stop() { clearInterval(timer); timer = null; play.textContent = "Play"; }
  play.addEventListener("click", () => {
    if (timer) return stop();
    play.textContent = "Pause";
    if (frame >= M.frames - 1) show(0);
    timer = setInterval(() => { if (frame >= M.frames - 1) return stop(); show(frame + 1); }, 66);
  });
  scrub.addEventListener("input", () => { stop(); show(Number(scrub.value)); });
  $("smv-spike").addEventListener("click", () => { stop(); show(M.relative_frames.indexOf(0)); });
  $("smv-contrast").addEventListener("change", () => show(frame));
  $("smv-mode").addEventListener("change", () => { win = globalWindow(); show(frame); });
  $("smv-marks").addEventListener("change", () => show(frame));
  const applySmooth = () => { base.style.filter = $("smv-smooth").checked ? `blur(${S * 0.9}px)` : "none"; base.style.imageRendering = $("smv-smooth").checked ? "auto" : "pixelated"; };
  $("smv-smooth").addEventListener("change", applySmooth); applySmooth();
  let scrubbing = false;
  const fromTrace = e => { const r = trace.getBoundingClientRect(); show(((e.clientX - r.left - 6) / Math.max(1, r.width - 12)) * (M.frames - 1)); };
  trace.addEventListener("pointerdown", e => { stop(); scrubbing = true; trace.setPointerCapture(e.pointerId); fromTrace(e); });
  trace.addEventListener("pointermove", e => { if (scrubbing) fromTrace(e); });
  trace.addEventListener("pointerup", () => { scrubbing = false; });
  window.addEventListener("resize", () => { if (M) { drawTrace(); drawGroups(); } });

  fetch(`/neurons/${NEURON_ID}/anatomy-crop/stack`)
    .then(async res => {
      if (!res.ok) { let m = `Movie unavailable (${res.status})`; try { m = (await res.json()).error || m; } catch (_) {} throw new Error(m); }
      return res.arrayBuffer();
    })
    .then(buf => {
      const hl = new DataView(buf).getUint32(0, true);
      M = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
      data = new Uint16Array(buf.slice(4 + hl));
      base.width = M.width; base.height = M.height;
      ovl.width = M.width * S; ovl.height = M.height * S;
      base.style.width = ovl.style.width = M.width * S + "px"; base.style.height = ovl.style.height = M.height * S + "px";
      stage.style.width = M.width * S + "px"; stage.style.height = M.height * S + "px";
      buildBaseMap();
      win = globalWindow();
      scrub.max = M.frames - 1;
      info.textContent = `${M.width}x${M.height} px, ${M.n_peaks} spikes`;
      buildGroups();
      show(Math.max(0, M.relative_frames.indexOf(0)));
      stage.parentNode._zv.fit();
      setTimeout(() => { if (G) { G.groups.forEach(g => { g.band = randomBand(g); }); drawGroups(); } }, 30);  // reference band after first paint
    })
    .catch(err => { info.textContent = err.message || "Movie unavailable"; });
})();
