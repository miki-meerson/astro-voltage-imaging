// The SNAPT propagation maps of a cell (the SNAPT step, utils/snapt.py): for the positive or the negative clusters, the delay, width, amplitude and fit quality of every pixel against the
// clusters' mean trace, and the fitted movie. The page only draws what the step saved (/neurons/<id>/snapt.json). Every view uses one rainbow scale: red lowest, blue highest.
(function () {
  const root = document.getElementById("snapt");
  if (!root) return;
  const PAGE = JSON.parse(document.getElementById("page-data").textContent), STATE = JSON.parse(document.getElementById("snapt-state").textContent);
  const $ = id => document.getElementById(id);
  const {setup, format: fmt} = PlotCanvas;
  const S = 3;  // on-screen pixels per map pixel
  // rainbow: red = the lowest value (the earliest delay), blue = the highest (the latest)
  const SCALE = [[225, 40, 40], [245, 190, 45], [90, 200, 80], [50, 200, 220], [60, 90, 235]], BG = [11, 15, 20], GRID = "#1c2630", TEXT = "#9aa8b7", LINE = "#8bd3c7";
  const VIEWS = {delay: "Delay (s, later is positive)", width: "Width (1 = as wide as the kernel)", amplitude: "Amplitude", r2: "R²", movie: "Fitted movie"};
  const base = $("snapt-base"), ovl = $("snapt-ovl"), stage = $("snapt-stage"), host = stage.parentNode;
  let D = null, frame = 0, timer = null;
  const movies = new Map();  // the fitted movie of a sign as bytes, once fetched

  const R = () => D.signs[$("snapt-sign").value], view = () => $("snapt-view").value;
  const colour = v => {  // the scale at v between 0 and 1
    const x = Math.max(0, Math.min(1, v)) * (SCALE.length - 1), i = Math.min(SCALE.length - 2, Math.floor(x)), f = x - i;
    return SCALE[i].map((c, j) => c + (SCALE[i + 1][j] - c) * f);
  };
  const percentile = (values, p) => { const s = [...values].sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : 0; };

  // a pixel is drawn when its fit was accepted and explains at least the chosen share of its trace
  const minR2 = () => Number($("snapt-r2").value), shown = (r, i) => r.accepted[i] && r.maps.r2[i] >= minR2();
  const top = (values, keep) => percentile(values.filter(keep), 0.99);

  // the range the colours cover, over the drawn pixels only: their 2nd to 98th percentile (delay, width, so a few outliers do not squash the scale), 0 to their 99th percentile (amplitude), 0 to 1 (R²)
  function range(name) {
    const r = R(), values = r.maps[name].filter((_, i) => shown(r, i));
    if (name === "r2") return [0, 1];
    if (name === "amplitude") return [0, Math.max(1e-9, percentile(values, 0.99))];
    const lo = percentile(values, 0.02), hi = percentile(values, 0.98);
    return hi > lo ? [lo, hi] : [lo - 0.5, hi + 0.5];
  }

  function paint(pixel) {  // pixel(i) -> an [r, g, b] colour, drawn on the base canvas
    base.width = D.width; base.height = D.height;
    const ctx = base.getContext("2d"), img = ctx.createImageData(D.width, D.height);
    for (let i = 0; i < D.width * D.height; i++) { const c = pixel(i); for (let j = 0; j < 3; j++) img.data[4 * i + j] = c[j]; img.data[4 * i + 3] = 255; }
    ctx.putImageData(img, 0, 0);
  }

  function drawStage() {
    const r = R(), name = view(), movie = name === "movie", [lo, hi] = range(movie ? "delay" : name);
    for (const id of ["snapt-scrub", "snapt-play"]) $(id).disabled = !movie;
    if (movie) {  // every pixel has the colour of its delay and lights up as its fitted response rises: the response (amplitude included) is the brightness
      const bytes = movies.get($("snapt-sign").value), n = D.width * D.height, [dlo, dhi] = range("delay");
      paint(i => {
        if (!shown(r, i)) return BG;
        const c = colour((r.maps.delay[i] - dlo) / (dhi - dlo)), w = bytes[frame * n + i] / 255;
        return c.map((v, j) => BG[j] + (v - BG[j]) * w);
      });
      $("snapt-scrub").value = frame;
      $("snapt-readout").textContent = `${r.movie.seconds[frame].toFixed(2)} s from the spike`;
    } else {
      const weight = $("snapt-weight").checked && name !== "amplitude", full = Math.max(1e-9, top(r.maps.amplitude, (_, i) => shown(r, i)));
      paint(i => {
        if (name !== "r2" && !shown(r, i)) return BG;  // the R² map shows every pixel: it is what the gate is chosen from
        const c = colour((r.maps[name][i] - lo) / (hi - lo)), w = weight ? Math.max(0, Math.min(1, 3 * r.maps.amplitude[i] / full - 0.1)) : 1;
        return c.map((v, j) => BG[j] + (v - BG[j]) * w);
      });
      $("snapt-readout").textContent = VIEWS[name];
    }
    const ctx = ovl.getContext("2d");
    ctx.clearRect(0, 0, ovl.width, ovl.height);
    if ($("snapt-roi").checked && D.roi.length) { ctx.strokeStyle = Polarity.NEURON; ctx.lineWidth = 1.5; FovOutline.draw(ctx, D.roi.map(p => p[0]), D.roi.map(p => p[1]), D.width, S); }
    if ($("snapt-clusters").checked) {  // the clusters the kernels are made from: the colour of their sign, solid for the whole-field result and dashed for the anatomy one, on a dark casing
      for (const [width, style] of [[4, () => "rgba(11,15,20,0.85)"], [2, c => Polarity.color(c.sign)]]) {
        ctx.lineWidth = width;
        for (const c of D.clusters) { ctx.strokeStyle = style(c); ctx.setLineDash(c.type === "anatomy" ? [6, 4] : []); FovOutline.draw(ctx, c.y, c.x, D.width, S); }
      }
      ctx.setLineDash([]); ctx.lineWidth = 1;
    }
    drawBar([lo, hi]);
  }

  // the colour scale with its two ends (the delay in the fitted movie)
  function drawBar([lo, hi]) {
    const [ctx, W, H] = setup($("snapt-bar")), L = 6, T = 4, bar = 10;
    for (let x = 0; x < W - 2 * L; x++) { ctx.fillStyle = `rgb(${colour(x / (W - 2 * L - 1))})`; ctx.fillRect(L + x, T, 1, bar); }
    ctx.font = "11px sans-serif"; ctx.fillStyle = TEXT; ctx.textBaseline = "top";
    ctx.textAlign = "left"; ctx.fillText(fmt(lo), L, T + bar + 3);
    ctx.textAlign = "right"; ctx.fillText(fmt(hi), W - L, T + bar + 3);
  }

  function drawKernel() {
    const r = R(), [ctx, W, H] = setup($("snapt-kernel")), L = 36, Rm = 10, T = 10, B = 34, t = r.kernel_seconds, X = v => L + (v - t[0]) / (t[t.length - 1] - t[0]) * (W - L - Rm), Y = v => T + (1 - v) * (H - T - B);
    ctx.font = "11px sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const v of [0, 0.5, 1]) { ctx.strokeStyle = GRID; ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - Rm, Y(v)); ctx.stroke(); ctx.fillStyle = TEXT; ctx.fillText(String(v), L - 6, Y(v)); }
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = TEXT;
    for (const v of TimeAxis.nice(t[0], t[t.length - 1], 6)) ctx.fillText(fmt(v), X(v), H - B + 4);
    ctx.fillText("seconds from the spike", L + (W - L - Rm) / 2, H - 14);
    ctx.strokeStyle = LINE; ctx.lineWidth = 2; ctx.beginPath(); r.kernel.forEach((v, i) => i ? ctx.lineTo(X(t[i]), Y(v)) : ctx.moveTo(X(t[i]), Y(v))); ctx.stroke(); ctx.lineWidth = 1;
    if (t[0] <= 0 && t[t.length - 1] >= 0) { ctx.setLineDash([3, 3]); ctx.strokeStyle = "rgba(255,255,255,0.45)"; ctx.beginPath(); ctx.moveTo(X(0), T); ctx.lineTo(X(0), H - B); ctx.stroke(); ctx.setLineDash([]); }
  }

  // delay (ms) against the distance (micrometres) from the centre of the clusters, for the drawn pixels: a dot each, the median of each distance bin, and the least-squares line with its slope
  function drawDistance() {
    const r = R(), [ctx, W, H] = setup($("snapt-dist")), L = 44, Rm = 10, T = 10, B = 34, BINS = 12, pts = [];
    r.accepted.forEach((_, i) => { if (shown(r, i)) pts.push([ScaleBar.length(Math.hypot(Math.floor(i / D.width) - r.origin[0], i % D.width - r.origin[1]), D.spatial_bin), r.maps.delay[i] * 1000]); });
    $("snapt-slope").textContent = "";
    if (pts.length < 10) return;
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]), xmax = xs.reduce((m, v) => Math.max(m, v), 0) || 1, lo = percentile(ys, 0.01), hi = percentile(ys, 0.99), pad = (hi - lo) * 0.1 || 1;
    const X = v => L + v / xmax * (W - L - Rm), Y = v => T + (1 - (v - (lo - pad)) / (hi - lo + 2 * pad)) * (H - T - B);
    ctx.font = "11px sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const v of TimeAxis.nice(lo - pad, hi + pad, 4)) { ctx.strokeStyle = v === 0 ? "#3b4a5b" : GRID; ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - Rm, Y(v)); ctx.stroke(); ctx.fillStyle = TEXT; ctx.fillText(fmt(v), L - 6, Y(v)); }
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = TEXT;
    for (const v of TimeAxis.nice(0, xmax, 6)) ctx.fillText(fmt(v), X(v), H - B + 4);
    ctx.fillText(`distance (${ScaleBar.UNIT}) / delay (ms)`, L + (W - L - Rm) / 2, H - 14);
    ctx.fillStyle = "rgba(139,211,199,0.18)";
    const step = Math.max(1, Math.floor(pts.length / 3000));  // at most about 3000 dots
    for (let i = 0; i < pts.length; i += step) ctx.fillRect(X(xs[i]) - 1, Y(ys[i]) - 1, 2, 2);
    const medians = [];
    for (let b = 0; b < BINS; b++) { const inBin = pts.filter(p => p[0] >= b * xmax / BINS && (p[0] < (b + 1) * xmax / BINS || b === BINS - 1)).map(p => p[1]); if (inBin.length >= 5) medians.push([(b + 0.5) * xmax / BINS, percentile(inBin, 0.5)]); }
    ctx.strokeStyle = "#e7edf4"; ctx.lineWidth = 2; ctx.beginPath(); medians.forEach((m, i) => i ? ctx.lineTo(X(m[0]), Y(m[1])) : ctx.moveTo(X(m[0]), Y(m[1]))); ctx.stroke(); ctx.lineWidth = 1;
    const mx = xs.reduce((u, v) => u + v, 0) / xs.length, my = ys.reduce((u, v) => u + v, 0) / ys.length;
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < xs.length; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; syy += (ys[i] - my) ** 2; }
    if (!(sxx > 0 && syy > 0)) return;
    const slope = sxy / sxx;  // ms per unit of distance
    ctx.setLineDash([5, 4]); ctx.strokeStyle = "#f2c14e"; ctx.beginPath(); ctx.moveTo(X(0), Y(my - slope * mx)); ctx.lineTo(X(xmax), Y(my + slope * (xmax - mx))); ctx.stroke(); ctx.setLineDash([]);
    $("snapt-slope").textContent = `slope ${slope.toFixed(3)} ms/${ScaleBar.UNIT}, ${Math.abs(slope) > 1e-9 ? `speed ${(1000 / Math.abs(slope)).toFixed(0)} ${ScaleBar.UNIT}/s ${slope > 0 ? "away from" : "towards"} the centre` : "no trend"}, r = ${(sxy / Math.sqrt(sxx * syy)).toFixed(2)}, ${pts.length} pixels`;
  }

  const draw = () => {
    const r = R(), count = r.accepted.reduce((n, _, i) => n + (shown(r, i) ? 1 : 0), 0);
    $("snapt-r2-value").textContent = minR2().toFixed(2);
    drawStage(); drawKernel(); drawDistance();
    $("snapt-info").textContent = `${r.n_clusters} cluster${r.n_clusters > 1 ? "s" : ""}, ${(100 * count / r.accepted.length).toFixed(0)}% of the pixels pass Min R², ${D.components} components`;
  };
  function stop() { clearInterval(timer); timer = null; $("snapt-play").textContent = "Play"; }

  async function loadMovie() {  // the fitted movie of the chosen sign, fetched once when first shown
    const sign = $("snapt-sign").value;
    if (movies.has(sign)) return;
    const res = await fetch(`/neurons/${PAGE.neuronId}/snapt.json?movie=${sign}`), body = await res.json();
    if (!res.ok) throw new Error(body.message || `Not available (${res.status})`);
    const bin = atob(body.signs[sign].movie.data), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    D.signs[sign].movie = {seconds: body.signs[sign].movie.seconds, frames: body.signs[sign].movie.frames};
    movies.set(sign, bytes);
  }

  async function show() {
    stop();
    if (view() === "movie") {
      try { await loadMovie(); } catch (error) { $("snapt-readout").textContent = error.message; return; }
      $("snapt-scrub").max = R().movie.frames - 1; frame = Math.min(frame, R().movie.frames - 1);
    }
    draw();
  }

  function build() {
    for (const el of [base, ovl]) { el.style.width = `${D.width * S}px`; el.style.height = `${D.height * S}px`; }
    ovl.width = D.width * S; ovl.height = D.height * S;
    stage.style.width = `${D.width * S}px`; stage.style.height = `${D.height * S}px`;
    if (ScaleBar.PIXEL_UM) host.dataset.um = ScaleBar.umPerPixel(D.spatial_bin) / S; else delete host.dataset.um;
    $("snapt-sign").innerHTML = Object.keys(D.signs).map(s => `<option value="${s}">${s === "positive" ? "Positive" : "Negative"}</option>`).join("");
    host._zv.fit(); host._zv.update();
    show();
  }

  function message(text) {  // the step has not run (or is running, or left this cell out): say so, with the button to start it by hand
    const box = $("snapt-msg"); box.hidden = false; $("snapt-body").hidden = true; box.textContent = text;
    if (!["running", "pending", "waiting", "queued"].includes(STATE.status)) {
      const form = document.createElement("form"), button = document.createElement("button"), back = document.createElement("input");
      form.method = "post"; form.action = `/experiments/${STATE.experiment_id}/run-snapt`; form.style.marginTop = "10px";
      back.type = "hidden"; back.name = "back"; back.value = "cells"; button.className = "secondary"; button.type = "submit"; button.textContent = STATE.status === "not_run" ? "Run SNAPT" : "Run SNAPT again";
      form.append(back, button); box.appendChild(form);
    } else box.append(` (the step is ${STATE.status})`);
  }

  async function load() {
    if (D) { $("snapt-msg").hidden = true; $("snapt-body").hidden = false; return; }
    const box = $("snapt-msg"); box.hidden = false; box.textContent = "Loading...";
    try {
      const res = await fetch(`/neurons/${PAGE.neuronId}/snapt.json`), body = await res.json();
      if (!res.ok) throw new Error(body.message || `Not available (${res.status})`);
      D = body; box.hidden = true; $("snapt-body").hidden = false;
      build();
    } catch (error) { message(error.message); }
  }

  $("cell-source").addEventListener("click", e => {
    const button = e.target.closest("button[data-src]");
    if (!button || button.disabled) return;
    root.hidden = button.dataset.src !== "snapt";
    if (root.hidden) return stop();
    load().then(() => { if (D) { host._zv.fit(); host._zv.update(); show(); } });
  });
  for (const id of ["snapt-sign", "snapt-view"]) $(id).addEventListener("change", show);
  for (const id of ["snapt-weight", "snapt-roi", "snapt-clusters", "snapt-r2"]) $(id).addEventListener("change", draw);
  $("snapt-r2").addEventListener("input", draw);
  $("snapt-smooth").addEventListener("change", () => { base.style.filter = $("snapt-smooth").checked ? `blur(${S * 0.9}px)` : "none"; base.style.imageRendering = $("snapt-smooth").checked ? "auto" : "pixelated"; });
  $("snapt-smooth").dispatchEvent(new Event("change"));
  $("snapt-scrub").addEventListener("input", () => { stop(); frame = Number($("snapt-scrub").value); draw(); });
  $("snapt-play").addEventListener("click", () => {
    if (timer) return stop();
    $("snapt-play").textContent = "Pause";
    if (frame >= R().movie.frames - 1) frame = 0;
    timer = setInterval(() => { if (frame >= R().movie.frames - 1) return stop(); frame += 1; draw(); }, 66);
  });
  window.addEventListener("resize", () => { if (D && !root.hidden) { drawKernel(); drawDistance(); drawBar(range(view() === "movie" ? "delay" : view())); } });
})();
