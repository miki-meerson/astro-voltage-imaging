// The PCA / ICA of the whole-field STA movie of a cell (the decomposition step, utils/sta_decomposition.py): the map and time course of every component, the explained variance, and a
// movie rebuilt from the first k components. The page only draws what the step saved (/neurons/<id>/decomposition.json). Components are not clusters: their maps use a neutral signed
// colour (green = the same sign as the response, purple = the opposite), never the red and blue of the polarities.
(function () {
  const root = document.getElementById("dec");
  if (!root) return;
  const PAGE = JSON.parse(document.getElementById("page-data").textContent), STATE = JSON.parse(document.getElementById("decomposition-state").textContent);
  const $ = id => document.getElementById(id);
  const {setup, format: fmt} = PlotCanvas;
  const S = 3;  // on-screen pixels per map pixel
  const GREEN = [120, 214, 140], PURPLE = [176, 120, 230], BG = [11, 15, 20], BAR = "#8da2b8", BAR_OFF = "#44505d", GRID = "#1c2630", TEXT = "#9aa8b7";
  const NAMES = {pca: "PC", ica: "IC"};
  const SHOWN_VARIANCES = 10;
  const base = $("dec-base"), ovl = $("dec-ovl"), stage = $("dec-stage"), host = stage.parentNode;
  let D = null, method = "pca", comp = 0, frame = 0, timer = null;
  const limits = new Map();  // the largest absolute value of the rebuilt movie, by method and k (the global contrast)

  const M = () => D[method], name = i => `${NAMES[method]}${i + 1}`;
  const movieView = () => $("dec-view").value === "movie";
  const kept = () => Math.max(1, Math.min(D.n_components, Number($("dec-k").value) || 1));
  const maxAbs = values => values.reduce((m, v) => Math.max(m, Math.abs(v)), 0);

  // the values of the map of component i, and of frame t rebuilt from components 0..k-1 (a component's map times its scale, times its time course)
  const mapValues = i => M().maps[i].map(v => v * M().scales[i]);
  function frameValues(t, k) {
    const out = new Float32Array(D.width * D.height);
    for (let i = 0; i < k; i++) { const c = M().courses[i][t] * M().scales[i], map = M().maps[i]; for (let p = 0; p < out.length; p++) out[p] += c * map[p]; }
    return out;
  }
  function globalLimit(k) {
    const key = `${method}:${k}`;
    if (!limits.has(key)) limits.set(key, Math.max(1e-12, ...D.rels.map((_, t) => maxAbs(frameValues(t, k)))));
    return limits.get(key);
  }

  function paint(canvas, values, limit) {
    canvas.width = D.width; canvas.height = D.height;
    const ctx = canvas.getContext("2d"), img = ctx.createImageData(D.width, D.height);
    for (let i = 0; i < values.length; i++) {
      const v = Math.max(-1, Math.min(1, values[i] / limit)), c = v >= 0 ? GREEN : PURPLE, a = Math.abs(v);
      for (let j = 0; j < 3; j++) img.data[4 * i + j] = BG[j] + (c[j] - BG[j]) * a;
      img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  function drawStage() {
    const movie = movieView(), k = kept();
    $("dec-k-field").hidden = !movie;
    for (const id of ["dec-scrub", "dec-play", "dec-spike"]) $(id).disabled = !movie;
    if (!movie) {
      paint(base, mapValues(comp), M().scales[comp]);
      $("dec-readout").textContent = `${name(comp)}: ${(100 * M().variance[comp]).toFixed(1)}% of the ${method === "pca" ? "variance" : "variance of the movie rebuilt from the independent components"}`;
    } else {
      const values = frameValues(frame, k);
      paint(base, values, $("dec-contrast").value === "frame" ? maxAbs(values) || 1 : globalLimit(k));
      $("dec-readout").textContent = `frame ${D.rels[frame]} (${(D.rels[frame] / D.fps * 1000).toFixed(0)} ms from the spike), rebuilt from ${name(0)} to ${name(k - 1)}`;
    }
    $("dec-scrub").value = frame;
    const ctx = ovl.getContext("2d");
    ctx.clearRect(0, 0, ovl.width, ovl.height);
    if ($("dec-roi").checked && D.roi.length) { ctx.strokeStyle = Polarity.NEURON; ctx.lineWidth = 1.5; FovOutline.draw(ctx, D.roi.map(p => p[0]), D.roi.map(p => p[1]), D.width, S); }
  }

  // the explained variance of the first components (bars) and what the first n components explain together (line): the kept components are the lighter bars
  function drawVariance() {
    const [ctx, W, H] = setup($("dec-ev")), L = 44, R = 10, T = 10, B = 24, shown = D.variance_explained.slice(0, SHOWN_VARIANCES), slot = (W - L - R) / shown.length;
    const top = Math.max(...shown) * 1.1, Y = v => T + (1 - v / top) * (H - T - B), YC = v => T + (1 - v) * (H - T - B);
    ctx.font = "11px sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const v of TimeAxis.nice(0, top, 4)) { ctx.strokeStyle = GRID; ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - R, Y(v)); ctx.stroke(); ctx.fillStyle = TEXT; ctx.fillText(`${fmt(v * 100)}%`, L - 6, Y(v)); }
    shown.forEach((v, i) => {
      const x = L + i * slot + slot * 0.2, w = slot * 0.6;
      ctx.fillStyle = i >= D.n_components ? BAR_OFF : i === comp ? `rgb(${GREEN})` : BAR; ctx.fillRect(x, Y(v), w, Y(0) - Y(v));
      ctx.fillStyle = TEXT; ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(String(i + 1), x + w / 2, Y(0) + 4);
    });
    let sum = 0; ctx.strokeStyle = "#e7edf4"; ctx.lineWidth = 1.5; ctx.beginPath();
    shown.forEach((v, i) => { sum += v; const x = L + i * slot + slot / 2; i ? ctx.lineTo(x, YC(sum)) : ctx.moveTo(x, YC(sum)); }); ctx.stroke(); ctx.lineWidth = 1;
    ctx.fillStyle = "#e7edf4"; ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText("line: explained together (0 to 100%)", L + 6, T);
  }

  // the time course of the chosen component against the seconds from the spike, with the spike and, in the movie view, the frame shown
  function drawCourse() {
    const [ctx, W, H] = setup($("dec-tc")), L = 44, R = 10, T = 10, B = 36, t = D.rels.map(r => r / D.fps), course = M().courses[comp].map(v => v * 100);
    const lo = Math.min(0, ...course), hi = Math.max(0, ...course), pad = (hi - lo) * 0.08 || 1, X = v => L + (v - t[0]) / (t[t.length - 1] - t[0]) * (W - L - R), Y = v => T + (1 - (v - (lo - pad)) / (hi - lo + 2 * pad)) * (H - T - B);
    $("dec-tc").geometry = {L, R, W};
    ctx.font = "11px sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const v of TimeAxis.nice(lo - pad, hi + pad, 4)) { ctx.strokeStyle = v === 0 ? "#3b4a5b" : GRID; ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - R, Y(v)); ctx.stroke(); ctx.fillStyle = TEXT; ctx.fillText(fmt(v), L - 6, Y(v)); }
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = TEXT;
    for (const v of TimeAxis.nice(t[0], t[t.length - 1], 8)) ctx.fillText(fmt(v), X(v), H - B + 4);
    ctx.fillText("seconds from the spike", L + (W - L - R) / 2, H - 12);
    ctx.strokeStyle = `rgb(${GREEN})`; ctx.lineWidth = 2; ctx.beginPath(); course.forEach((v, i) => i ? ctx.lineTo(X(t[i]), Y(v)) : ctx.moveTo(X(t[i]), Y(v))); ctx.stroke(); ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]); ctx.strokeStyle = "rgba(255,255,255,0.45)"; ctx.beginPath(); ctx.moveTo(X(0), T); ctx.lineTo(X(0), H - B); ctx.stroke(); ctx.setLineDash([]);
    if (movieView()) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(t[frame]), T); ctx.lineTo(X(t[frame]), H - B); ctx.stroke(); ctx.lineWidth = 1; }
    $("dec-tc-title").textContent = `Time course of ${name(comp)} (dF/F, %)`;
  }

  function buildThumbs() {
    const box = $("dec-thumbs"); box.innerHTML = "";
    for (let i = 0; i < D.n_components; i++) {
      const cell = document.createElement("div"), canvas = document.createElement("canvas");
      cell.className = "dec-thumb"; cell.append(canvas, document.createTextNode(name(i)));
      paint(canvas, mapValues(i), M().scales[i]);
      cell.addEventListener("click", () => { comp = i; draw(); });
      box.appendChild(cell);
    }
  }
  const markThumbs = () => [...$("dec-thumbs").children].forEach((cell, i) => cell.classList.toggle("on", i === comp));

  function draw() { drawStage(); drawVariance(); drawCourse(); markThumbs(); }

  function stop() { clearInterval(timer); timer = null; $("dec-play").textContent = "Play"; }
  function show(f) { frame = Math.max(0, Math.min(D.rels.length - 1, Math.round(f))); draw(); }

  function build() {
    for (const el of [base, ovl]) { el.style.width = `${D.width * S}px`; el.style.height = `${D.height * S}px`; }
    ovl.width = D.width * S; ovl.height = D.height * S;
    stage.style.width = `${D.width * S}px`; stage.style.height = `${D.height * S}px`;
    if (ScaleBar.PIXEL_UM) host.dataset.um = ScaleBar.umPerPixel(D.spatial_bin) / S; else delete host.dataset.um;  // a map pixel is a block of recorded pixels
    $("dec-scrub").max = D.rels.length - 1; $("dec-k").max = D.n_components; $("dec-k").value = Math.min(3, D.n_components);
    frame = Math.max(0, D.rels.indexOf(0));
    const share = D.variance_explained.slice(0, D.n_components).reduce((u, v) => u + v, 0);
    $("dec-info").textContent = `${D.n_components} components, ${(100 * share).toFixed(0)}% of the variance, ${D.n_peaks} spikes${D.ica_converged ? "" : ", ICA did not converge"}`;
    host._zv.fit(); host._zv.update();
    buildThumbs(); draw();
  }

  function message(text) {  // the step has not run (or is running, or left this cell out): say so, with the button to start it by hand
    const box = $("dec-msg"); box.hidden = false; $("dec-body").hidden = true; box.textContent = text;
    if (!["running", "pending", "waiting", "queued"].includes(STATE.status)) {
      const form = document.createElement("form"), button = document.createElement("button"), back = document.createElement("input");
      form.method = "post"; form.action = `/experiments/${STATE.experiment_id}/run-decomposition`; form.style.marginTop = "10px";
      back.type = "hidden"; back.name = "back"; back.value = "cells"; button.className = "secondary"; button.type = "submit"; button.textContent = STATE.status === "not_run" ? "Run PCA / ICA" : "Run PCA / ICA again";
      form.append(back, button); box.appendChild(form);
    } else box.append(` (the step is ${STATE.status})`);
  }

  async function load() {
    if (D) { $("dec-msg").hidden = true; $("dec-body").hidden = false; return; }
    const box = $("dec-msg"); box.hidden = false; box.textContent = "Loading...";
    try {
      const res = await fetch(`/neurons/${PAGE.neuronId}/decomposition.json`), body = await res.json();
      if (!res.ok) throw new Error(body.message || `Not available (${res.status})`);
      D = body; box.hidden = true; $("dec-body").hidden = false;
      build();
    } catch (error) { message(error.message); }
  }

  $("cell-source").addEventListener("click", e => {
    const button = e.target.closest("button[data-src]");
    if (!button || button.disabled) return;
    [...$("cell-source").children].forEach(b => b.classList.toggle("on", b === button));
    const source = button.dataset.src;
    $("smv").hidden = source !== "sta"; root.hidden = source === "sta";
    if (source === "sta") { stop(); window.dispatchEvent(new Event("resize")); return; }
    method = source;
    load().then(() => { if (D) { buildThumbs(); host._zv.fit(); host._zv.update(); draw(); } });
  });
  for (const id of ["dec-view", "dec-k", "dec-contrast", "dec-roi"]) $(id).addEventListener("change", draw);
  $("dec-k").addEventListener("input", draw);
  $("dec-smooth").addEventListener("change", () => { base.style.filter = $("dec-smooth").checked ? `blur(${S * 0.9}px)` : "none"; base.style.imageRendering = $("dec-smooth").checked ? "auto" : "pixelated"; });
  $("dec-smooth").dispatchEvent(new Event("change"));
  $("dec-scrub").addEventListener("input", () => { stop(); show(Number($("dec-scrub").value)); });
  $("dec-spike").addEventListener("click", () => { stop(); show(D.rels.indexOf(0)); });
  $("dec-play").addEventListener("click", () => {
    if (timer) return stop();
    $("dec-play").textContent = "Pause";
    if (frame >= D.rels.length - 1) show(0);
    timer = setInterval(() => { if (frame >= D.rels.length - 1) return stop(); show(frame + 1); }, 66);
  });
  $("dec-ev").addEventListener("click", e => {  // a click on a bar chooses that component (only the kept ones have a map)
    const box = e.currentTarget.getBoundingClientRect(), slot = (box.width - 54) / Math.min(SHOWN_VARIANCES, D.variance_explained.length), i = Math.floor((e.clientX - box.left - 44) / slot);
    if (i >= 0 && i < D.n_components) { comp = i; draw(); }
  });
  let dragging = false;
  const fromCourse = e => { const g = $("dec-tc").geometry, box = $("dec-tc").getBoundingClientRect(); if (movieView()) { stop(); show(((e.clientX - box.left - g.L) / Math.max(1, g.W - g.L - g.R)) * (D.rels.length - 1)); } };
  $("dec-tc").addEventListener("pointerdown", e => { dragging = true; $("dec-tc").setPointerCapture(e.pointerId); fromCourse(e); });
  $("dec-tc").addEventListener("pointermove", e => { if (dragging) fromCourse(e); });
  $("dec-tc").addEventListener("pointerup", () => { dragging = false; });
  window.addEventListener("resize", () => { if (D && !root.hidden) { drawVariance(); drawCourse(); } });
})();
