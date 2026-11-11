// The delay and distance panel of the Hypotheses tab: for every significant cluster, the delay between its response peak and its neuron's calcium peak against the distance of the
// cluster from the neuron, in the distance the user picks. Next to it, for the neuron and cluster picked: the field of view with the points the distance is measured between, and the
// two traces with the peaks the delay is measured between. Everything is drawn from the saved cluster map (app/services/cluster_map.py `cluster_delays`, utils/delay_distance.py).
(function () {
  const root = document.getElementById("delay-panel");
  if (!root) return;
  const PAGE = JSON.parse(document.getElementById("page-data").textContent);
  const EXP_ID = PAGE.expId, HYPOTHESIS = PAGE.hypothesis;
  const $ = id => document.getElementById(id);
  const {setup, format: fmt, pValue} = PlotCanvas;
  const MEASURES = {"center-center": "ROI centre to cluster centre", "nearest-nearest": "nearest ROI pixel to nearest cluster pixel", "center-nearest": "cluster centre to nearest ROI pixel", along: "along the ROI's long axis", across: "across the ROI's long axis"};
  const L = 56, R = 16, T = 26, B = 40;
  const GRID = "#1c2630", TEXT = "#9aa8b7", BRIGHT = "#edf3fa";
  const image = new Image();
  let D = null, pts = [], types = null, current = null;

  const measure = () => $("dd-measure").value;
  const usable = () => D.points.filter(p => p.distances[measure()] !== null);
  const ms = seconds => seconds * 1000;
  const pointsOf = cell => usable().filter(p => p.cell === cell);
  const label = p => `${p.result_type}, ${p.sign}${p.p == null ? "" : ", p=" + pValue(p.p)}`;

  function draw() {
    const [ctx, W, H] = setup($("dd-scatter")), data = usable(), key = measure();
    pts = [];
    ctx.font = "12px sans-serif"; ctx.textBaseline = "top"; ctx.textAlign = "left";
    if (!data.length) { ctx.fillStyle = TEXT; ctx.fillText("No clusters with this distance", 20, 30); return; }
    const xs = data.map(p => p.distances[key]), ys = data.map(p => ms(p.delay));
    const xMax = Math.max(...xs) * 1.08 || 1, span = Math.max(...ys.map(Math.abs), 10) * 1.15;
    const X = v => L + v / xMax * (W - L - R), Y = v => T + (1 - (v + span) / (2 * span)) * (H - T - B);
    ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.font = "11px sans-serif";
    TimeAxis.nice(-span, span, 6).forEach(v => { ctx.strokeStyle = v === 0 ? "#3b4a5b" : GRID; ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - R, Y(v)); ctx.stroke(); ctx.fillStyle = TEXT; ctx.fillText(fmt(v), L - 6, Y(v)); });
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    TimeAxis.nice(0, xMax, 8).forEach(v => { ctx.strokeStyle = GRID; ctx.beginPath(); ctx.moveTo(X(v), T); ctx.lineTo(X(v), H - B); ctx.stroke(); ctx.fillStyle = TEXT; ctx.fillText(fmt(v), X(v), H - B + 4); });
    const stat = D.stats[key];
    if (stat) {  // the straight line the slope is of (seconds per pixel -> milliseconds)
      ctx.strokeStyle = "#e7edf4"; ctx.lineWidth = 1.4; ctx.setLineDash([6, 4]); ctx.beginPath();
      ctx.moveTo(X(0), Y(ms(stat.intercept))); ctx.lineTo(X(xMax), Y(ms(stat.intercept + stat.slope * xMax))); ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1;
    }
    data.forEach(p => {
      const x = X(p.distances[key]), y = Y(ms(p.delay));
      ctx.globalAlpha = 0.85; ctx.fillStyle = Polarity.color(p.sign); ctx.beginPath();
      if (p.result_type === "all") ctx.rect(x - 5, y - 5, 10, 10); else ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = "#0b0f14"; ctx.stroke(); pts.push({x, y, p});
    });
    if (current && current.distances[key] !== null) {  // the cluster the picture on the right shows
      ctx.strokeStyle = BRIGHT; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(current.distances[key]), Y(ms(current.delay)), 9, 0, 2 * Math.PI); ctx.stroke(); ctx.lineWidth = 1;
    }
    ctx.fillStyle = BRIGHT; ctx.font = "12px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText(`distance, ${MEASURES[key]} (px)`, L + (W - L - R) / 2, H - 4);
    ctx.save(); ctx.translate(12, T + (H - T - B) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("cluster peak minus neuron peak (ms)", 0, 0); ctx.restore();
    ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillStyle = TEXT; ctx.font = "11px sans-serif";
    ctx.fillText(stat ? `${key}: n=${stat.n}, Spearman rho=${stat.rho.toFixed(2)}, p=${pValue(stat.p)}; line ${ms(stat.slope).toFixed(2)} ms/px, p=${pValue(stat.slope_p)}` : `${key}: too few points for a correlation`, L, 5);
  }

  // The mean astro image as the background, with the neuron's ROI (grey), the cluster the distance is computed by (coloured) and the two points the distance of the chosen kind is
  // measured between: the one on the neuron (blue ring) and the one on the cluster (orange ring); `along` and `across` also show the foot of the cluster on the long axis (white).
  // `full`: the whole field of view (to see the distance relative to the field); otherwise zoomed on the ROI and the cluster.
  function drawFov(canvas, full) {
    const [ctx, W, H] = setup(canvas);
    if (!current) return;
    const roi = D.rois[current.cell], key = measure(), path = current.paths[key], axis = key === "along" || key === "across";
    const neuronPoint = axis ? current.paths.along[0] : path[0], clusterPoint = axis ? current.paths.across[1] : path[1];
    const all = [...roi.ys.map((y, i) => [y, roi.xs[i]]), ...current.ys.map((y, i) => [y, current.xs[i]]), neuronPoint, clusterPoint];
    const y0 = full ? 0 : Math.max(0, Math.min(...all.map(a => a[0])) - 3), y1 = full ? image.naturalHeight : Math.min(image.naturalHeight, Math.max(...all.map(a => a[0])) + 4);
    const x0 = full ? 0 : Math.max(0, Math.min(...all.map(a => a[1])) - 3), x1 = full ? image.naturalWidth : Math.min(image.naturalWidth, Math.max(...all.map(a => a[1])) + 4);
    const scale = Math.min(W / (x1 - x0), H / (y1 - y0)), offX = (W - (x1 - x0) * scale) / 2, offY = (H - (y1 - y0) * scale) / 2;
    const X = x => offX + (x - x0 + 0.5) * scale, Y = y => offY + (y - y0 + 0.5) * scale;  // the centre of a pixel
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(image, x0, y0, x1 - x0, y1 - y0, offX, offY, (x1 - x0) * scale, (y1 - y0) * scale);
    const paint = (ys, xs, color, alpha) => { ctx.globalAlpha = alpha; ctx.fillStyle = color; ys.forEach((y, i) => ctx.fillRect(X(xs[i]) - scale / 2, Y(y) - scale / 2, Math.max(scale, 1.5), Math.max(scale, 1.5))); ctx.globalAlpha = 1; };
    paint(roi.ys, roi.xs, Polarity.NEURON, 0.5);
    paint(current.ys, current.xs, Polarity.color(current.sign), 0.7);
    const line = (a, b, dashed) => { ctx.strokeStyle = BRIGHT; ctx.lineWidth = 2; ctx.setLineDash(dashed ? [5, 4] : []); ctx.beginPath(); ctx.moveTo(X(a[1]), Y(a[0])); ctx.lineTo(X(b[1]), Y(b[0])); ctx.stroke(); ctx.setLineDash([]); };
    if (axis) { line(current.paths.along[0], current.paths.along[1], false); line(current.paths.across[0], current.paths.across[1], true); } else line(path[0], path[1], false);
    const radius = full ? 6 : 7, mark = (q, color) => { ctx.strokeStyle = "#0b0f14"; ctx.lineWidth = 4.5; ctx.beginPath(); ctx.arc(X(q[1]), Y(q[0]), radius, 0, 2 * Math.PI); ctx.stroke(); ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.stroke(); ctx.lineWidth = 1; };
    if (axis) { ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(X(current.paths.along[1][1]), Y(current.paths.along[1][0]), 3, 0, 2 * Math.PI); ctx.fill(); }
    mark(neuronPoint, Polarity.NEURON); mark(clusterPoint, Polarity.color(current.sign));
    ctx.font = "12px sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillStyle = BRIGHT;
    ctx.fillText(full ? `field of view, cell ${current.cell}` : `${key} ${current.distances[key].toFixed(1)} px`, 8, 6);
    if (!full) { ctx.font = "11px sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "bottom"; ctx.fillStyle = Polarity.NEURON; ctx.fillText("on the neuron", W - 6, H - 18); ctx.fillStyle = Polarity.color(current.sign); ctx.fillText("on the cluster", W - 6, H - 4); }
  }

  // The neuron's and the cluster's traces on one time axis, each scaled to its own largest value (their units differ), with the peaks the delay is measured between and the cluster's window.
  function drawTraces() {
    const [ctx, W, H] = setup($("dd-traces"));
    if (!current) return;
    const p = current, fps = D.fps, t = p.t.map(f => ms(f / fps)), left = 14, right = 14, top = 16, bottom = 34, h = H - top - bottom;
    const tMin = t[0], tMax = t[t.length - 1], X = v => left + (v - tMin) / (tMax - tMin) * (W - left - right);
    const at = (trace, frame) => { const k = frame - p.t[0], i = Math.max(0, Math.min(trace.length - 2, Math.floor(k))); return trace[i] + (trace[i + 1] - trace[i]) * (k - i); };
    const scaled = [p.neuron_trace, p.trace].map(trace => { const big = Math.max(...trace.map(Math.abs)) || 1; return trace.map(v => v / big); });
    const lo = Math.min(-0.1, ...scaled.flat()), hi = Math.max(0.1, ...scaled.flat()), Y = v => top + (1 - (v - lo) / (hi - lo) * 0.92 - 0.04) * h;
    ctx.fillStyle = Polarity.color(p.sign); ctx.globalAlpha = 0.12; ctx.fillRect(X(ms(p.window[0] / fps)), top, X(ms(p.window[1] / fps)) - X(ms(p.window[0] / fps)), h); ctx.globalAlpha = 1;
    ctx.strokeStyle = "#3b4a5b"; ctx.lineWidth = 1; ctx.strokeRect(left, top, W - left - right, h);
    ctx.beginPath(); ctx.moveTo(left, Y(0)); ctx.lineTo(W - right, Y(0)); ctx.moveTo(X(0), top); ctx.lineTo(X(0), top + h); ctx.stroke();  // zero, and the spike
    const lines = [{trace: p.neuron_trace, shown: scaled[0], color: Polarity.NEURON, name: "neuron dF/F", peak: p.neuron_peak}, {trace: p.trace, shown: scaled[1], color: Polarity.color(current.sign), name: `cluster (${p.sign})`, peak: p.cluster_peak}];
    lines.forEach(l => {
      ctx.strokeStyle = l.color; ctx.lineWidth = 1.8; ctx.beginPath(); l.shown.forEach((v, i) => i ? ctx.lineTo(X(t[i]), Y(v)) : ctx.moveTo(X(t[i]), Y(v))); ctx.stroke(); ctx.lineWidth = 1;
      const big = Math.max(...l.trace.map(Math.abs)) || 1, px = X(ms(l.peak)), py = Y(at(l.trace, l.peak * fps) / big);
      ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, top + h); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = l.color; ctx.beginPath(); ctx.arc(px, py, 5, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = "#0b0f14"; ctx.stroke();
    });
    const a = X(ms(p.neuron_peak)), b = X(ms(p.cluster_peak)), arrowY = top + 10;  // the delay between the two peaks
    ctx.strokeStyle = BRIGHT; ctx.fillStyle = BRIGHT; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(a, arrowY); ctx.lineTo(b, arrowY); ctx.stroke(); ctx.lineWidth = 1;
    ctx.font = "12px sans-serif"; ctx.textBaseline = "middle"; ctx.textAlign = a <= b ? "left" : "right"; ctx.fillText(`delay ${ms(p.delay).toFixed(0)} ms`, (a <= b ? Math.max(a, b) + 8 : Math.min(a, b) - 8), arrowY);
    ctx.font = "11px sans-serif"; ctx.textBaseline = "top"; ctx.textAlign = "left";
    lines.forEach((l, i) => { ctx.fillStyle = l.color; ctx.fillText(l.name, left + 6, top + h - 30 + i * 14); });
    ctx.fillStyle = TEXT; ctx.textAlign = "center";
    TimeAxis.nice(tMin, tMax, 8).forEach(v => ctx.fillText(fmt(v), X(v), top + h + 4));
    ctx.fillText("time from the spike (ms)", left + (W - left - right) / 2, H - 14);
    if (!p.clear) { ctx.textAlign = "right"; ctx.fillStyle = BRIGHT; ctx.fillText("a peak is on the edge of its window", W - right - 4, top + h - 16); }
  }

  function drawDetail() { drawFov($("dd-fov"), false); drawFov($("dd-fov-full"), true); drawTraces(); draw(); }

  // Choose the cluster the picture shows (and its neuron); fills the dropdowns when the neuron changes.
  function pick(p) {
    current = p;
    const cells = [...new Set(usable().map(q => q.cell))].sort((x, y) => x - y);
    $("dd-cell").innerHTML = cells.map(cell => `<option value="${cell}"${cell === p.cell ? " selected" : ""}>${cell}</option>`).join("");
    $("dd-cluster").innerHTML = pointsOf(p.cell).map((q, i) => `<option value="${i}"${q === p ? " selected" : ""}>${label(q)}</option>`).join("");
    $("dd-open").href = `/experiments/${EXP_ID}/cells?cell=${p.cell}&h=${encodeURIComponent(HYPOTHESIS)}`;
    drawDetail();
  }

  function step(by) {
    const cells = [...new Set(usable().map(q => q.cell))].sort((x, y) => x - y), i = cells.indexOf(current.cell);
    pick(pointsOf(cells[(i + by + cells.length) % cells.length])[0]);
  }

  function show() {
    const excluded = D.excluded, notes = [];
    if (excluded.motion.length) notes.push(`${excluded.motion.length} cells with motion suspicion left out`);
    if (excluded.no_neuron_trace.length) notes.push(`${excluded.no_neuron_trace.length} cells without a neuron trace`);
    if (excluded.window_outside_trace.length) notes.push(`${excluded.window_outside_trace.length} cells whose cluster lies outside the saved trace`);
    if (D.n_unclear) notes.push(`${D.n_unclear} clusters without a clear peak`);
    $("dd-summary").textContent = `${D.points.length} clusters of the newest run (${D.run_id}) of this hypothesis${notes.length ? "; " + notes.join("; ") : ""}.`;
    const data = usable();
    if (!data.length) { current = null; $("dd-cell").innerHTML = ""; $("dd-cluster").innerHTML = ""; drawDetail(); return; }
    pick(current && data.includes(current) ? current : data.find(p => current && p.cell === current.cell) || data[0]);
  }

  async function load() {
    const params = new URLSearchParams({h: HYPOTHESIS, clear: $("dd-clear").checked ? "1" : "0"});
    if (types) params.set("types", types.join(","));
    try {
      const res = await fetch(`/experiments/${EXP_ID}/cluster-delays.json?${params}`), body = await res.json();
      if (!res.ok) throw new Error(body.message || `Request failed (${res.status})`);
      if (!body.available) throw new Error(body.message);
      current = null; D = body; show();
    } catch (error) { $("dd-summary").textContent = error.message; }
  }

  async function start() {
    try {  // the result types come from the cluster offsets, which the other panel loads too
      const res = await fetch(`/experiments/${EXP_ID}/cluster-offsets.json?h=${encodeURIComponent(HYPOTHESIS)}`), body = await res.json();
      if (!res.ok || !body.available) throw new Error(body.message || `Request failed (${res.status})`);
      $("dd-measure").innerHTML = Object.entries(MEASURES).map(([key, text]) => `<option value="${key}">${key}: ${text}</option>`).join("");
      $("dd-types").innerHTML = body.result_types.map(name => `<label class="check"><input type="checkbox" value="${name}" checked> ${name}</label>`).join(" ");
      const chosen = () => [...root.querySelectorAll("#dd-types input:checked")].map(box => box.value);
      root.querySelectorAll("#dd-types input").forEach(box => box.addEventListener("change", () => { types = chosen(); load(); }));
      $("dd-clear").addEventListener("change", load);
      $("dd-measure").addEventListener("change", show);
      $("dd-cell").addEventListener("change", () => pick(pointsOf(Number($("dd-cell").value))[0]));
      $("dd-cluster").addEventListener("change", () => pick(pointsOf(current.cell)[Number($("dd-cluster").value)]));
      $("dd-prev").addEventListener("click", () => current && step(-1));
      $("dd-next").addEventListener("click", () => current && step(1));
      window.addEventListener("resize", () => D && drawDetail());
      await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error("The mean astro image could not be loaded: the motion correction step saves it.")); image.src = `/experiments/${EXP_ID}/preprocessing/roi-fov.png?channel=astro`; });
      await load();
    } catch (error) { $("dd-summary").textContent = error.message; }
  }

  const tip = $("dd-tip"), canvas = $("dd-scatter"), near = (e, rect) => pts.map(o => ({o, d: Math.hypot(o.x - (e.clientX - rect.left), o.y - (e.clientY - rect.top))})).filter(h => h.d < 10).sort((a, b) => a.d - b.d)[0];
  canvas.addEventListener("mousemove", e => {
    const hit = near(e, canvas.getBoundingClientRect());
    if (!hit) { tip.style.display = "none"; canvas.style.cursor = "default"; return; }
    const p = hit.o.p; canvas.style.cursor = "pointer";
    tip.innerHTML = `Cell ${p.cell} &middot; ${p.result_type} &middot; ${p.sign}<br>delay ${ms(p.delay).toFixed(0)} ms &middot; ${measure()} ${p.distances[measure()].toFixed(1)} px &middot; p=${p.p == null ? "n/a" : pValue(p.p)} &middot; ${p.n_px} px${p.clear ? "" : " &middot; peak on the edge of its window"}`;
    tip.style.display = "block"; tip.style.left = (e.clientX + 12) + "px"; tip.style.top = (e.clientY + 12) + "px";
  });
  canvas.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  canvas.addEventListener("click", e => { const hit = near(e, canvas.getBoundingClientRect()); if (hit) pick(hit.o.p); });
  start();
})();
