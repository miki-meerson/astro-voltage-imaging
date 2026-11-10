// The cluster map of the Hypotheses tab. Two views of the cells with a significant cluster, chosen in the panel:
//   raw traces: one cell at a time, chosen in a dropdown; a full-width plot per significant cluster of the cell, the mean intensity of the cluster's pixels in every frame of the
//               recording with the spikes of the cell as dashed vertical lines. The height of each plot fits the part of the trace that is shown;
//   cards: a card per cell with the mean astro field of view (the cell's ROI and its clusters outlined), the clusters' mean traces next to the cell's own spike-triggered dF/F,
//          and, for each cluster, the moments the mean is made of (the cluster's pixels around every spike, and the pixels right next to the cluster around the same spikes).
// Every plot zooms and pans (plot_view.js). What is shown is what the STA job saved (cluster_map_summary.json, app/services/cluster_map.py).
(function () {
  const root = document.getElementById("cluster-map-panel");
  if (!root) return;
  const PAGE = JSON.parse(document.getElementById("page-data").textContent);
  const EXP_ID = PAGE.expId, HYPOTHESIS = PAGE.hypothesis;
  const $ = id => document.getElementById(id);
  const YELLOW = "#f7d154", FG = "#c9d4df", GRID = "#1c2630", CONTROL = "#c9d4df", SPIKE_LINE = "rgba(255, 255, 255, 0.85)", DASH = {all: [], anatomy: [5, 3]};
  const PLOT = {L: 44, R: 44, T: 20, B: 28}, SPIKE_PLOT = {L: 44, R: 10, T: 20, B: 28}, RAW_ROW = {L: 64, R: 14, T: 6, B: 6}, RAW_AXIS = {L: 64, R: 14, T: 2, B: 18};
  const image = new Image();
  let D = null, repaintRaw = () => {};  // repaintRaw: draws the rows again (the window changed size)
  const {setup, format: fmt, pValue} = PlotCanvas;
  const pct = a => a.map(v => v * 100);
  const span = a => { let lo = Infinity, hi = -Infinity; for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; } const pad = (hi - lo) * 0.08 || 1; return [lo - pad, hi + pad]; };

  // the field of view: the part of the image in `view` (x and y in image pixels), the cell's ROI and its clusters on it
  function drawFov(canvas, view, cell, clusters) {
    const [ctx, W, H] = setup(canvas), width = image.naturalWidth, scale = W / (view.x[1] - view.x[0]);
    ctx.drawImage(image, view.x[0], view.y[0], view.x[1] - view.x[0], view.y[1] - view.y[0], 0, 0, W, H);
    ctx.save(); ctx.translate(-view.x[0] * scale, -view.y[0] * scale);
    ctx.strokeStyle = YELLOW; ctx.lineWidth = 1.6;
    FovOutline.draw(ctx, cell.roi.ys, cell.roi.xs, width, scale);
    for (const c of clusters) {
      ctx.strokeStyle = Polarity.color(c.sign); ctx.lineWidth = 1.8; ctx.setLineDash(DASH[c.result_type] || []);
      FovOutline.draw(ctx, c.ys, c.xs, width, scale);
      ctx.setLineDash([]); ctx.globalAlpha = 0.7; ctx.lineWidth = 1;  // a ring: the clusters are a few pixels
      ctx.beginPath(); ctx.arc(c.center[1] * scale, c.center[0] * scale, 10, 0, 2 * Math.PI); ctx.stroke(); ctx.globalAlpha = 1;
    }
    ctx.restore();
  }

  // the frame of a plot of traces against the seconds from the spike: the grid, the ticks of the left axis, the time axis and the line at the spike, for the shown ranges
  // `xr` and `yr`. Returns the functions that put a time and a value on the canvas.
  function frame(ctx, W, H, xr, yr, a, label) {
    const X = v => a.L + (v - xr[0]) / (xr[1] - xr[0]) * (W - a.L - a.R), Y = v => a.T + (1 - (v - yr[0]) / (yr[1] - yr[0])) * (H - a.T - a.B);
    ctx.font = "10px sans-serif"; ctx.textBaseline = "middle"; ctx.textAlign = "right";
    for (const v of TimeAxis.nice(yr[0], yr[1], 4)) { ctx.strokeStyle = GRID; ctx.beginPath(); ctx.moveTo(a.L, Y(v)); ctx.lineTo(W - a.R, Y(v)); ctx.stroke(); ctx.fillStyle = FG; ctx.fillText(fmt(v), a.L - 4, Y(v)); }
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = FG;
    for (const v of TimeAxis.nice(xr[0], xr[1], 6)) ctx.fillText(fmt(v), X(v), H - a.B + 4);
    ctx.fillText("seconds from the spike", a.L + (W - a.L - a.R) / 2, H - 13);
    ctx.strokeStyle = "#3b4a5b"; ctx.beginPath(); ctx.moveTo(X(0), a.T); ctx.lineTo(X(0), H - a.B); ctx.moveTo(a.L, Y(0)); ctx.lineTo(W - a.R, Y(0)); ctx.stroke();
    ctx.save(); ctx.translate(11, a.T + (H - a.T - a.B) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillStyle = FG; ctx.fillText(label, 0, 0); ctx.restore();
    return [X, Y];
  }

  const clipTo = (ctx, W, H, a) => { ctx.save(); ctx.beginPath(); ctx.rect(a.L, a.T, W - a.L - a.R, H - a.T - a.B); ctx.clip(); };  // what is drawn next stays inside the plot area

  function line(ctx, t, values, X, Y, color, width, dash) {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.beginPath();
    values.forEach((v, k) => k ? ctx.lineTo(X(t[k]), Y(v)) : ctx.moveTo(X(t[k]), Y(v))); ctx.stroke(); ctx.setLineDash([]);
  }

  function band(ctx, c, X, a, H) {
    const x0 = X((c.first_rel - 0.5) / D.fps), x1 = X((c.last_rel + 0.5) / D.fps);
    ctx.globalAlpha = 0.14; ctx.fillStyle = Polarity.color(c.sign); ctx.fillRect(x0, a.T, x1 - x0, H - a.T - a.B); ctx.globalAlpha = 1;
  }

  // the clusters' mean traces (left axis) and the neuron's own trace (right axis, held to the same part of the plot as the left one)
  function drawTrace(canvas, view, cell, clusters) {
    const [ctx, W, H] = setup(canvas), a = PLOT, t = clusters[0].t.map(f => f / D.fps), neuron = pct(cell.neuron_trace);
    const [X, Y] = frame(ctx, W, H, view.x, view.y, a, `astro, dF/F (%), ${Polarity.AXIS_NOTE}`);
    const shownAs = [(view.y[0] - view.base.y[0]) / (view.base.y[1] - view.base.y[0]), (view.y[1] - view.base.y[0]) / (view.base.y[1] - view.base.y[0])];  // the shown part of the left axis
    const nr = neuron.length ? span(neuron) : [0, 1], neuronRange = [nr[0] + shownAs[0] * (nr[1] - nr[0]), nr[0] + shownAs[1] * (nr[1] - nr[0])];
    const YN = v => a.T + (1 - (v - neuronRange[0]) / (neuronRange[1] - neuronRange[0])) * (H - a.T - a.B);
    if (neuron.length) { ctx.font = "10px sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillStyle = YELLOW; for (const v of TimeAxis.nice(neuronRange[0], neuronRange[1], 4)) ctx.fillText(fmt(v), W - a.R + 4, YN(v)); }
    clipTo(ctx, W, H, a);
    clusters.forEach(c => band(ctx, c, X, a, H));
    clusters.forEach(c => line(ctx, t, pct(c.trace), X, Y, Polarity.color(c.sign), 1.3, DASH[c.result_type]));
    if (neuron.length) line(ctx, t, neuron, X, YN, YELLOW, 1.8);
    ctx.restore();
    if (neuron.length) { ctx.save(); ctx.translate(W - 8, a.T + (H - a.T - a.B) / 2); ctx.rotate(Math.PI / 2); ctx.textAlign = "center"; ctx.fillStyle = YELLOW; ctx.fillText("neuron dF/F (%)", 0, 0); ctx.restore(); }
    ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.font = "10px sans-serif"; let lx = a.L + 4;
    for (const c of clusters) { const label = `${c.result_type} p=${pValue(c.p)}`; ctx.fillStyle = Polarity.color(c.sign); ctx.fillRect(lx, 8, 12, 3); ctx.fillStyle = FG; ctx.fillText(label, lx + 16, 3); lx += 28 + ctx.measureText(label).width; }
  }

  // one group of pixels around every spike: the individual spikes thin, their mean bold
  function drawSpikes(canvas, view, c, traces, color, title) {
    const [ctx, W, H] = setup(canvas), a = SPIKE_PLOT, t = c.t.map(f => f / D.fps);
    const rows = traces.map(pct), mean = t.map((_, k) => rows.reduce((s, r) => s + r[k], 0) / rows.length);
    const [X, Y] = frame(ctx, W, H, view.x, view.y, a, "dF/F (%)");
    clipTo(ctx, W, H, a);
    band(ctx, c, X, a, H);
    ctx.globalAlpha = 0.28; rows.forEach(r => line(ctx, t, r, X, Y, color, 0.8)); ctx.globalAlpha = 1;
    line(ctx, t, mean, X, Y, color, 2.6);
    ctx.restore();
    ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.font = "10px sans-serif"; ctx.fillStyle = FG; ctx.fillText(`${title} (${rows.length} spikes, ${c.n_px} px)`, a.L + 4, 3);
  }

  function spikeSection(card, c) {
    const box = document.createElement("details");
    box.innerHTML = `<summary>Individual spikes: ${c.result_type} cluster, p=${pValue(c.p)}, ${(c.first_rel / D.fps).toFixed(2)}..${(c.last_rel / D.fps).toFixed(2)} s</summary>
      <div class="cm-spikes-pair"><canvas class="cm-spike-plot"></canvas><canvas class="cm-spike-plot"></canvas></div>`;
    card.appendChild(box);
    box.addEventListener("toggle", () => {  // drawn when opened: a closed canvas has no size
      if (!box.open || box.dataset.ready) return;
      box.dataset.ready = "1";
      const [own, control] = box.querySelectorAll("canvas"), t = c.t.map(f => f / D.fps);
      const draw = () => {  // the two plots share one view: zooming one zooms the other, on the same scale
        drawSpikes(own, view, c, c.spike_traces, Polarity.color(c.sign), "cluster pixels");
        drawSpikes(control, view, c, c.control.spike_traces, CONTROL, "pixels right next to it");
      };
      const view = PlotView.create({x: [t[0], t[t.length - 1]], y: span(c.spike_traces.concat(c.control.spike_traces).flatMap(pct))}, draw);
      [own, control].forEach(canvas => PlotView.attach(view, canvas, () => SPIKE_PLOT));
      draw();
    });
  }

  function shown() {  // the cells to draw, with their clusters of the checked result types
    const types = [...document.querySelectorAll("#cm-types input:checked")].map(b => b.value), motion = $("cm-motion").checked;
    return Object.keys(D.cells).map(Number).sort((a, b) => a - b).map(cell => ({cell, record: D.cells[cell], clusters: D.clusters.filter(c => c.cell === cell && types.includes(c.result_type))}))
      .filter(o => o.clusters.length && (motion || o.record.motion !== "motion_suspected"));
  }

  function cardFor(o) {
    const types = [...new Set(o.clusters.map(c => c.result_type))].join(" + "), flagged = o.record.motion === "motion_suspected";
    const card = document.createElement("div"); card.className = "cm-card";
    card.innerHTML = `<div class="cm-head"><a href="/experiments/${EXP_ID}/cells?cell=${o.cell}&h=${encodeURIComponent(HYPOTHESIS)}" title="Open the cell"><b>Cell ${o.cell}</b></a><span class="muted">${types}${flagged ? " &middot; motion suspected" : ""} &middot; ${o.record.n_spikes} spikes</span></div>
      <div class="cm-pair"><canvas class="cm-fov"></canvas><canvas class="cm-trace"></canvas></div>`;
    $("cm-grid").appendChild(card);
    const fov = card.querySelector(".cm-fov"), trace = card.querySelector(".cm-trace"), t = o.clusters[0].t.map(f => f / D.fps);
    fov.style.aspectRatio = `${image.naturalWidth} / ${image.naturalHeight}`;
    const fovView = PlotView.create({x: [0, image.naturalWidth], y: [0, image.naturalHeight]}, () => drawFov(fov, fovView, o.record, o.clusters));
    PlotView.attach(fovView, fov, () => ({L: 0, R: 0, T: 0, B: 0}), {inside: true, yDown: true});
    const traceView = PlotView.create({x: [t[0], t[t.length - 1]], y: span(o.clusters.flatMap(c => pct(c.trace)))}, () => drawTrace(trace, traceView, o.record, o.clusters));
    PlotView.attach(traceView, trace, () => PLOT);
    drawFov(fov, fovView, o.record, o.clusters); drawTrace(trace, traceView, o.record, o.clusters);
    o.clusters.forEach(c => spikeSection(card, c));
  }

  // the time axis above the rows: the same stretch of time as every row
  function drawRawAxis(canvas, xr) {
    const [ctx, W, H] = setup(canvas), a = RAW_AXIS, X = v => a.L + (v - xr[0]) / (xr[1] - xr[0]) * (W - a.L - a.R);
    ctx.font = "10px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = FG;
    for (const v of TimeAxis.nice(xr[0], xr[1], 12)) { ctx.fillText(fmt(v), X(v), 2); ctx.strokeStyle = "#3b4a5b"; ctx.beginPath(); ctx.moveTo(X(v), H - 4); ctx.lineTo(X(v), H); ctx.stroke(); }
    ctx.textAlign = "left"; ctx.fillText("s", 4, 2);
  }

  // the path of a trace over the shown stretch of time with at most two points per pixel column (the lowest and the highest value in the column): the same picture as
  // every frame of a long recording, much faster to draw
  function tracePath(ctx, values, X, Y, xr, columns) {
    const first = Math.max(0, Math.floor(xr[0] * D.fps) - 1), last = Math.min(values.length - 1, Math.ceil(xr[1] * D.fps) + 1), n = last - first + 1;
    ctx.beginPath();
    if (n <= 2 * columns) { for (let k = first; k <= last; k++) k === first ? ctx.moveTo(X(k / D.fps), Y(values[k])) : ctx.lineTo(X(k / D.fps), Y(values[k])); return; }
    for (let c = 0; c < columns; c++) {
      const from = first + Math.floor(c * n / columns), to = Math.min(last, first + Math.floor((c + 1) * n / columns));
      let lo = Infinity, hi = -Infinity;
      for (let k = from; k <= to; k++) { const v = values[k]; if (v < lo) lo = v; if (v > hi) hi = v; }
      const x = X((from + to) / 2 / D.fps);
      c ? ctx.lineTo(x, Y(lo)) : ctx.moveTo(x, Y(lo));
      ctx.lineTo(x, Y(hi));
    }
  }

  // the height that fits the frames of `values` inside the time range `xr` (seconds)
  function fitRange(values, xr) {
    const first = Math.max(0, Math.floor(xr[0] * D.fps)), last = Math.min(values.length - 1, Math.ceil(xr[1] * D.fps));
    return span(values.slice(first, last + 1));
  }

  // a cluster's plot: the mean intensity of its pixels in every frame, the spikes of the cell as dashed vertical lines drawn over the trace (the trace is dense)
  function drawRawRow(canvas, view, o, c) {
    const [ctx, W, H] = setup(canvas), a = RAW_ROW, X = v => a.L + (v - view.x[0]) / (view.x[1] - view.x[0]) * (W - a.L - a.R), Y = v => a.T + (1 - (v - view.y[0]) / (view.y[1] - view.y[0])) * (H - a.T - a.B);
    ctx.font = "10px sans-serif"; ctx.textBaseline = "middle"; ctx.textAlign = "right";
    for (const v of TimeAxis.nice(view.y[0], view.y[1], 5)) { ctx.strokeStyle = GRID; ctx.beginPath(); ctx.moveTo(a.L, Y(v)); ctx.lineTo(W - a.R, Y(v)); ctx.stroke(); ctx.fillStyle = FG; ctx.fillText(fmt(v), a.L - 4, Y(v)); }
    clipTo(ctx, W, H, a);
    (D.excluded_segments || []).forEach(([from, to]) => { ctx.fillStyle = "rgba(240, 180, 60, 0.16)"; ctx.fillRect(X(from / D.fps), a.T, X(to / D.fps) - X(from / D.fps), H - a.T - a.B); });  // left out of the analysis
    ctx.strokeStyle = Polarity.color(c.sign); ctx.lineWidth = 1.3; ctx.setLineDash(DASH[c.result_type] || []);
    tracePath(ctx, c.raw_trace, X, Y, view.x, W - a.L - a.R);
    ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = SPIKE_LINE; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    o.record.spikes.forEach(s => { ctx.beginPath(); ctx.moveTo(X(s / D.fps), a.T); ctx.lineTo(X(s / D.fps), H - a.B); ctx.stroke(); });
    ctx.setLineDash([]);
    ctx.restore();
    const flagged = o.record.motion === "motion_suspected";
    const label = `Cell ${o.cell} · ${c.result_type} cluster, p=${pValue(c.p)}${flagged ? " · motion suspected" : ""} · ${o.record.spikes.length} spikes`;
    ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillStyle = "rgba(11, 15, 20, 0.7)"; ctx.fillRect(a.L + 2, a.T + 1, ctx.measureText(label).width + 8, 13); ctx.fillStyle = FG; ctx.fillText(label, a.L + 6, a.T + 3);
  }

  // the plots of one cell, one per cluster: one time axis for all of them (zooming or panning the time in one moves every plot and fits each height to what is shown again)
  function renderRaw(o) {
    const holder = $("cm-raw");
    holder.innerHTML = "";
    if (!o) return;
    const seconds = (o.clusters[0].raw_trace.length - 1) / D.fps, axis = document.createElement("canvas"), rows = [];
    axis.className = "cm-raw-axis"; holder.appendChild(axis);
    let waiting = false;
    const paint = () => {  // once per frame, however many events came
      waiting = false;
      drawRawAxis(axis, rows[0].view.x);
      rows.forEach(row => drawRawRow(row.canvas, row.view, o, row.cluster));
    };
    const schedule = () => { if (!waiting) { waiting = true; requestAnimationFrame(paint); } };
    const redraw = changed => {
      rows.forEach(row => {
        if (row.view !== changed) row.view.x = changed.x.slice();
        if (row.view.x[0] !== row.fitted[0] || row.view.x[1] !== row.fitted[1]) {  // the time changed: the height fits the frames now shown; a change of the height alone is kept
          row.view.y = fitRange(row.cluster.raw_trace, row.view.x);
          row.fitted = row.view.x.slice();
        }
      });
      schedule();
    };
    o.clusters.forEach(cluster => {
      const canvas = document.createElement("canvas"), view = PlotView.create({x: [0, seconds], y: fitRange(cluster.raw_trace, [0, seconds])}, redraw);
      canvas.className = "cm-raw-row"; holder.appendChild(canvas);
      PlotView.attach(view, canvas, () => RAW_ROW);
      rows.push({canvas, view, cluster, fitted: view.x.slice()});
    });
    repaintRaw = schedule;
    schedule();
  }

  function render() {
    const cards = shown(), raw = $("cm-view").value === "raw";
    $("cm-raw").hidden = !raw; $("cm-grid").hidden = raw; $("cm-cell-field").hidden = !raw;
    const chosen = $("cm-cell").value;  // the dropdown lists the cells that are shown; the one that was open stays open when it still is
    $("cm-cell").innerHTML = cards.map(o => `<option value="${o.cell}">Cell ${o.cell} · ${[...new Set(o.clusters.map(c => c.result_type))].join(" + ")} · ${o.record.spikes.length} spikes${o.record.motion === "motion_suspected" ? " · motion suspected" : ""}</option>`).join("");
    if (cards.some(o => String(o.cell) === chosen)) $("cm-cell").value = chosen;
    $("cm-summary").textContent = `${cards.length} cells, ${cards.reduce((n, o) => n + o.clusters.length, 0)} significant clusters, of the newest run (${D.run_id}) of this hypothesis.`;
    $("cm-grid").innerHTML = "";
    if (raw) renderRaw(cards.find(o => String(o.cell) === $("cm-cell").value)); else cards.forEach(cardFor);
  }

  // the cards of the listed cells on paper: the print dialog (choose Save as PDF) shows only this panel's cards (print rules in pages.css); the view that was open comes back afterwards
  function exportCardsToPdf() {
    const view = $("cm-view"), before = view.value;
    if (before !== "cards") { view.value = "cards"; render(); }
    window.addEventListener("afterprint", () => { if (view.value !== before) { view.value = before; render(); } }, {once: true});
    requestAnimationFrame(() => window.print());
  }

  async function load() {
    try {
      const res = await fetch(`/experiments/${EXP_ID}/cluster-map.json?h=${encodeURIComponent(HYPOTHESIS)}`), body = await res.json();
      if (!res.ok) throw new Error(body.message || `Request failed (${res.status})`);
      await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error("The mean astro image could not be loaded: the motion correction step saves it.")); image.src = `/experiments/${EXP_ID}/preprocessing/roi-fov.png?channel=astro`; });
      D = body;
      $("cm-types").innerHTML = D.result_types.map(name => `<label class="check"><input type="checkbox" value="${name}" checked> ${name}</label>`).join(" ");
      root.querySelectorAll("input[type=checkbox]").forEach(box => box.addEventListener("change", render));
      if (new URLSearchParams(location.search).get("view") === "cards") $("cm-view").value = "cards";  // a link to the cards view
      $("cm-view").addEventListener("change", render);
      $("cm-cell").addEventListener("change", render);
      $("cm-pdf").addEventListener("click", exportCardsToPdf);
      render();
    } catch (error) { $("cm-summary").textContent = error.message; }
  }
  window.addEventListener("resize", () => repaintRaw());
  load();
})();
