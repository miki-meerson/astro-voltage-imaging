// The cluster map of the Hypotheses tab: a card for every cell with a significant cluster, with the mean astro field of view (the cell's ROI and its clusters outlined) and the
// clusters' mean traces next to the cell's own spike-triggered dF/F. Under each card, for each cluster, the moments the mean is made of: the cluster's pixels around every spike
// of the cell, and the pixels right next to the cluster around the same spikes. Every plot zooms and pans (plot_view.js). What is shown is what the STA job saved
// (cluster_map_summary.json, app/services/cluster_map.py).
(function () {
  const root = document.getElementById("cluster-map-panel");
  if (!root) return;
  const PAGE = JSON.parse(document.getElementById("page-data").textContent);
  const EXP_ID = PAGE.expId, HYPOTHESIS = PAGE.hypothesis;
  const $ = id => document.getElementById(id);
  const YELLOW = "#f7d154", FG = "#c9d4df", GRID = "#1c2630", CONTROL = "#c9d4df", DASH = {all: [], anatomy: [5, 3]};
  const PLOT = {L: 44, R: 44, T: 20, B: 28}, SPIKE_PLOT = {L: 44, R: 10, T: 20, B: 28};
  const image = new Image();
  let D = null;
  const {setup, format: fmt} = PlotCanvas;
  const pct = a => a.map(v => v * 100);
  const span = a => { let lo = Math.min(...a), hi = Math.max(...a); const pad = (hi - lo) * 0.08 || 1; return [lo - pad, hi + pad]; };

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
  function frame(ctx, W, H, xr, yr, a, label, xLabel = "seconds from the spike") {
    const X = v => a.L + (v - xr[0]) / (xr[1] - xr[0]) * (W - a.L - a.R), Y = v => a.T + (1 - (v - yr[0]) / (yr[1] - yr[0])) * (H - a.T - a.B);
    ctx.font = "10px sans-serif"; ctx.textBaseline = "middle"; ctx.textAlign = "right";
    for (const v of TimeAxis.nice(yr[0], yr[1], 4)) { ctx.strokeStyle = GRID; ctx.beginPath(); ctx.moveTo(a.L, Y(v)); ctx.lineTo(W - a.R, Y(v)); ctx.stroke(); ctx.fillStyle = FG; ctx.fillText(fmt(v), a.L - 4, Y(v)); }
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = FG;
    for (const v of TimeAxis.nice(xr[0], xr[1], 6)) ctx.fillText(fmt(v), X(v), H - a.B + 4);
    ctx.fillText(xLabel, a.L + (W - a.L - a.R) / 2, H - 13);
    if (xLabel === "seconds from the spike") { ctx.strokeStyle = "#3b4a5b"; ctx.beginPath(); ctx.moveTo(X(0), a.T); ctx.lineTo(X(0), H - a.B); ctx.moveTo(a.L, Y(0)); ctx.lineTo(W - a.R, Y(0)); ctx.stroke(); }
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
    for (const c of clusters) { const label = `${c.result_type} p=${c.p.toFixed(3)}`; ctx.fillStyle = Polarity.color(c.sign); ctx.fillRect(lx, 8, 12, 3); ctx.fillStyle = FG; ctx.fillText(label, lx + 16, 3); lx += 28 + ctx.measureText(label).width; }
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

  // the whole recording: the mean intensity of a group of pixels in every frame, the spikes of the cell as dashed vertical lines
  function drawFull(canvas, view, trace, spikes, color, title) {
    const [ctx, W, H] = setup(canvas), a = SPIKE_PLOT, t = trace.map((_, k) => k / D.fps);
    const [X, Y] = frame(ctx, W, H, view.x, view.y, a, "intensity (a.u.)", "seconds from the start of the recording");
    clipTo(ctx, W, H, a);
    ctx.strokeStyle = FG; ctx.globalAlpha = 0.45; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    spikes.forEach(s => { ctx.beginPath(); ctx.moveTo(X(s / D.fps), a.T); ctx.lineTo(X(s / D.fps), H - a.B); ctx.stroke(); });
    ctx.setLineDash([]); ctx.globalAlpha = 1;
    line(ctx, t, trace, X, Y, color, 1.1);
    ctx.restore();
    ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.font = "10px sans-serif"; ctx.fillStyle = FG; ctx.fillText(`${title} (dashed: ${spikes.length} spikes)`, a.L + 4, 3);
  }

  function spikeSection(card, c, cell) {
    const box = document.createElement("details");
    box.innerHTML = `<summary>Individual spikes: ${c.result_type} cluster, p=${c.p.toFixed(3)}, ${(c.first_rel / D.fps).toFixed(2)}..${(c.last_rel / D.fps).toFixed(2)} s</summary>
      <div class="cm-spikes-pair"><canvas class="cm-spike-plot"></canvas><canvas class="cm-spike-plot"></canvas></div>
      <div class="cm-full"><canvas class="cm-full-plot"></canvas><canvas class="cm-full-plot"></canvas></div>`;
    card.appendChild(box);
    box.addEventListener("toggle", () => {  // drawn when opened: a closed canvas has no size
      if (!box.open || box.dataset.ready) return;
      box.dataset.ready = "1";
      const [own, control] = box.querySelectorAll("canvas"), t = c.t.map(f => f / D.fps);
      const draw = () => {  // the two plots share one view: zooming one zooms the other, on the same scale
        drawSpikes(own, view, c, c.spike_traces, Polarity.color(c.sign), "cluster pixels");
        drawSpikes(control, view, c, c.control.spike_traces, CONTROL, "pixels right next to it");
      };
      const view = PlotView.create({x: [t[0], t[t.length - 1]], y: span([...c.spike_traces, ...c.control.spike_traces].flatMap(pct))}, draw);
      [own, control].forEach(canvas => PlotView.attach(view, canvas, () => SPIKE_PLOT));
      draw();
      const [fullOwn, fullControl] = box.querySelectorAll(".cm-full-plot"), traces = [c.raw_trace, c.control.raw_trace];
      const drawBoth = () => {
        drawFull(fullOwn, fullViews[0], c.raw_trace, cell.spikes, Polarity.color(c.sign), "cluster pixels, whole recording");
        drawFull(fullControl, fullViews[1], c.control.raw_trace, cell.spikes, CONTROL, "pixels right next to it, whole recording");
      };
      const linkTime = changed => { fullViews.forEach(other => { if (other !== changed) other.x = changed.x.slice(); }); drawBoth(); };  // the two plots show the same stretch of time
      const fullViews = traces.map(trace => PlotView.create({x: [0, (trace.length - 1) / D.fps], y: span(trace)}, linkTime));
      [fullOwn, fullControl].forEach((canvas, i) => PlotView.attach(fullViews[i], canvas, () => SPIKE_PLOT));
      drawBoth();
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
    o.clusters.forEach(c => spikeSection(card, c, o.record));
  }

  function render() {
    const cards = shown();
    $("cm-grid").innerHTML = "";
    $("cm-summary").textContent = `${cards.length} cells, ${cards.reduce((n, o) => n + o.clusters.length, 0)} significant clusters, of the newest run (${D.run_id}) of this hypothesis.`;
    cards.forEach(cardFor);
  }

  async function load() {
    try {
      const res = await fetch(`/experiments/${EXP_ID}/cluster-map.json?h=${encodeURIComponent(HYPOTHESIS)}`), body = await res.json();
      if (!res.ok) throw new Error(body.message || `Request failed (${res.status})`);
      await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error("The mean astro image could not be loaded: the motion correction step saves it.")); image.src = `/experiments/${EXP_ID}/preprocessing/roi-fov.png?channel=astro`; });
      D = body;
      $("cm-types").innerHTML = D.result_types.map(name => `<label class="check"><input type="checkbox" value="${name}" checked> ${name}</label>`).join(" ");
      root.querySelectorAll("input[type=checkbox]").forEach(box => box.addEventListener("change", render));
      render();
    } catch (error) { $("cm-summary").textContent = error.message; }
  }
  load();
})();
