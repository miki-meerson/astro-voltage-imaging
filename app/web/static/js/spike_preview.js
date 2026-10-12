(() => {
  const NEURON_ID = JSON.parse(document.getElementById("page-data").textContent).neuronId;
  const canvas = document.getElementById(`spike-preview-${NEURON_ID}`);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const plot = {left: 76, right: 22, top: 28, bottom: 62};
  const colors = {simple: "#4fa3ff", complex: "#f0a33a", exclude: "#e05260"};
  let trace = [], peaks = [], labels = [], fps = 1, start = 0, stop = 0;

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.max(300, Math.round(rect.width * dpr));
    canvas.height = Math.max(120, Math.round(rect.height * dpr));
  }
  function plotWidth() { return canvas.width - plot.left - plot.right; }
  function plotHeight() { return canvas.height - plot.top - plot.bottom; }
  function frameToX(f) { return plot.left + (f - start) / Math.max(1, stop - start) * plotWidth(); }
  function valToY(v, lo, hi) { return plot.top + (1 - (v - lo) / Math.max(1e-9, hi - lo)) * plotHeight(); }
  function fmtTime(frame) {
    const seconds = frame / Math.max(1e-9, fps);
    return seconds >= 60 ? `${(seconds / 60).toFixed(1)} min` : `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  }
  function fmtDff(v) { return Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2); }
  function niceTicks(lo, hi, n = 5) {
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return [lo, hi];
    const raw = (hi - lo) / Math.max(1, n - 1);
    const pow = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map(x => x * pow).find(x => x >= raw) || raw;
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 0.5; v += step) ticks.push(v);
    return ticks;
  }
  function range() {
    const values = trace.slice(start, stop).filter(Number.isFinite);
    let lo = Math.min(...values), hi = Math.max(...values);
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return [0, 1];
    const pad = Math.max((hi - lo) * 0.14, 0.02);
    return [lo - pad, hi + pad];
  }
  function drawMessage(message) {
    resize();
    ctx.fillStyle = "#0b0f14";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#9aa8b7";
    ctx.font = "18px Segoe UI";
    ctx.textAlign = "center";
    ctx.fillText(message, canvas.width / 2, canvas.height / 2);
  }
  function draw() {
    resize();
    if (!trace.length) return drawMessage("No spike trace available.");
    const [lo, hi] = range();
    ctx.fillStyle = "#0b0f14";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#43505f";
    ctx.fillStyle = "#aeb9c6";
    ctx.lineWidth = 1;
    ctx.font = "16px Segoe UI";
    ctx.beginPath();
    ctx.moveTo(plot.left, plot.top);
    ctx.lineTo(plot.left, plot.top + plotHeight());
    ctx.lineTo(plot.left + plotWidth(), plot.top + plotHeight());
    ctx.stroke();
    for (const v of niceTicks(lo, hi, 5)) {
      const y = valToY(v, lo, hi);
      ctx.strokeStyle = "#222c36";
      ctx.beginPath(); ctx.moveTo(plot.left, y); ctx.lineTo(plot.left + plotWidth(), y); ctx.stroke();
      ctx.fillStyle = "#aeb9c6";
      ctx.textAlign = "right";
      ctx.fillText(fmtDff(v), plot.left - 10, y);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i < 5; i++) {
      const f = Math.round(start + i / 4 * (stop - start));
      ctx.fillText(fmtTime(f), frameToX(f), plot.top + plotHeight() + 12);
    }
    ctx.fillStyle = "#d6dde6";
    ctx.fillText("Time", plot.left + plotWidth() / 2, canvas.height - 18);
    ctx.save(); ctx.translate(22, plot.top + plotHeight() / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("dF/F", 0, 0); ctx.restore();
    ctx.save();
    ctx.beginPath(); ctx.rect(plot.left, plot.top, plotWidth(), plotHeight()); ctx.clip();
    const grad = ctx.createLinearGradient(plot.left, 0, plot.left + plotWidth(), 0);
    grad.addColorStop(0, "#8cc7ff"); grad.addColorStop(0.55, "#d6dde6"); grad.addColorStop(1, "#37c6a6");
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let moved = false;
    const step = Math.max(1, Math.ceil((stop - start) / Math.max(1, plotWidth() * 1.5)));
    for (let f = start; f < stop && f < trace.length; f += step) {
      if (!Number.isFinite(trace[f])) continue;
      const x = frameToX(f), y = valToY(trace[f], lo, hi);
      if (!moved) { ctx.moveTo(x, y); moved = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
    for (let i = 0; i < peaks.length; i++) {
      const f = Number(peaks[i]);
      if (f < start || f >= stop || !Number.isFinite(trace[f])) continue;
      ctx.fillStyle = colors[labels[i]] || "#d6dde6";
      ctx.beginPath(); ctx.arc(frameToX(f), valToY(trace[f], lo, hi), 5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
  }
  fetch(`/neurons/${NEURON_ID}/spikes/data`)
    .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
    .then(data => {
      if (data.error) throw new Error(data.error);
      trace = data.trace || [];
      peaks = data.peaks || [];
      labels = data.labels || [];
      fps = Number(data.fps || 1);
      start = 0;
      stop = trace.length;
      draw();
      new ResizeObserver(draw).observe(canvas);
      window.addEventListener("resize", draw);
    })
    .catch(error => drawMessage(error.message || "Spike plot failed to load."));
})();
