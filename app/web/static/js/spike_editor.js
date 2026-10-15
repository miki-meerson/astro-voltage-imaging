(function () {
  const PAGE_DATA = JSON.parse(document.getElementById("page-data").textContent);
  const NEURON_ID = PAGE_DATA.neuronId;
  const canvas = document.getElementById("spike-canvas");
  const ctx = canvas.getContext("2d");
  let trace = [], peaks = [], labels = [], selected = -1, fps = 1;
  let start = 0, stop = 1000;
  let initialDetectionParams = {}, detectTimer = null, detectRequestId = 0;
  const colors = {spike: "#4fa3ff", exclude: "#e05260"};
  const MIN_SPAN = 20;  // frames: the most zoomed-in view
  const plot = {left: 76, right: 22, top: 28, bottom: 62};
  // the detector settings (and their defaults) come from the server with the cell's data; nothing is defined here
  const paramFormat = {
    mad_factor: v => Number(v).toFixed(1),
  };

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const w = Math.max(900, Math.round(rect.width * dpr));
    const h = Math.max(420, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }
  function yRange() {
    const part = trace.slice(start, stop).filter(Number.isFinite);
    const peakVals = peaks
      .map(f => Number(f))
      .filter(f => f >= start && f < stop && Number.isFinite(trace[f]))
      .map(f => trace[f]);
    const values = part.concat(peakVals);
    let lo = Math.min(...values), hi = Math.max(...values);
    if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return [0, 1];
    const pad = Math.max((hi - lo) * 0.14, 0.02);
    return [lo - pad, hi + pad];
  }
  function plotWidth() { return canvas.width - plot.left - plot.right; }
  function plotHeight() { return canvas.height - plot.top - plot.bottom; }
  function xToFrame(x) {
    const frac = Math.max(0, Math.min(1, (x - plot.left) / Math.max(1, plotWidth())));
    return Math.round(start + frac * (stop - start));
  }
  function frameToX(f) {
    return plot.left + (f - start) / Math.max(1, stop - start) * plotWidth();
  }
  function valToY(v, lo, hi) {
    return plot.top + (1 - (v - lo) / Math.max(1e-9, hi - lo)) * plotHeight();
  }
  function fmtTime(frame) {
    const seconds = frame / Math.max(1e-9, fps);
    if (seconds >= 60) return `${(seconds / 60).toFixed(1)} min`;
    return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  }
  function fmtDff(v) {
    return Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2);
  }
  function niceTicks(lo, hi, n = 5) {
    const ticks = [];
    if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return [lo, hi];
    const raw = (hi - lo) / Math.max(1, n - 1);
    const pow = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map(x => x * pow).find(x => x >= raw) || raw;
    const first = Math.ceil(lo / step) * step;
    for (let v = first; v <= hi + step * 0.5; v += step) ticks.push(v);
    return ticks;
  }
  function drawAxes(lo, hi) {
    ctx.save();
    ctx.strokeStyle = "#43505f";
    ctx.fillStyle = "#aeb9c6";
    ctx.lineWidth = 1;
    ctx.font = "16px Segoe UI";
    ctx.textBaseline = "middle";
    ctx.beginPath();
    ctx.moveTo(plot.left, plot.top);
    ctx.lineTo(plot.left, plot.top + plotHeight());
    ctx.lineTo(plot.left + plotWidth(), plot.top + plotHeight());
    ctx.stroke();

    for (const v of niceTicks(lo, hi, 5)) {
      const y = valToY(v, lo, hi);
      ctx.strokeStyle = "#222c36";
      ctx.beginPath();
      ctx.moveTo(plot.left, y);
      ctx.lineTo(plot.left + plotWidth(), y);
      ctx.stroke();
      ctx.strokeStyle = "#43505f";
      ctx.beginPath();
      ctx.moveTo(plot.left - 5, y);
      ctx.lineTo(plot.left, y);
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(fmtDff(v), plot.left - 10, y);
    }

    const xTicks = 5;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i < xTicks; i++) {
      const f = Math.round(start + i / (xTicks - 1) * (stop - start));
      const x = frameToX(f);
      ctx.strokeStyle = "#43505f";
      ctx.beginPath();
      ctx.moveTo(x, plot.top + plotHeight());
      ctx.lineTo(x, plot.top + plotHeight() + 5);
      ctx.stroke();
      ctx.fillText(fmtTime(f), x, plot.top + plotHeight() + 12);
    }

    ctx.fillStyle = "#d6dde6";
    ctx.textAlign = "center";
    ctx.fillText("Time", plot.left + plotWidth() / 2, canvas.height - 18);
    ctx.save();
    ctx.translate(22, plot.top + plotHeight() / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("dF/F", 0, 0);
    ctx.restore();
    ctx.restore();
  }
  function draw() {
    resizeCanvas();
    ctx.fillStyle = "#0b0f14";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const [lo, hi] = yRange();
    drawAxes(lo, hi);
    ctx.save();
    ctx.beginPath();
    ctx.rect(plot.left, plot.top, plotWidth(), plotHeight());
    ctx.clip();
    const grad = ctx.createLinearGradient(plot.left, 0, plot.left + plotWidth(), 0);
    grad.addColorStop(0, "#8cc7ff");
    grad.addColorStop(0.55, "#d6dde6");
    grad.addColorStop(1, "#37c6a6");
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let moved = false;
    for (let f = start; f < stop && f < trace.length; f++) {
      if (!Number.isFinite(trace[f])) continue;
      const x = frameToX(f), y = valToY(trace[f], lo, hi);
      if (!moved) { ctx.moveTo(x, y); moved = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
    for (let i = 0; i < peaks.length; i++) {
      const f = Number(peaks[i]);
      if (f < start || f >= stop) continue;
      const value = trace[f];
      if (!isFinite(value)) continue;
      const x = frameToX(f), y = valToY(value, lo, hi);
      ctx.fillStyle = colors[labels[i]] || "#d6dde6";
      ctx.beginPath();
      ctx.arc(x, y, selected === i ? 8 : 5, 0, Math.PI * 2);
      ctx.fill();
      if (selected === i) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
    ctx.restore();
    ctx.fillStyle = "#d6dde6";
    ctx.font = "18px Segoe UI";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    if (selection) {
      ctx.fillStyle = "rgba(74, 163, 255, 0.22)";
      ctx.strokeStyle = "rgba(74, 163, 255, 0.9)";
      ctx.lineWidth = 1;
      const x0 = Math.min(selection.x0, selection.x1), w = Math.abs(selection.x1 - selection.x0);
      ctx.fillRect(x0, plot.top, w, plotHeight());
      ctx.strokeRect(x0, plot.top, w, plotHeight());
    }
    ctx.fillStyle = "#d6dde6";
    ctx.fillText(`Frames ${start} - ${stop}  (${fmtTime(start)} - ${fmtTime(stop)})`, plot.left, 22);
    const legendY = 24;
    let lx = canvas.width - 220;
    for (const item of [["spike", colors.spike], ["excluded", colors.exclude]]) {
      ctx.fillStyle = item[1];
      ctx.beginPath();
      ctx.arc(lx, legendY - 5, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#d6dde6";
      ctx.fillText(item[0], lx + 12, legendY);
      lx += 96;
    }
  }
  function updateSelected() {
    const el = document.getElementById("selected-info");
    el.textContent = selected >= 0 ? `Frame ${peaks[selected]} / ${labels[selected]}` : "No Spike Selected";
  }
  function updateDetectionStats() {
    const excluded = labels.filter(label => label === "exclude").length;
    document.getElementById("detection-stats").textContent = `${peaks.length} spikes (${excluded} excluded)`;
  }
  function getDetectionParams() {
    const params = {};
    document.querySelectorAll("[data-detect-param]").forEach(input => {
      const key = input.dataset.detectParam;
      params[key] = Number(input.value);
    });
    return params;
  }
  function setDetectionParams(params) {
    document.querySelectorAll("[data-detect-param]").forEach(input => {
      const key = input.dataset.detectParam;
      input.value = params[key];
      updateParamOutput(input);
    });
  }
  function updateParamOutput(input) {
    const output = document.getElementById(`${input.id}-value`);
    const formatter = paramFormat[input.dataset.detectParam] || (v => String(v));
    if (output) output.value = formatter(input.value);
  }
  function scheduleDetect(delay = 180) {
    window.clearTimeout(detectTimer);
    document.getElementById("detection-status").textContent = "Updating detection preview...";
    detectTimer = window.setTimeout(runDetection, delay);
  }
  async function runDetection() {
    const requestId = ++detectRequestId;
    const status = document.getElementById("detection-status");
    status.textContent = "Detecting...";
    try {
      const res = await fetch(`/neurons/${NEURON_ID}/spikes/detect`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(getDetectionParams()),
      });
      const data = await res.json();
      if (requestId !== detectRequestId) return;
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
      peaks = (data.peaks || []).map(Number);
      labels = (data.labels || []).map(String);
      selected = -1;
      updateSelected();
      updateDetectionStats();
      draw();
      status.textContent = "Preview updated. Save Edits to keep these peaks.";
    } catch (error) {
      if (requestId === detectRequestId) status.textContent = error.message || "Detection failed.";
    }
  }
  function addPeak(frame, label) {
    frame = Math.max(0, Math.min(trace.length - 1, Number(frame)));
    const idx = peaks.findIndex(p => Math.abs(p - frame) <= 1);
    if (idx >= 0) { selected = idx; labels[idx] = label; }
    else { peaks.push(frame); labels.push(label); selected = peaks.length - 1; }
    updateSelected(); updateDetectionStats(); draw();
  }
  function canvasPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return {x: (event.clientX - rect.left) * canvas.width / rect.width, y: (event.clientY - rect.top) * canvas.height / rect.height};
  }
  function insidePlot(p) {
    return p.x >= plot.left && p.x <= plot.left + plotWidth() && p.y >= plot.top && p.y <= plot.top + plotHeight();
  }
  // the view is `span` frames from `first`, kept inside the trace
  function setView(first, span) {
    span = Math.max(Math.min(MIN_SPAN, trace.length), Math.min(trace.length, Math.round(span)));
    start = Math.max(0, Math.min(trace.length - span, Math.round(first)));
    stop = start + span;
    draw();
  }
  // zoom so that the frame under the cursor stays under the cursor
  function zoomAt(frame, factor) {
    const span = stop - start, newSpan = span * factor;
    setView(frame - (frame - start) / span * newSpan, newSpan);
  }
  const ZOOM_STEP = 1.25;
  function zoomIn() { zoomAt((start + stop) / 2, 1 / ZOOM_STEP); }
  function zoomOut() { zoomAt((start + stop) / 2, ZOOM_STEP); }
  function fitAll() { setView(0, trace.length); }
  function applyLabel(label) {
    if (selected < 0) return;
    if (label === "delete") { peaks.splice(selected, 1); labels.splice(selected, 1); selected = -1; }
    else labels[selected] = label;
    updateSelected(); updateDetectionStats(); draw();
  }

  canvas.addEventListener("wheel", (event) => {
    const p = canvasPoint(event);
    if (!insidePlot(p)) return;
    event.preventDefault();
    zoomAt(xToFrame(p.x), event.deltaY < 0 ? 1 / ZOOM_STEP : ZOOM_STEP);
  }, {passive: false});

  // drag across the trace = mark the section to zoom into; shift + drag = move the view; a click that did not move selects the nearest spike;
  // ctrl + click adds a spike; a double-click shows the whole trace again
  let drag = null, dragged = false, selection = null;
  canvas.addEventListener("mousedown", (event) => {
    const p = canvasPoint(event);
    dragged = false;
    if (event.button === 0 && insidePlot(p)) drag = {x: p.x, first: start, span: stop - start, pan: event.shiftKey};
  });
  window.addEventListener("mousemove", (event) => {
    if (!drag) return;
    const x = Math.max(plot.left, Math.min(plot.left + plotWidth(), canvasPoint(event).x));
    if (!dragged && Math.abs(x - drag.x) <= 4) return;
    dragged = true;
    if (drag.pan) {
      canvas.style.cursor = "grabbing";
      setView(drag.first - (x - drag.x) / plotWidth() * drag.span, drag.span);
    } else {
      selection = {x0: drag.x, x1: x};
      draw();
    }
  });
  window.addEventListener("mouseup", () => {
    if (drag && selection) {
      const first = xToFrame(Math.min(selection.x0, selection.x1)), last = xToFrame(Math.max(selection.x0, selection.x1));
      selection = null;
      setView(first, last - first);
    }
    drag = null;
    canvas.style.cursor = "";
  });
  canvas.addEventListener("click", (event) => {
    if (dragged) { dragged = false; return; }
    const p = canvasPoint(event);
    if (!insidePlot(p)) return;
    const frame = xToFrame(p.x);
    if (event.ctrlKey || event.metaKey) { addPeak(frame, "spike"); return; }
    let best = -1, dist = Infinity;
    for (let i = 0; i < peaks.length; i++) {
      const d = Math.abs(peaks[i] - frame);
      if (d < dist) { dist = d; best = i; }
    }
    selected = dist <= Math.max(4, (stop - start) * 0.01) ? best : -1;
    updateSelected(); draw();
  });
  canvas.addEventListener("dblclick", (event) => {
    if (insidePlot(canvasPoint(event))) fitAll();
  });
  document.querySelectorAll("[data-label]").forEach(btn => btn.addEventListener("click", () => applyLabel(btn.dataset.label)));
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, select, textarea")) return;
    const span = stop - start;
    if (event.key === "ArrowLeft") setView(start - span * 0.2, span);
    else if (event.key === "ArrowRight") setView(start + span * 0.2, span);
    else if (event.key === "+" || event.key === "=") zoomIn();
    else if (event.key === "-") zoomOut();
    else if (event.key === "Home") fitAll();
    else if (event.key === "Delete" || event.key === "Backspace") applyLabel("delete");
    else if (event.key === "x") applyLabel("exclude");
    else return;
    event.preventDefault();
  });
  document.getElementById("add-spike").onclick = () => addPeak(document.getElementById("frame-input").value, "spike");
  document.querySelectorAll("[data-detect-param]").forEach(input => {
    input.addEventListener("input", () => {
      updateParamOutput(input);
      scheduleDetect();
    });
    input.addEventListener("change", () => scheduleDetect(0));
  });
  document.getElementById("redetect-now").onclick = () => {
    window.clearTimeout(detectTimer);
    runDetection();
  };
  document.getElementById("reset-detection").onclick = () => {
    setDetectionParams(initialDetectionParams);
    runDetection();
  };
  document.getElementById("save-spikes").onclick = async () => {
    const status = document.getElementById("save-status");
    status.textContent = "Saving...";
    const pairs = peaks.map((p, i) => ({p, label: labels[i]})).sort((a, b) => a.p - b.p);
    const res = await fetch(`/neurons/${NEURON_ID}/spikes/save`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({peaks: pairs.map(x => x.p), labels: pairs.map(x => x.label), params: getDetectionParams()}),
    });
    const data = await res.json();
    if (data.ok) {
      status.textContent = `Saved ${data.n_peaks} spikes. The cell is marked as reviewed.`;
    } else {
      status.textContent = data.error || "Save failed.";
    }
  };
  const markReviewed = document.getElementById("mark-reviewed");
  markReviewed.onclick = async () => {
    const status = document.getElementById("save-status");
    const res = await fetch(`/neurons/${NEURON_ID}/review/mark`, {method: "POST"});
    const data = await res.json();
    status.textContent = data.ok ? "Marked as reviewed." : (data.error || "Could not mark the cell as reviewed.");
  };
  fetch(`/neurons/${NEURON_ID}/spikes/data`).then(async r => {
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || `HTTP ${r.status}`);
    return data;
  }).then(data => {
    trace = data.trace || [];
    fps = Number(data.fps || 1);
    peaks = data.peaks || [];
    labels = data.labels || [];
    initialDetectionParams = data.params;
    setDetectionParams(initialDetectionParams);
    updateDetectionStats();
    start = 0;
    stop = trace.length;
    draw();
  }).catch(error => {
    document.getElementById("detection-status").textContent = error.message || "Spike data failed to load.";
    draw();
  });
  new ResizeObserver(draw).observe(canvas);
  window.addEventListener("resize", draw);
})();
