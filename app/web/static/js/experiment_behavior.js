(function () {
  const EXP_ID = JSON.parse(document.getElementById("page-data").textContent).expId;
  const behaviorCanvas = document.getElementById("behavior-canvas");
  const behaviorCtx = behaviorCanvas.getContext("2d");
  function resizeBehaviorCanvas() {
    const rect = behaviorCanvas.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    behaviorCanvas.width = Math.max(900, Math.round(rect.width * dpr));
    behaviorCanvas.height = Math.max(320, Math.round(rect.height * dpr));
  }
  function finiteRange(values, floor = 0) {
    const finite = values.filter(Number.isFinite);
    if (!finite.length) return [0, 1];
    let lo = Math.min(...finite), hi = Math.max(...finite);
    lo = Math.min(lo, floor);
    if (hi <= lo) hi = lo + 1;
    const pad = Math.max((hi - lo) * 0.12, 0.5);
    return [lo - pad, hi + pad];
  }
  function drawBehavior(data) {
    resizeBehaviorCanvas();
    const ctx = behaviorCtx, w = behaviorCanvas.width, h = behaviorCanvas.height;
    const plot = {left: 72, right: 24, top: 28, bottom: 54};
    const pw = w - plot.left - plot.right, ph = h - plot.top - plot.bottom;
    ctx.fillStyle = "#0b0f14";
    ctx.fillRect(0, 0, w, h);
    if (!data.available) {
      ctx.fillStyle = "#9aa8b7";
      ctx.font = "18px Segoe UI";
      ctx.textAlign = "center";
      ctx.fillText(data.message || "No behavior results.", w / 2, h / 2);
      return;
    }
    const t = data.time_s || [];
    const velocity = ((data.velocity && data.velocity.length) ? data.velocity : data.speed || []).map(v => v === null ? NaN : Number(v));
    const segments = data.segments || [];
    const hasVelocity = velocity.some(Number.isFinite);
    const x0 = t.length ? t[0] : 0, x1 = t.length ? t[t.length - 1] : 1;
    let [y0, y1] = finiteRange(velocity, 0);
    const threshold = Math.abs(Number(data.threshold || 0));
    const maxAbs = Math.max(Math.abs(y0), Math.abs(y1), threshold * 1.15, 1e-6);
    if (data.behavior_source === "ts" && data.ts_velocity_signed) {
      y0 = -maxAbs;
      y1 = maxAbs;
    }
    const x = i => plot.left + ((t[i] - x0) / Math.max(1e-9, x1 - x0)) * pw;
    const xTime = value => plot.left + ((value - x0) / Math.max(1e-9, x1 - x0)) * pw;
    const y = v => plot.top + (1 - (v - y0) / Math.max(1e-9, y1 - y0)) * ph;
    const labelColor = {active: "rgba(55,198,166,0.23)", inactive: "rgba(74,163,255,0.20)", unlabeled: "rgba(228,173,69,0.15)"};
    for (const segment of segments) {
      const color = labelColor[segment.label];
      if (color) {
        const x0s = xTime(Number(segment.t0));
        const x1s = xTime(Number(segment.t1));
        ctx.fillStyle = color;
        ctx.fillRect(x0s, plot.top, Math.max(1, x1s - x0s), ph);
      }
    }
    ctx.strokeStyle = "#26303a";
    ctx.lineWidth = 1;
    for (let k = 0; k < 5; k++) {
      const yy = plot.top + k / 4 * ph;
      ctx.beginPath(); ctx.moveTo(plot.left, yy); ctx.lineTo(plot.left + pw, yy); ctx.stroke();
    }
    const zeroY = y(0);
    ctx.strokeStyle = "rgba(241,246,251,0.28)";
    ctx.beginPath(); ctx.moveTo(plot.left, zeroY); ctx.lineTo(plot.left + pw, zeroY); ctx.stroke();
    const thresholdY = y(threshold);
    const negThresholdY = y(-threshold);
    ctx.strokeStyle = "#e4ad45";
    ctx.setLineDash([8, 8]);
    ctx.beginPath(); ctx.moveTo(plot.left, thresholdY); ctx.lineTo(plot.left + pw, thresholdY); ctx.stroke();
    if (data.behavior_source === "ts" && data.ts_velocity_signed) {
      ctx.beginPath(); ctx.moveTo(plot.left, negThresholdY); ctx.lineTo(plot.left + pw, negThresholdY); ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.fillStyle = "#f1f6fb";
    ctx.font = "14px Segoe UI";
    ctx.textAlign = "right";
    ctx.fillText(`|velocity| threshold ${threshold.toFixed(1)} ${data.velocity_units || ""}`, plot.left + pw - 8, Math.max(plot.top + 16, thresholdY - 8));
    ctx.strokeStyle = "#f1f6fb";
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    let moved = false;
    for (let i = 0; i < velocity.length; i++) {
      if (!Number.isFinite(velocity[i])) continue;
      if (!moved) { ctx.moveTo(x(i), y(velocity[i])); moved = true; }
      else ctx.lineTo(x(i), y(velocity[i]));
    }
    ctx.stroke();
    if (!hasVelocity) {
      ctx.fillStyle = "#e4ad45";
      ctx.font = "17px Segoe UI";
      ctx.textAlign = "center";
      ctx.fillText("Velocity trace is unavailable for this run. Rerun Behavior Split to save it.", plot.left + pw / 2, plot.top + ph / 2);
    }
    ctx.strokeStyle = "#43505f";
    ctx.strokeRect(plot.left, plot.top, pw, ph);
    ctx.fillStyle = "#aeb9c6";
    ctx.font = "15px Segoe UI";
    ctx.textAlign = "right";
    for (const v of [y0, (y0 + y1) / 2, y1]) ctx.fillText(v.toFixed(1), plot.left - 10, y(v));
    ctx.textAlign = "center";
    ctx.fillText("Time (s)", plot.left + pw / 2, h - 16);
    ctx.save(); ctx.translate(22, plot.top + ph / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(`Velocity (${data.velocity_units || "cm/s"})`, 0, 0); ctx.restore();
  }
  function showBehaviorError(message) {
    document.getElementById("behavior-run").textContent = "Error";
    document.getElementById("behavior-summary").textContent = message || "Behavior results failed to load.";
    drawBehavior({available: false, message: message || "Behavior results failed to load."});
  }
  fetch(`/experiments/${EXP_ID}/behavior/data`).then(r => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }).then(data => {
    document.getElementById("behavior-run").textContent = data.available ? `Run ${data.run_id}` : "No Run";
    const sourceText = data.behavior_source === "csv_fallback_from_ts"
      ? `CSV fallback${data.ts_fallback_reason ? " (TS unavailable)" : ""}`
      : data.behavior_source === "ts"
      ? `TS ${data.ts_velocity_key || ""}${data.ts_encoder_mode ? " (" + data.ts_encoder_mode + ")" : ""}`.trim()
      : (data.behavior_source || "behavior");
    document.getElementById("behavior-summary").textContent = data.available
      ? `${sourceText} | ${data.active_frames} active, ${data.inactive_frames} inactive, ${data.unlabeled_frames} unlabeled frames | threshold ${Number(data.threshold || 0).toFixed(1)} ${data.velocity_units || "cm/s"}`
      : (data.message || "No completed behavior run is available.");
    drawBehavior(data);
    new ResizeObserver(() => drawBehavior(data)).observe(behaviorCanvas);
    window.addEventListener("resize", () => drawBehavior(data));
  }).catch(error => showBehaviorError(error.message));
})();
