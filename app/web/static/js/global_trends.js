// The Global trends panel of the Hypotheses tab: the traces of the signals the whole field shares and a map of where the chosen one is strong.
(function () {
  const root = document.getElementById("global-trends-panel");
  if (!root) return;
  const PAGE = JSON.parse(document.getElementById("page-data").textContent);
  const EXP_ID = PAGE.expId, HYPOTHESIS = PAGE.hypothesis;
  const $ = id => document.getElementById(id);
  const COLORS = {global_mean: "#7bd88f", dx: "#f7d154", dy: "#9bd0ff", shift_mag: "#ff9f6b", reg_corr: "#c3a6ff"};
  const GROUPS = [["global_mean"], ["dx", "dy", "shift_mag"], ["reg_corr"]];  // traces drawn together, one plot per group
  const RAMP = [[44, 90, 168], [244, 241, 232], [196, 38, 44]];  // weak = blue, strong = red
  let D = null, layout = null;

  function setup(canvas) {
    const ratio = window.devicePixelRatio || 1, box = canvas.getBoundingClientRect();
    canvas.width = Math.round(box.width * ratio); canvas.height = Math.round(box.height * ratio);
    const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, box.width, box.height);
    return [ctx, box.width, box.height];
  }
  const rampColor = t => {
    const x = Math.max(0, Math.min(1, t)) * (RAMP.length - 1), i = Math.min(RAMP.length - 2, Math.floor(x)), f = x - i;
    return RAMP[i].map((c, k) => Math.round(c + (RAMP[i + 1][k] - c) * f));
  };
  const fmt = v => Math.abs(v) >= 100 || Number.isInteger(v) ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2);
  const chosen = () => $("gt-series").value;

  function drawTraces() {
    const [ctx, W, H] = setup($("gt-traces")), L = 60, R = 12, T = 8, B = 30, gap = 14;
    const groups = GROUPS.map(g => D.series.filter(s => g.includes(s.name))).filter(g => g.length);
    const t0 = D.times[0], t1 = D.times[D.times.length - 1], X = t => L + (t - t0) / Math.max(1e-9, t1 - t0) * (W - L - R);
    const plotH = (H - T - B - gap * (groups.length - 1)) / groups.length;
    groups.forEach((group, i) => {
      const top = T + i * (plotH + gap), all = group.flatMap(s => s.values);
      let lo = Math.min(...all), hi = Math.max(...all); const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
      const Y = v => top + (1 - (v - lo) / (hi - lo)) * plotH;
      ctx.font = "11px sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      for (const v of TimeAxis.nice(lo, hi, 4)) { ctx.strokeStyle = "#1c2630"; ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - R, Y(v)); ctx.stroke(); ctx.fillStyle = "#9aa8b7"; ctx.fillText(fmt(v), L - 6, Y(v)); }
      ctx.save(); ctx.beginPath(); ctx.rect(L, top, W - L - R, plotH); ctx.clip();
      D.cuts.forEach(c => { ctx.setLineDash([2, 4]); ctx.strokeStyle = "#5a6b7d"; ctx.beginPath(); ctx.moveTo(X(c), top); ctx.lineTo(X(c), top + plotH); ctx.stroke(); ctx.setLineDash([]); });
      group.forEach(s => {
        const on = s.name === chosen();
        ctx.strokeStyle = COLORS[s.name]; ctx.globalAlpha = on ? 1 : 0.65; ctx.lineWidth = on ? 2 : 1.1; ctx.beginPath();
        s.values.forEach((v, k) => k ? ctx.lineTo(X(D.times[k]), Y(v)) : ctx.moveTo(X(D.times[k]), Y(v))); ctx.stroke();
      });
      ctx.restore(); ctx.globalAlpha = 1; ctx.lineWidth = 1;
      ctx.textAlign = "left"; ctx.textBaseline = "top"; let lx = L + 6;
      group.forEach(s => { ctx.fillStyle = COLORS[s.name]; ctx.fillRect(lx, top + 5, 12, 3); ctx.fillStyle = "#e7edf4"; const label = `${s.label} (${s.unit})`; ctx.fillText(label, lx + 16, top + 2); lx += 30 + ctx.measureText(label).width; });
    });
    ctx.fillStyle = "#9aa8b7"; ctx.font = "11px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "top";
    const base = H - B + 4;
    for (const t of TimeAxis.nice(t0, t1, 8)) ctx.fillText(fmt(t), X(t), base);
    ctx.fillStyle = "#e7edf4"; ctx.fillText("seconds of the recording (after the bad segments were removed)", L + (W - L - R) / 2, base + 14);
  }

  function drawMap() {
    const series = D.series.find(s => s.name === chosen()), rows = series.map, h = rows.length, w = rows[0].length, vmax = series.map_max || 1;
    const [ctx, W, H] = setup($("gt-map")), legend = 46, size = Math.min(W, H - legend);
    const image = document.createElement("canvas"); image.width = w; image.height = h;
    const data = image.getContext("2d").createImageData(w, h);
    rows.forEach((row, y) => row.forEach((v, x) => { const [r, g, b] = rampColor(v / vmax), i = 4 * (y * w + x); data.data[i] = r; data.data[i + 1] = g; data.data[i + 2] = b; data.data[i + 3] = 255; }));
    image.getContext("2d").putImageData(data, 0, 0);
    const x0 = (W - size) / 2; layout = {x0, y0: 0, size, w, h, rows};
    ctx.imageSmoothingEnabled = false; ctx.drawImage(image, x0, 0, size, size);
    const b = D.border_bins, k = size / w;  // the border the STA test leaves out: a dashed frame at its inner edge
    if (b.top + b.bottom + b.left + b.right > 0) { ctx.setLineDash([5, 4]); ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.2; ctx.strokeRect(x0 + b.left * k, b.top * k, size - (b.left + b.right) * k, size - (b.top + b.bottom) * k); ctx.setLineDash([]); }
    const barY = size + 12, barX = x0, barW = size;
    for (let i = 0; i < barW; i++) { const [r, g, b] = rampColor(i / (barW - 1)); ctx.fillStyle = `rgb(${r},${g},${b})`; ctx.fillRect(barX + i, barY, 1, 12); }
    ctx.fillStyle = "#9aa8b7"; ctx.font = "11px sans-serif"; ctx.textBaseline = "top"; ctx.textAlign = "left"; ctx.fillText("weak", barX, barY + 15);
    ctx.textAlign = "center"; ctx.fillText(fmt(vmax / 2), barX + barW / 2, barY + 15);
    ctx.textAlign = "right"; ctx.fillText(`strong (${fmt(vmax)})`, barX + barW, barY + 15);
    $("gt-readout").textContent = layout.caption = `Amplitude of "${series.label}" in ${D.map_unit}. Hover the map for values.`;
  }

  function draw() { if (D) { drawTraces(); drawMap(); } }
  $("gt-map").addEventListener("mousemove", e => {
    if (!layout) return;
    const box = e.currentTarget.getBoundingClientRect(), x = Math.floor((e.clientX - box.left - layout.x0) / layout.size * layout.w), y = Math.floor((e.clientY - box.top - layout.y0) / layout.size * layout.h);
    $("gt-readout").textContent = x >= 0 && y >= 0 && x < layout.w && y < layout.h ? `row ${y}, column ${x} (map pixel of ${D.spatial_bin} px): ${layout.rows[y][x].toFixed(2)}` : layout.caption;
  });
  $("gt-series").addEventListener("change", draw);
  window.addEventListener("resize", draw);

  fetch(`/experiments/${EXP_ID}/global-trends.json?h=${encodeURIComponent(HYPOTHESIS)}`)
    .then(async res => { const body = await res.json(); if (!res.ok) throw new Error(body.message || `Not available (${res.status})`); return body; })
    .then(d => {
      D = d;
      $("gt-series").innerHTML = d.series.map(s => `<option value="${s.name}">${s.label}</option>`).join("");
      $("gt-summary").textContent = `Run ${d.run_id}: ${d.frames} frames (${(d.frames / d.fps / 60).toFixed(1)} min); the STA test leaves out a border of ${[d.border_bins.top, d.border_bins.bottom, d.border_bins.left, d.border_bins.right].map(v => v * d.spatial_bin).join(" / ")} px (top / bottom / left / right).`;
      draw();
    })
    .catch(err => { $("gt-summary").textContent = err.message || "Not available."; root.querySelector(".gt-grid").style.display = "none"; });
})();
