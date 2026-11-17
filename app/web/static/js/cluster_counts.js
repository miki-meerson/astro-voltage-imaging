// The Clusters vs. spikes panel of the Hypotheses tab: one dot per tested cell, its number of spikes against its number of significant clusters (of every result type added up, or of the one
// the user picks), with the line of the linear fit and the Spearman correlation. Drawn from the saved summary of the run (app/services/hypothesis_summary.py `cluster_counts`).
(function () {
  const root = document.getElementById("counts-panel");
  if (!root) return;
  const PAGE = JSON.parse(document.getElementById("page-data").textContent);
  const EXP_ID = PAGE.expId, HYPOTHESIS = PAGE.hypothesis;
  const $ = id => document.getElementById(id);
  const {setup, format: fmt, pValue} = PlotCanvas;
  const L = 44, R = 12, T = 24, B = 38, GRID = "#1c2630", TEXT = "#9aa8b7", BRIGHT = "#edf3fa", DOT = "#8bd3c7";
  let D = null, pts = [];

  const clustersOf = (c, choice) => choice === D.count_all ? Object.values(c.clusters).reduce((u, v) => u + v, 0) : c.clusters[choice] || 0;

  function draw() {
    const [ctx, W, H] = setup($("cs-scatter")), choice = $("cs-type").value, stat = D.stats[choice];
    pts = [];
    const xMax = Math.max(...D.cells.map(c => c.spikes)) * 1.06 || 1, yMax = Math.max(1, ...D.cells.map(c => clustersOf(c, choice))) + 0.7;
    const X = v => L + v / xMax * (W - L - R), Y = v => T + (1 - v / yMax) * (H - T - B);
    ctx.font = "11px sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let v = 0; v <= yMax; v += Math.max(1, Math.ceil(yMax / 6))) { ctx.strokeStyle = GRID; ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - R, Y(v)); ctx.stroke(); ctx.fillStyle = TEXT; ctx.fillText(fmt(v), L - 6, Y(v)); }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    TimeAxis.nice(0, xMax, 5).forEach(v => { ctx.strokeStyle = GRID; ctx.beginPath(); ctx.moveTo(X(v), T); ctx.lineTo(X(v), H - B); ctx.stroke(); ctx.fillStyle = TEXT; ctx.fillText(fmt(v), X(v), H - B + 4); });
    if (stat) {
      ctx.strokeStyle = "#e7edf4"; ctx.lineWidth = 1.4; ctx.setLineDash([6, 4]); ctx.beginPath();
      ctx.moveTo(X(0), Y(stat.intercept)); ctx.lineTo(X(xMax), Y(stat.intercept + stat.slope * xMax)); ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1;
    }
    D.cells.forEach(c => {  // the clusters are whole numbers: a small fixed shift per cell keeps the dots on the same count apart
      const x = X(c.spikes), y = Y(clustersOf(c, choice) + ((c.cell * 37) % 11 - 5) / 40), suspected = c.motion === D.motion_suspected;
      ctx.beginPath(); ctx.arc(x, y, 5, 0, 2 * Math.PI); ctx.globalAlpha = 0.85; ctx.fillStyle = DOT; if (!suspected) ctx.fill();
      ctx.globalAlpha = 1; ctx.strokeStyle = suspected ? DOT : "#0b0f14"; ctx.lineWidth = suspected ? 1.8 : 1; ctx.stroke(); ctx.lineWidth = 1; pts.push({x, y, c});
    });
    ctx.fillStyle = BRIGHT; ctx.font = "12px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText("spikes", L + (W - L - R) / 2, H - 4);
    ctx.save(); ctx.translate(12, T + (H - T - B) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("significant clusters", 0, 0); ctx.restore();
    ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillStyle = TEXT; ctx.font = "11px sans-serif";
    ctx.fillText(stat ? `n=${stat.n}, Spearman rho=${stat.rho.toFixed(2)}, p=${pValue(stat.p)}; line p=${pValue(stat.slope_p)}` : "too few cells for a correlation", L, 5);
  }

  const tip = $("cs-tip"), canvas = $("cs-scatter");
  canvas.addEventListener("mousemove", e => {
    const rect = canvas.getBoundingClientRect(), hit = pts.map(o => ({o, d: Math.hypot(o.x - (e.clientX - rect.left), o.y - (e.clientY - rect.top))})).filter(h => h.d < 10).sort((a, b) => a.d - b.d)[0];
    if (!hit) { tip.style.display = "none"; return; }
    const c = hit.o.c;
    tip.textContent = `Cell ${c.cell}: ${c.spikes} spikes, ${clustersOf(c, $("cs-type").value)} clusters${c.motion === D.motion_suspected ? " (motion suspected)" : ""}`;
    tip.style.display = "block"; tip.style.left = (e.clientX + 12) + "px"; tip.style.top = (e.clientY + 12) + "px";
  });
  canvas.addEventListener("mouseleave", () => { tip.style.display = "none"; });

  async function load() {
    try {
      const res = await fetch(`/experiments/${EXP_ID}/cluster-counts.json?h=${encodeURIComponent(HYPOTHESIS)}`), body = await res.json();
      if (!res.ok) throw new Error(body.message || `Request failed (${res.status})`);
      if (!body.available) throw new Error(body.message);
      D = body;
      $("cs-type").innerHTML = [[D.count_all, "all result types"], ...D.result_types.map(name => [name, name])].map(([value, text]) => `<option value="${value}">${text}</option>`).join("");
      $("cs-summary").textContent = `${D.cells.length} tested cells of the newest run (${D.run_id}); hollow: motion suspected.`;
      $("cs-type").addEventListener("change", draw);
      window.addEventListener("resize", draw);
      draw();
    } catch (error) { $("cs-summary").textContent = error.message; }
  }
  load();
})();
