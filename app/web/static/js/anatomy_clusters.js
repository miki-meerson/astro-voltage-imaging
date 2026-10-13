(function () {
  const EXP_ID = JSON.parse(document.getElementById("page-data").textContent).expId;
  const COL = Polarity.COLORS;
  const $ = id => document.getElementById(id);
  let D = null, pts = [];
  function setup(c) {
    const dpr = window.devicePixelRatio || 1, r = c.getBoundingClientRect();
    c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr);
    const ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, r.width, r.height);
    return [ctx, r.width, r.height];
  }
  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
    const step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag, out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.abs(v) < 1e-12 ? 0 : v);
    return out;
  }
  const fmt = v => Math.abs(v) >= 10 || Number.isInteger(v) ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2);
  function median(a) { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }

  function drawTraces() {
    const [ctx, W, H] = setup($("ac-traces")), L = 58, R = 14, T = 26, B = 40;
    const groups = {[Polarity.DEP]: [], [Polarity.HYPER]: []};
    D.clusters.forEach(c => groups[Polarity.of(c.sign)].push(c));
    const t = D.clusters.length ? D.clusters[0].t : [];
    if (!t.length) { ctx.fillStyle = "#9aa8b7"; ctx.font = "13px sans-serif"; ctx.fillText("No clusters", 20, 30); return; }
    const stats = {};
    for (const [g, list] of Object.entries(groups)) {
      if (!list.length) continue;
      const n = list.length, mean = t.map((_, i) => list.reduce((s, c) => s + c.trace[i], 0) / n);
      const sem = t.map((_, i) => n > 1 ? Math.sqrt(list.reduce((s, c) => s + (c.trace[i] - mean[i]) ** 2, 0) / (n - 1) / n) : 0);
      stats[g] = {mean, sem, n};
    }
    const showInd = $("ac-individual").checked, showSem = $("ac-sem").checked;
    let lo = Infinity, hi = -Infinity;
    const take = v => { lo = Math.min(lo, v * 100); hi = Math.max(hi, v * 100); };
    for (const [g, st] of Object.entries(stats)) st.mean.forEach((m, i) => { take(m + (showSem ? st.sem[i] : 0)); take(m - (showSem ? st.sem[i] : 0)); });
    const fitMean = $("ac-yrange").value === "mean";
    if (showInd && !fitMean) D.clusters.forEach(c => c.trace.forEach(take));
    const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad; lo = Math.min(lo, 0); hi = Math.max(hi, 0);
    const X = f => L + (f - t[0]) / (t[t.length - 1] - t[0]) * (W - L - R), Y = v => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
    // tested window shading (-10..+10)
    ctx.fillStyle = "rgba(120,140,170,0.10)"; ctx.fillRect(X(-10), T, X(10) - X(-10), H - T - B);
    ctx.font = "11px sans-serif"; ctx.fillStyle = "#9aa8b7"; ctx.textAlign = "center"; ctx.textBaseline = "top";
    niceTicks(t[0], t[t.length - 1], 8).forEach(f => { ctx.strokeStyle = "#1c2630"; ctx.beginPath(); ctx.moveTo(X(f), T); ctx.lineTo(X(f), H - B); ctx.stroke(); ctx.fillText(fmt(f), X(f), H - B + 4); });
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    niceTicks(lo, hi, 6).forEach(v => { ctx.strokeStyle = v === 0 ? "#3b4a5b" : "#1c2630"; ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - R, Y(v)); ctx.stroke(); ctx.fillStyle = "#9aa8b7"; ctx.fillText(fmt(v), L - 6, Y(v)); });
    ctx.strokeStyle = "#5a6b7d"; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(0), T); ctx.lineTo(X(0), H - B); ctx.stroke(); ctx.setLineDash([]);
    ctx.save(); ctx.beginPath(); ctx.rect(L, T, W - L - R, H - T - B); ctx.clip();  // traces outside the y range are cut at the frame
    if (showInd) D.clusters.forEach(c => { ctx.globalAlpha = 0.2; ctx.strokeStyle = Polarity.color(c.sign); ctx.lineWidth = 1; ctx.beginPath(); c.trace.forEach((v, i) => i ? ctx.lineTo(X(t[i]), Y(v * 100)) : ctx.moveTo(X(t[i]), Y(v * 100))); ctx.stroke(); ctx.globalAlpha = 1; });
    for (const [g, st] of Object.entries(stats)) {
      if (showSem && st.n > 1) { ctx.globalAlpha = 0.25; ctx.fillStyle = COL[g]; ctx.beginPath(); st.mean.forEach((m, i) => { const x = X(t[i]), y = Y((m + st.sem[i]) * 100); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); for (let i = t.length - 1; i >= 0; i--) ctx.lineTo(X(t[i]), Y((st.mean[i] - st.sem[i]) * 100)); ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1; }
      ctx.strokeStyle = COL[g]; ctx.lineWidth = 2.5; ctx.beginPath(); st.mean.forEach((m, i) => i ? ctx.lineTo(X(t[i]), Y(m * 100)) : ctx.moveTo(X(t[i]), Y(m * 100))); ctx.stroke(); ctx.lineWidth = 1;
    }
    ctx.restore();
    ctx.fillStyle = "#edf3fa"; ctx.font = "12px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText("Frames from spike", L + (W - L - R) / 2, H - 4);
    ctx.save(); ctx.translate(14, T + (H - T - B) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(`Cluster mean trace, dF/F (%), ${Polarity.AXIS_NOTE}`, 0, 0); ctx.restore();
    ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.font = "12px sans-serif";
    let lx = L + 6; for (const [g, st] of Object.entries(stats)) { ctx.fillStyle = COL[g]; ctx.fillRect(lx, 7, 14, 3); ctx.fillStyle = "#edf3fa"; const label = `${g} (n=${st.n})`; ctx.fillText(label, lx + 18, 3); lx += 30 + ctx.measureText(label).width; }
    ctx.fillStyle = "#6f7f90"; ctx.font = "10px sans-serif"; ctx.textAlign = "right"; ctx.fillText("shaded: tested window (-10..+10)", W - R, 3);
  }

  function drawScatter() {
    const [ctx, W, H] = setup($("ac-scatter")), L = 46, R = 14, T = 26, B = 40;
    pts = [];
    if (!D.clusters.length) return;
    const maxAbs = Math.max(8, ...D.clusters.map(c => Math.max(Math.abs(c.dx), Math.abs(c.dy)))) * 1.18;
    const side = Math.min(W - L - R, H - T - B), cx0 = L + (W - L - R) / 2, cy0 = T + (H - T - B) / 2, k = side / 2 / maxAbs;
    const PX = x => cx0 + x * k, PY = y => cy0 + y * k;  // image orientation: +dy points down
    ctx.font = "11px sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
    const rings = niceTicks(0, maxAbs, 4).filter(r => r > 0);
    rings.forEach(r => { ctx.strokeStyle = "#26303a"; ctx.beginPath(); ctx.arc(cx0, cy0, r * k, 0, 2 * Math.PI); ctx.stroke(); ctx.fillStyle = "#7d8c9c"; ctx.fillText(`${fmt(r)} px`, cx0 + r * k * 0.707 + 2, cy0 - r * k * 0.707 - 2); });
    ctx.strokeStyle = "#3b4a5b"; ctx.beginPath(); ctx.moveTo(cx0 - side / 2, cy0); ctx.lineTo(cx0 + side / 2, cy0); ctx.moveTo(cx0, cy0 - side / 2); ctx.lineTo(cx0, cy0 + side / 2); ctx.stroke();
    ctx.strokeStyle = "#f7d154"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(cx0 - 6, cy0); ctx.lineTo(cx0 + 6, cy0); ctx.moveTo(cx0, cy0 - 6); ctx.lineTo(cx0, cy0 + 6); ctx.stroke(); ctx.lineWidth = 1;
    D.clusters.forEach(c => { const x = PX(c.dx), y = PY(c.dy); ctx.globalAlpha = 0.85; ctx.fillStyle = Polarity.color(c.sign); ctx.beginPath(); ctx.arc(x, y, 5, 0, 2 * Math.PI); ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = "#0b0f14"; ctx.stroke(); pts.push({x, y, c}); });
    ctx.fillStyle = "#edf3fa"; ctx.font = "12px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText("dx from ROI centre (px)", cx0, H - 4);
    ctx.save(); ctx.translate(12, cy0); ctx.rotate(-Math.PI / 2); ctx.fillText("dy from ROI centre (px, + = down)", 0, 0); ctx.restore();
    const med = g => { const d = D.clusters.filter(c => Polarity.of(c.sign) === g).map(c => c.dist); return d.length ? `${g}: median ${median(d).toFixed(1)} px (n=${d.length})` : null; };
    ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillStyle = "#9aa8b7"; ctx.font = "11px sans-serif";
    ctx.fillText([med(Polarity.DEP), med(Polarity.HYPER)].filter(Boolean).join("   |   ") + "   |   yellow + = ROI centre", L, 5);
  }

  const tip = $("ac-tip");
  $("ac-scatter").addEventListener("mousemove", e => {
    const r = e.currentTarget.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const hit = pts.map(p => ({p, d: Math.hypot(p.x - mx, p.y - my)})).filter(o => o.d < 10).sort((a, b) => a.d - b.d)[0];
    if (!hit) { tip.style.display = "none"; e.currentTarget.style.cursor = "default"; return; }
    const c = hit.p.c; e.currentTarget.style.cursor = "pointer";
    tip.innerHTML = `Cell ${c.cell} &middot; ${Polarity.of(c.sign)}<br>distance ${c.dist.toFixed(1)} px &middot; p=${c.p == null ? 'n/a' : c.p.toFixed(3)} &middot; ${c.n_px} px (bin ${c.bin}), frames ${c.first_rel}..${c.last_rel}<br><span style="color:#9aa8b7">click to open the cell</span>`;
    tip.style.display = "block"; tip.style.left = (e.clientX + 12) + "px"; tip.style.top = (e.clientY + 12) + "px";
  });
  $("ac-scatter").addEventListener("mouseleave", () => { tip.style.display = "none"; });
  $("ac-scatter").addEventListener("click", e => {
    const r = e.currentTarget.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const hit = pts.map(p => ({p, d: Math.hypot(p.x - mx, p.y - my)})).filter(o => o.d < 10).sort((a, b) => a.d - b.d)[0];
    const sel = document.getElementById("neuron-select");
    if (hit && hit.p.c.neuron_id && sel) { sel.value = String(hit.p.c.neuron_id); sel.dispatchEvent(new Event("change")); const cv = document.getElementById("cell-view"); if (cv) cv.scrollIntoView({behavior: "smooth"}); }
  });
  const ROWS = [
    ["cells_with_cluster", "Cells with a significant cluster", v => v.toFixed(0), "more cells than chance"],
    ["one_sidedness", "Share of the more common polarity", v => (v * 100).toFixed(0) + "%", "more one-sided than chance"],
    ["similarity_same_time", "Clusters alike, same timing (correlation)", v => v.toFixed(2), "more alike than chance"],
    ["similarity_shift3", "Clusters alike, allowing a shift of up to 3 frames", v => v.toFixed(2), "more alike than chance"],
    ["distance_px", "Distance from the ROI centre (px)", v => v.toFixed(1), "closer to the ROI than chance"],
  ];
  function showTest(g) {
    const el = $("ac-test");
    if (!g) { el.innerHTML = '<div class="muted" style="font-size:12px">Real vs fake test: not computed yet for this run (cluster job: <code>cluster-code/run_anatomy_group_test.sh</code>).</div>'; return; }
    const rows = ROWS.map(([k, label, f, claim]) => {
      const s = g.stats[k];
      if (!s || s.p === null || s.p === undefined) return `<tr><td>${label}</td><td>${s && s.real != null ? f(s.real) : "n/a"}</td><td colspan="3" class="muted">too few clusters in the fake groups to compare</td></tr>`;
      const hit = s.p < 0.05, col = hit ? "#37c6a6" : "#9aa8b7";
      return `<tr><td>${label}</td><td><b>${f(s.real)}</b></td><td>${f(s.fake_median)} <span class="muted">(${f(s.fake_low)} to ${f(s.fake_high)})</span></td><td>${s.p < 0.001 ? "<0.001" : s.p.toFixed(3)}</td><td style="color:${col}">${hit ? claim : "within chance"}</td></tr>`;
    }).join("");
    el.innerHTML = `
      <h3 style="margin:0 0 4px">Is this more than chance?</h3>
      <p class="muted" style="font-size:12px; margin:0 0 6px">For every cell we kept ${g.n_fakes_per_cell} fake versions of its result (the spike times shifted in time, so spikes no longer line up with the movie). We ran the very same cluster search on the fakes, so fake clusters are picked exactly like the real ones. Then we built ${g.n_draws.toLocaleString()} fake groups (one fake per cell, ${g.n_cells} cells) and counted how often a fake group looked at least as striking as the real one. That fraction is the p-value. Fake columns: typical value and the middle 95% of the fake groups. Five numbers are tested, so one p below 0.05 by luck is not surprising.</p>
      <div class="table-wrap"><table><tr><th>What we compare</th><th>Real</th><th>Fake groups</th><th>p</th><th>Verdict</th></tr>${rows}</table></div>`;
    const pf = g.polarity_in_fakes;
    if (pf && pf.hyperpolarizing_share_fake != null) {
      const share = pf.hyperpolarizing_share_fake, lopsided = share >= 0.9 || share <= 0.1;
      el.insertAdjacentHTML("beforeend", `<p style="font-size:13px; margin:8px 0 0; padding:6px 10px; border:1px solid ${lopsided ? "#b8860b" : "var(--line)"}; border-radius:6px; color:${lopsided ? "#f0c040" : "var(--muted)"}">${lopsided ? "&#9888; " : ""}<b>Polarity check:</b> in the fake data ${pf.n_fake_clusters.toLocaleString()} "significant" clusters were found, and <b>${(share * 100).toFixed(1)}%</b> of them are hyperpolarizing (real clusters: ${pf.hyperpolarizing_share_real == null ? "n/a" : (pf.hyperpolarizing_share_real * 100).toFixed(0) + "%"}). ${lopsided ? "The statistic finds ${share >= 0.9 ? "hyperpolarizing clusters far more easily than depolarizing" : "depolarizing clusters far more easily than hyperpolarizing"} ones even in noise, so a lopsided polarity of the real clusters says nothing, and real ${share >= 0.9 ? "depolarizing" : "hyperpolarizing"} clusters are hard to detect." : "The fake clusters are mixed in polarity."}</p>`);
    }
  }
  const redraw = () => { if (D && D.available) { drawTraces(); drawScatter(); } };
  $("ac-individual").addEventListener("change", redraw); $("ac-sem").addEventListener("change", redraw); $("ac-yrange").addEventListener("change", redraw);
  window.addEventListener("resize", redraw);
  fetch(`/experiments/${EXP_ID}/anatomy-clusters.json`)
    .then(r => r.json())
    .then(d => {
      D = d;
      if (!d.available) { $("ac-summary").textContent = d.message || "Not available."; return; }
      const c = d.counts;
      $("ac-summary").textContent = `Run ${d.run_id}: ${c.cells_shown} cells, ${c.clusters_shown} clusters shown (${c.cells_with_sig_anatomy_cluster} cells have a significant anatomy cluster).`;
      const ex = [];
      if (c.motion_flagged) ex.push(`${c.motion_flagged} motion-suspected excluded (cells ${d.excluded.motion_flagged.join(", ")})`);
      if (c.no_motion_diagnostics) ex.push(`${c.no_motion_diagnostics} without motion diagnostics excluded (cells ${d.excluded.no_motion_diagnostics.join(", ")})`);
      $("ac-excluded").textContent = ex.join("; ");
      redraw();
      showTest(d.group_test);
    })
    .catch(err => { $("ac-summary").textContent = "Failed to load: " + err; });
})();
