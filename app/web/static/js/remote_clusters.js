// The remote clusters panel of the Hypotheses tab, over all cells: the clusters found anatomically; the remote clusters (clusters of the "all" configuration that are disjoint from the
// anatomical ones: same cell, same sign and sharing less than OVERLAP of the smaller cluster's pixels with every anatomical cluster); and, among the remote ones, those explained by the
// same spikes: the cluster lies in the window of the anatomical configuration (the square of half-size `anatomy_radius_px` around the ROI, around each part of a merged ROI) of another
// cell whose spikes are the same (the same-spikes parameters of the cluster map panel). Counted from the saved cluster map (cluster-map.json).
(function () {
  const root = document.getElementById("remote-panel");
  if (!root) return;
  const PAGE = JSON.parse(document.getElementById("page-data").textContent);
  const EXP_ID = PAGE.expId, HYPOTHESIS = PAGE.hypothesis;
  const $ = id => document.getElementById(id);
  const {setup, format: fmt} = PlotCanvas;
  const COLORS = {anatomy: "#7f93a8", remote: "#ff9f43", explained: "#37c6a6"};
  const OVERLAP = 0.5;  // the share of the smaller cluster that two clusters share for them to be the same cluster
  let D = null, result = null;

  const isAnatomy = c => c.result_type.startsWith("anatomy");  // anatomy, anatomy_part2, ...
  const pixelKey = (y, x) => y * 100000 + x;
  const sameCluster = (a, b) => {
    const pixels = new Set(a.ys.map((y, i) => pixelKey(y, a.xs[i])));
    return a.sign === b.sign && b.ys.filter((y, i) => pixels.has(pixelKey(y, b.xs[i]))).length >= OVERLAP * Math.min(a.ys.length, b.ys.length);
  };

  // whether the centre of a cluster is inside the square the anatomical configuration looks at around a cell (around each part of a merged ROI)
  function inWindow(c, record) {
    const mean = values => values.reduce((u, v) => u + v, 0) / values.length;
    const centers = record.part_centers && record.part_centers.length ? record.part_centers : [[mean(record.roi.ys), mean(record.roi.xs)]];
    return centers.some(([y, x]) => Math.abs(c.center[0] - y) <= D.anatomy_radius_px && Math.abs(c.center[1] - x) <= D.anatomy_radius_px);
  }

  function count() {
    const minSpikes = Math.max(1, Number($("cm-min-spikes").value) || 1), frames = Math.max(0, Number($("cm-frames").value) || 0);
    const share = Math.min(1, Math.max(0.1, (Number($("cm-share").value) || 100) / 100));
    const cards = Object.keys(D.cells).map(Number).map(cell => ({cell, record: D.cells[cell]}));
    const groupOf = new Map();
    SameSpikes.groups(cards, minSpikes, frames, share).forEach(group => group.forEach(o => groupOf.set(o.cell, group)));
    const byCell = new Map();
    D.clusters.forEach(c => byCell.set(c.cell, [...(byCell.get(c.cell) || []), c]));
    let anatomy = 0, remote = 0, explained = 0;
    byCell.forEach((clusters, cell) => {
      if (D.cells[cell].motion !== "ok") return;  // as in the offsets: cells with a motion suspicion or without motion diagnostics are left out
      const anat = clusters.filter(isAnatomy);
      anatomy += anat.length;
      clusters.filter(c => c.result_type === "all" && !anat.some(a => sameCluster(a, c))).forEach(c => {
        remote++;
        if ((groupOf.get(cell) || []).some(o => o.cell !== cell && inWindow(c, o.record))) explained++;
      });
    });
    return {anatomy, remote, explained, minSpikes, frames, share, groups: new Set([...groupOf.values()]).size};
  }

  // three bars: the clusters found anatomically, the remote clusters, and the remote clusters explained by the same spikes (a subset of the remote ones)
  function bars(r) {
    const [ctx, W, H] = setup($("rc-hist")), L = 46, Rr = 12, T = 16, B = 46;
    const bars = [["anatomy", r.anatomy, "Anatomical"], ["remote", r.remote, "Remote"], ["explained", r.explained, "Explained by other neuron|with similar spikes"]];
    const yMax = Math.max(1, ...bars.map(b => b[1])), slot = (W - L - Rr) / bars.length, width = slot * 0.5, X = i => L + (i + 0.5) * slot, Y = v => T + 14 + (1 - v / yMax) * (H - T - 14 - B);
    ctx.font = "11px sans-serif"; ctx.textBaseline = "middle"; ctx.textAlign = "right";
    TimeAxis.nice(0, yMax, 5).filter(v => Number.isInteger(v)).forEach(v => { ctx.strokeStyle = "#1c2630"; ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - Rr, Y(v)); ctx.stroke(); ctx.fillStyle = "#9aa8b7"; ctx.fillText(fmt(v), L - 6, Y(v)); });
    bars.forEach(([key, value, text], i) => {
      ctx.fillStyle = COLORS[key]; ctx.fillRect(X(i) - width / 2, Y(value), width, Y(0) - Y(value));
      ctx.fillStyle = "#edf3fa"; ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText(String(value), X(i), Y(value) - 3);
      ctx.fillStyle = "#9aa8b7"; ctx.textBaseline = "top";
      text.split("|").forEach((line, k) => ctx.fillText(line, X(i), Y(0) + 6 + k * 13));
    });
    ctx.save(); ctx.translate(12, T + 14 + (H - T - 14 - B) / 2); ctx.rotate(-Math.PI / 2); ctx.fillStyle = "#edf3fa"; ctx.font = "12px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText("clusters (all cells)", 0, 0); ctx.restore();
  }

  function render() {
    if (!D) return;
    if (D.anatomy_radius_px === null) { $("rc-summary").textContent = "This run has no anatomical configuration (no anatomy result type), so there is no window to say what is near."; return; }
    const r = count();
    $("rc-rule").textContent = `same spikes: at least ${r.minSpikes} spikes, ${(r.share * 100).toFixed(0)}% within ${r.frames} frames (set in the cluster map above); ${r.groups} groups`;
    $("rc-summary").textContent = `${r.anatomy} anatomical clusters, ${r.remote} remote clusters, of which ${r.explained} (${r.remote ? (100 * r.explained / r.remote).toFixed(0) : 0}%) are explained by a cell with the same spikes. Cells with a motion suspicion are left out.`;
    result = r; bars(r);
  }

  async function start() {
    try {
      const res = await fetch(`/experiments/${EXP_ID}/cluster-map.json?h=${encodeURIComponent(HYPOTHESIS)}`), body = await res.json();
      if (!res.ok) throw new Error(body.message || `Request failed (${res.status})`);
      D = body;
      ["cm-min-spikes", "cm-frames", "cm-share"].forEach(id => $(id).addEventListener("change", render));
      window.addEventListener("resize", () => result && bars(result));
      render();
    } catch (error) { $("rc-summary").textContent = error.message; }
  }
  start();
})();
