// The remote clusters panel of the Hypotheses tab, over all cells: the clusters found anatomically; the remote clusters (clusters of the "all" configuration that are disjoint from the
// anatomical ones: same cell, same sign and sharing less than OVERLAP of the smaller cluster's pixels with every anatomical cluster); and, among the remote ones, those explained by the
// same spikes: the cluster lies in the window of the anatomical configuration (the square of half-size `anatomy_radius_px` around the ROI, around each part of a merged ROI) of another
// cell whose spikes are the same (the same-spikes parameters of the cluster map panel). Counted from the saved cluster map (cluster-map.json).
// The same counts, without the duplicates, also give the two plots next to it: depolarizing against hyperpolarizing clusters, and the soma / dendrite / uncertain ROIs, each bar opaque for the
// ROIs that have a significant cluster (the ROI classes and the cells the run tested are read from roi-classes.json and cluster-counts.json).
(function () {
  const root = document.getElementById("remote-panel");
  if (!root) return;
  const PAGE = JSON.parse(document.getElementById("page-data").textContent);
  const EXP_ID = PAGE.expId, HYPOTHESIS = PAGE.hypothesis;
  const $ = id => document.getElementById(id);
  const {setup, format: fmt} = PlotCanvas;
  const COLORS = {anatomy: "#7f93a8", remote: "#ff9f43", explained: "#37c6a6"};
  const OVERLAP = 0.5;  // the share of the smaller cluster that two clusters share for them to be the same cluster
  let D = null, result = null, classes = null, tested = null;  // classes: the soma / dendrite labels of the ROIs; tested: the cells the run tested

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
    const perCell = new Map();  // cell -> its clusters once the "all" ones that repeat an anatomical one are removed
    const signs = {[Polarity.DEP]: 0, [Polarity.HYPER]: 0};  // the anatomical and the remote clusters by polarity
    byCell.forEach((clusters, cell) => {
      if (D.cells[cell].motion !== "ok") return;  // as in the offsets: cells with a motion suspicion or without motion diagnostics are left out
      const anat = clusters.filter(isAnatomy);
      anatomy += anat.length; perCell.set(cell, anat.length);
      anat.forEach(c => signs[Polarity.of(c.sign)]++);
      clusters.filter(c => c.result_type === "all" && !anat.some(a => sameCluster(a, c))).forEach(c => {
        remote++; perCell.set(cell, perCell.get(cell) + 1); signs[Polarity.of(c.sign)]++;
        if ((groupOf.get(cell) || []).some(o => o.cell !== cell && inWindow(c, o.record))) explained++;
      });
    });
    return {anatomy, remote, explained, perCell, signs, minSpikes, frames, share, groups: new Set([...groupOf.values()]).size};
  }

  // the horizontal grid lines of a plot of counts, with their whole-number labels on the left
  function grid(ctx, yMax, W, L, Rr, Y) {
    ctx.font = "11px sans-serif"; ctx.textBaseline = "middle"; ctx.textAlign = "right";
    TimeAxis.nice(0, yMax, 5).filter(v => Number.isInteger(v)).forEach(v => { ctx.strokeStyle = "#1c2630"; ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - Rr, Y(v)); ctx.stroke(); ctx.fillStyle = "#9aa8b7"; ctx.fillText(fmt(v), L - 6, Y(v)); });
  }

  // three bars: the clusters found anatomically, the remote clusters, and the remote clusters explained by the same spikes (a subset of the remote ones)
  function bars(r) {
    const [ctx, W, H] = setup($("rc-hist")), L = 46, Rr = 12, T = 16, B = 58;
    const bars = [["anatomy", r.anatomy, "Anatomical"], ["remote", r.remote, "Remote"], ["explained", r.explained, "Explained by|same spikes"]];
    const yMax = Math.max(1, ...bars.map(b => b[1])), slot = (W - L - Rr) / bars.length, width = slot * 0.5, X = i => L + (i + 0.5) * slot, Y = v => T + 14 + (1 - v / yMax) * (H - T - 14 - B);
    grid(ctx, yMax, W, L, Rr, Y);
    bars.forEach(([key, value, text], i) => {
      ctx.fillStyle = COLORS[key]; ctx.fillRect(X(i) - width / 2, Y(value), width, Y(0) - Y(value));
      ctx.fillStyle = "#edf3fa"; ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText(String(value), X(i), Y(value) - 3);
      ctx.fillStyle = "#9aa8b7"; ctx.textBaseline = "top";
      text.split("|").forEach((line, k) => ctx.fillText(line, X(i), Y(0) + 6 + k * 13));
    });
    ctx.save(); ctx.translate(12, T + 14 + (H - T - 14 - B) / 2); ctx.rotate(-Math.PI / 2); ctx.fillStyle = "#edf3fa"; ctx.font = "12px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText("clusters (all cells)", 0, 0); ctx.restore();
  }

  // the clusters without the duplicates (the anatomical ones and the remote ones), as a depolarizing bar and a hyperpolarizing one
  function polarityBars(r) {
    const [ctx, W, H] = setup($("pl-bars")), L = 46, Rr = 12, T = 16, B = 58;
    const bars = [Polarity.DEP, Polarity.HYPER].map(kind => [kind, r.signs[kind]]);
    const yMax = Math.max(1, ...bars.map(b => b[1])), slot = (W - L - Rr) / bars.length, width = slot * 0.5, Y = v => T + 14 + (1 - v / yMax) * (H - T - 14 - B);
    grid(ctx, yMax, W, L, Rr, Y);
    bars.forEach(([kind, n], i) => {
      const x = L + (i + 0.5) * slot - width / 2;
      ctx.fillStyle = Polarity.COLORS[kind]; ctx.fillRect(x, Y(n), width, Y(0) - Y(n));
      ctx.fillStyle = "#edf3fa"; ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText(String(n), x + width / 2, Y(n) - 3);
      ctx.fillStyle = "#9aa8b7"; ctx.textBaseline = "top"; ctx.fillText(kind[0].toUpperCase() + kind.slice(1), x + width / 2, Y(0) + 6);
    });
    ctx.save(); ctx.translate(12, T + 14 + (H - T - 14 - B) / 2); ctx.rotate(-Math.PI / 2); ctx.fillStyle = "#edf3fa"; ctx.font = "12px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText("clusters (all cells)", 0, 0); ctx.restore();
  }

  // soma, dendrite and uncertain ROIs of the tested cells without a motion suspicion: a bar as high as the number of ROIs of the class, in the colour of the class, opaque for the ROIs
  // with a significant cluster (an anatomical one or a remote one, so the duplicates count once) and faded for the others
  function classBars(r) {
    const [ctx, W, H] = setup($("sd-bars")), L = 46, Rr = 12, T = 16, B = 58, labelOf = new Map(classes.rois.map(o => [o.cell, o.label]));
    const cells = tested.filter(c => c.motion === "ok");
    const groups = classes.classes.map(k => { const own = cells.filter(c => labelOf.get(c.cell) === k.name); return {k, n: own.length, significant: own.filter(c => (r.perCell.get(c.cell) || 0) > 0).length, clusters: own.reduce((u, c) => u + (r.perCell.get(c.cell) || 0), 0)}; });
    const top = T + 30, yMax = Math.max(1, ...groups.map(o => o.n)), slot = (W - L - Rr) / groups.length, width = slot * 0.5, Y = v => top + (1 - v / yMax) * (H - top - B);  // room above the bars for two lines of text
    grid(ctx, yMax, W, L, Rr, Y);
    groups.forEach((o, i) => {
      const x = L + (i + 0.5) * slot - width / 2, share = o.n ? o.significant / o.n : 0;
      ctx.fillStyle = o.k.color; ctx.globalAlpha = 0.28; ctx.fillRect(x, Y(o.n), width, Y(0) - Y(o.n)); ctx.globalAlpha = 1;  // all the ROIs of the class, faded
      ctx.fillRect(x, Y(o.significant), width, Y(0) - Y(o.significant));  // the ROIs with a significant cluster, opaque, from the bottom
      ctx.fillStyle = "#edf3fa"; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      ctx.fillText(o.n ? `${o.significant} significant (${(100 * share).toFixed(0)}%)` : "none", x + width / 2, Y(o.n) - 3);
      ctx.fillText(`${o.n} ROI${o.n === 1 ? "" : "s"}`, x + width / 2, Y(o.n) - 16);
      ctx.fillStyle = "#9aa8b7"; ctx.textBaseline = "top"; ctx.fillText(o.k.label, x + width / 2, Y(0) + 6);
    });
    const keyY = H - 16;  // what opaque and faded mean
    ctx.font = "11px sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    [[1, "with a significant cluster"], [0.28, "without"]].reduce((x, [alpha, text]) => {
      ctx.globalAlpha = alpha; ctx.fillStyle = "#9aa8b7"; ctx.fillRect(x, keyY - 5, 10, 10); ctx.globalAlpha = 1; ctx.fillText(text, x + 15, keyY);
      return x + 15 + ctx.measureText(text).width + 14;
    }, L);
    ctx.save(); ctx.translate(12, top + (H - top - B) / 2); ctx.rotate(-Math.PI / 2); ctx.fillStyle = "#edf3fa"; ctx.font = "12px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText("ROIs", 0, 0); ctx.restore();
    const total = groups.reduce((u, o) => u + o.n, 0), unclassified = cells.length - total;
    $("sd-summary").textContent = `${total} ROI${total === 1 ? "" : "s"} by class${unclassified ? `; ${unclassified} unclassified left out` : ""}.`;
  }

  function render() {
    if (!D) return;
    if (D.anatomy_radius_px === null) { $("rc-summary").textContent = $("pl-summary").textContent = $("sd-summary").textContent = "This run has no anatomical configuration (no anatomy result type), so there is no window to say what is near."; return; }
    const r = count();
    $("rc-rule").textContent = `same spikes: at least ${r.minSpikes} spikes, ${(r.share * 100).toFixed(0)}% within ${r.frames} frames (set in the cluster map above); ${r.groups} groups`;
    $("rc-summary").textContent = `${r.anatomy} anatomical clusters, ${r.remote} remote clusters, of which ${r.explained} (${r.remote ? (100 * r.explained / r.remote).toFixed(0) : 0}%) are explained by a cell with the same spikes. Cells with a motion suspicion are left out.`;
    const dep = r.signs[Polarity.DEP], hyper = r.signs[Polarity.HYPER];
    $("pl-summary").textContent = `${dep} depolarizing and ${hyper} hyperpolarizing (${(100 * hyper / Math.max(1, dep + hyper)).toFixed(0)}%) once the "all" clusters that repeat an anatomical one are removed.`;
    result = r; bars(r); polarityBars(r); if (classes && tested) classBars(r);
  }

  // the classes of the ROIs (Preprocessing, ROI types) and the cells the run tested; when one is missing the soma / dendrite plot says why
  async function loadClassesAndTested() {
    try {
      const [classRes, testedRes] = await Promise.all([fetch(`/experiments/${EXP_ID}/preprocessing/roi-classes.json`), fetch(`/experiments/${EXP_ID}/cluster-counts.json?h=${encodeURIComponent(HYPOTHESIS)}`)]);
      const classBody = await classRes.json(), testedBody = await testedRes.json();
      if (!classRes.ok) throw new Error(classBody.message || `Request failed (${classRes.status})`);
      if (!testedRes.ok || !testedBody.available) throw new Error(testedBody.message || `Request failed (${testedRes.status})`);
      if (!classBody.auto_available) throw new Error("This experiment has no ROI classification yet: run it on the ROI types tab of the Preprocessing page.");
      classes = classBody; tested = testedBody.cells;
    } catch (error) { $("sd-summary").textContent = error.message; }
  }

  async function start() {
    try {
      const res = await fetch(`/experiments/${EXP_ID}/cluster-map.json?h=${encodeURIComponent(HYPOTHESIS)}`), body = await res.json();
      if (!res.ok) throw new Error(body.message || `Request failed (${res.status})`);
      D = body;
      await loadClassesAndTested();
      ["cm-min-spikes", "cm-frames", "cm-share"].forEach(id => $(id).addEventListener("change", render));
      window.addEventListener("resize", () => { if (result) { bars(result); polarityBars(result); if (classes && tested) classBars(result); } });
      render();
    } catch (error) { $("rc-summary").textContent = error.message; }
  }
  start();
})();
