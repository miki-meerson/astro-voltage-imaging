// The ROI groups tab of the Preprocessing page: groups of accepted Suite2p ROIs whose traces are alike (app/services/roi_groups.py). The field of view shows the ROIs (the ROIs of a group
// in one colour, the ROIs of no group faint); the group chosen in the list shows its members in their own colours, their traces on one zoomable plot (the height fits the part
// that is shown) and their correlation. What is shown is what the server saved; "Find groups" makes the saved file anew.
(function () {
  const root = document.getElementById("rg-panel");
  if (!root) return;
  const EXP_ID = JSON.parse(document.getElementById("page-data").textContent).expId;
  const URL = `/experiments/${EXP_ID}/preprocessing`;
  const $ = id => document.getElementById(id);
  const S = 2;  // on-screen pixels per field-of-view pixel
  const COLORS = ["#4fa3ff", "#ff7a59", "#2fbf71", "#d65db1", "#f2b84b", "#7bd3ea", "#b79cff", "#e8e26d", "#ff6b9a", "#9ad06a"];
  const UNGROUPED = "rgba(154, 168, 183, 0.7)", FG = "#c9d4df", GRID = "#1c2630", PLOT = {L: 56, R: 14, T: 10, B: 30};
  const {setup, format: fmt} = PlotCanvas;
  const base = $("rg-base"), ovl = $("rg-ovl"), stage = $("rg-stage"), canvas = $("rg-traces"), image = new Image();
  let D = null, owner = null, W = 0, H = 0, selected = null, loaded = false, down = null, T = null, view = null, hidden = new Set(), fitted = null;

  const groupColor = id => COLORS[id % COLORS.length];
  const memberColor = index => COLORS[index % COLORS.length];
  const traceColor = (entry, index) => entry.merged ? "#ffffff" : memberColor(index);  // the merged ROI's trace is white, the ROIs it was made of have their colours

  function drawOverlay() {
    const ctx = ovl.getContext("2d");
    ctx.clearRect(0, 0, ovl.width, ovl.height);
    const members = selected === null ? [] : D.groups[selected].cells;
    for (const roi of D.rois) {
      const inSelected = roi.group !== null && roi.group === selected;
      ctx.strokeStyle = roi.group === null ? UNGROUPED : inSelected ? memberColor(members.indexOf(roi.cell)) : groupColor(roi.group);
      ctx.globalAlpha = roi.group === null ? 1 : inSelected ? 1 : selected === null ? 1 : 0.45;
      ctx.lineWidth = inSelected ? 3 : roi.group === null ? 1 : 2;
      FovOutline.draw(ctx, roi.ypix, roi.xpix, W, S);
      if (inSelected) { ctx.fillStyle = ctx.strokeStyle; ctx.globalAlpha = 0.22; roi.ypix.forEach((y, i) => ctx.fillRect(roi.xpix[i] * S, y * S, S, S)); }
    }
    ctx.globalAlpha = 1;
  }

  function drawSummary() {
    const grouped = D.rois.filter(r => r.group !== null).length, merged = D.merged !== null;
    $("rg-threshold").value = D.threshold;
    $("rg-threshold").disabled = merged; $("rg-find").hidden = merged; $("rg-confirm").hidden = merged || !D.groups.length; $("rg-undo").hidden = !merged;
    $("rg-summary").textContent = merged
      ? `${D.groups.length} group(s) were merged into single ROIs on ${D.merged.confirmed_at.replace("T", " ")} (traces correlated at least ${D.threshold}). The merged ROIs are ordinary cells now, with the average of their members' traces. Undo the merge to change the groups.`
      : D.saved
        ? `${D.groups.length} group(s) of ROIs whose traces correlate at least ${D.threshold} on average: ${grouped} of the ${D.rois.length} accepted ROIs (iscell = 1) are in a group. Confirm them to merge each group into one ROI.`
        : `${D.rois.length} accepted ROIs (iscell = 1). No groups are saved yet: set the correlation and press Find groups.`;
    if (D.warning) $("rg-summary").textContent += " " + D.warning;
    const select = $("rg-group");
    select.innerHTML = D.groups.map(g => `<option value="${g.id}">Group ${g.id + 1} · ${g.size} ROIs · mean r ${g.mean_corr.toFixed(2)}</option>`).join("");
    select.disabled = !D.groups.length;
    if (selected !== null) select.value = selected;
  }

  const shade = r => r >= 0 ? `rgba(79, 163, 255, ${Math.min(1, r) * 0.75})` : `rgba(255, 122, 89, ${Math.min(1, -r) * 0.75})`;

  function drawMembers() {
    const box = $("rg-members");
    if (selected === null) { box.innerHTML = '<p class="muted">Choose a group in the list, or click a grouped ROI in the field of view.</p>'; return; }
    const g = D.groups[selected];
    const chips = T ? T.cells.map((c, i) => `<button type="button" class="secondary rg-chip${hidden.has(c.roi) ? " off" : ""}" data-roi="${c.roi}" title="Show or hide this trace"><i style="background:${traceColor(c, i)}"></i>${c.label}${c.merged ? "" : ` <span class="muted small">(ROI ${c.roi})</span>`}</button>`).join("") : "";
    const labels = g.cells_before || g.cells;  // after the merge the matrix is the one of the members before it, in their old cell numbers
    const rows = g.corr.map((row, i) => `<tr><th>cell ${labels[i]}</th>${row.map(r => `<td style="background:${shade(r)}">${r.toFixed(2)}</td>`).join("")}</tr>`).join("");
    box.innerHTML = `<p class="small">Group ${g.id + 1}: ${g.size} ROIs, correlation between members from ${g.min_corr.toFixed(2)} to 1 (mean ${g.mean_corr.toFixed(2)}). Cell numbers are the ones of the spikes and analyses${D.merged ? " (the table: before the merge)" : ""}.</p>
      <div class="rg-chips">${chips}</div>
      <table class="rg-matrix"><tr><th></th>${labels.map(c => `<th>${c}</th>`).join("")}</tr>${rows}</table>`;
    box.querySelectorAll("[data-roi]").forEach(button => button.addEventListener("click", () => {
      const roi = Number(button.dataset.roi);
      hidden.has(roi) ? hidden.delete(roi) : hidden.add(roi);
      drawMembers(); fitHeight(); drawTraces();
    }));
  }

  // the height that fits the shown traces over the shown stretch of time (seconds)
  function fitHeight() {
    if (!T) return;
    const first = Math.max(0, Math.floor(view.x[0] * T.fps)), last = Math.min(T.cells[0].trace.length - 1, Math.ceil(view.x[1] * T.fps));
    let lo = Infinity, hi = -Infinity;
    for (const c of T.cells) {
      if (hidden.has(c.roi)) continue;
      for (let k = first; k <= last; k++) { const v = c.trace[k]; if (v < lo) lo = v; if (v > hi) hi = v; }
    }
    if (!(lo < hi)) { lo = 0; hi = 1; }
    const pad = (hi - lo) * 0.08;
    view.y = [lo - pad, hi + pad]; fitted = view.x.slice();
  }

  function drawTraces() {
    const [ctx, w, h] = setup(canvas);
    if (!T) { ctx.fillStyle = FG; ctx.font = "13px sans-serif"; ctx.fillText(selected === null ? "" : "Loading the traces...", PLOT.L, 30); return; }
    const a = PLOT, X = v => a.L + (v - view.x[0]) / (view.x[1] - view.x[0]) * (w - a.L - a.R), Y = v => a.T + (1 - (v - view.y[0]) / (view.y[1] - view.y[0])) * (h - a.T - a.B);
    ctx.font = "10px sans-serif"; ctx.textBaseline = "middle"; ctx.textAlign = "right";
    for (const v of TimeAxis.nice(view.y[0], view.y[1], 5)) { ctx.strokeStyle = GRID; ctx.beginPath(); ctx.moveTo(a.L, Y(v)); ctx.lineTo(w - a.R, Y(v)); ctx.stroke(); ctx.fillStyle = FG; ctx.fillText(fmt(v), a.L - 4, Y(v)); }
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = FG;
    for (const v of TimeAxis.nice(view.x[0], view.x[1], 8)) ctx.fillText(fmt(v), X(v), h - a.B + 4);
    ctx.fillText("seconds", a.L + (w - a.L - a.R) / 2, h - 13);
    ctx.save(); ctx.translate(11, a.T + (h - a.T - a.B) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("dF/F (%)", 0, 0); ctx.restore();
    ctx.save(); ctx.beginPath(); ctx.rect(a.L, a.T, w - a.L - a.R, h - a.T - a.B); ctx.clip();
    const first = Math.max(0, Math.floor(view.x[0] * T.fps) - 1), last = Math.min(T.cells[0].trace.length - 1, Math.ceil(view.x[1] * T.fps) + 1);
    ctx.lineWidth = 1.2;
    T.cells.forEach((c, i) => {
      if (hidden.has(c.roi)) return;
      ctx.strokeStyle = traceColor(c, i); ctx.lineWidth = c.merged ? 2 : 1.2; ctx.beginPath();
      for (let k = first; k <= last; k++) k === first ? ctx.moveTo(X(k / T.fps), Y(c.trace[k])) : ctx.lineTo(X(k / T.fps), Y(c.trace[k]));
      ctx.stroke();
    });
    ctx.restore();
  }

  async function request(path, options) {
    const res = await fetch(`${URL}/${path}`, options), body = await res.json();
    if (!res.ok) throw new Error(body.message || `Request failed (${res.status})`);
    return body;
  }

  async function select(id) {
    selected = id; T = null; hidden = new Set();
    canvas.hidden = id === null;
    drawOverlay(); drawMembers(); drawTraces();
    if (id === null) return;
    $("rg-group").value = id;
    try {
      const traces = await request(`roi-groups/${id}/traces.json`);
      if (selected !== id) return;  // another group was chosen meanwhile
      T = traces;
      drawMembers();
      const seconds = (T.cells[0].trace.length - 1) / T.fps;
      if (!view) {  // one view for the canvas, made with the first group: the zoom and pan are wired to it once
        view = PlotView.create({x: [0, seconds], y: [0, 1]}, v => { if (v.x[0] !== fitted[0] || v.x[1] !== fitted[1]) fitHeight(); drawTraces(); });  // a change of the time refits the height; a change of the height alone is kept
        PlotView.attach(view, canvas, () => PLOT);
      }
      view.base = {x: [0, seconds], y: [0, 1]}; view.x = [0, seconds];
      fitHeight(); view.base.y = view.y.slice();
      drawTraces();
    } catch (error) { $("rg-summary").textContent = error.message; }
  }

  function setPayload(payload) {
    D = payload;
    owner = new Int32Array(W * H).fill(-1);
    D.rois.forEach((r, k) => r.ypix.forEach((y, i) => { owner[y * W + r.xpix[i]] = k; }));
    if (selected !== null && !D.groups.some(g => g.id === selected)) selected = null;
    if (selected === null && D.groups.length) selected = 0;
    drawSummary();
    select(selected);
  }

  async function find() {
    const threshold = Number($("rg-threshold").value);
    $("rg-find").disabled = true; $("rg-summary").textContent = "Finding groups...";
    try { selected = null; setPayload(await request("roi-groups/find", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({threshold})})); }
    catch (error) { $("rg-summary").textContent = error.message; }
    $("rg-find").disabled = false;
  }

  const host = stage.parentNode;
  host.addEventListener("pointerdown", e => { down = [e.clientX, e.clientY]; });
  host.addEventListener("click", e => {  // a click, not the end of a drag; not a click on the zoom buttons
    if (!D || e.target.closest(".zv-bar") || (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4)) return;
    const box = ovl.getBoundingClientRect(), x = Math.floor((e.clientX - box.left) / box.width * W), y = Math.floor((e.clientY - box.top) / box.height * H);
    const k = x >= 0 && y >= 0 && x < W && y < H ? owner[y * W + x] : -1;
    if (k >= 0 && D.rois[k].group !== null) select(D.rois[k].group);
    else if (k >= 0) $("rg-summary").textContent = `Cell ${D.rois[k].cell} (ROI ${D.rois[k].roi}) is in no group.`;
  });
  $("rg-group").addEventListener("change", e => select(Number(e.target.value)));
  $("rg-find").addEventListener("click", find);

  // confirming merges the groups into single ROIs in the Suite2p result (a backup is kept); undoing puts it back
  async function change(path, question) {
    if (!confirm(question)) return;
    for (const id of ["rg-find", "rg-confirm", "rg-undo"]) $(id).disabled = true;
    try { selected = null; setPayload(await request(path, {method: "POST"})); }
    catch (error) { $("rg-summary").textContent = error.message; }
    for (const id of ["rg-find", "rg-confirm", "rg-undo"]) $(id).disabled = false;
  }
  $("rg-confirm").addEventListener("click", () => change("roi-groups/confirm", `Merge the ${D.groups.length} group(s) into single ROIs? In the Suite2p result each group becomes one ROI (its members stop being cells; its trace is the average of theirs), like the merge of the Suite2p GUI. The files as they are now are kept in a backup folder and the merge can be undone here. It changes the cell numbers: do it before Confirm Suite2p.`));
  $("rg-undo").addEventListener("click", () => change("roi-groups/undo", "Undo the merge? The Suite2p result goes back to what it was before the groups were merged."));
  window.addEventListener("resize", () => { if (T) drawTraces(); });

  function load() {
    loaded = true;
    image.onload = () => {
      W = image.naturalWidth; H = image.naturalHeight;
      for (const c of [base, ovl]) { c.width = W * S; c.height = H * S; }
      stage.style.width = W * S + "px"; stage.style.height = H * S + "px";
      base.getContext("2d").drawImage(image, 0, 0, W * S, H * S);
      request("roi-groups.json").then(payload => { setPayload(payload); stage.parentNode._zv.fit(); }).catch(error => { $("rg-summary").textContent = error.message; });
    };
    image.onerror = () => { $("rg-summary").textContent = "The field of view could not be loaded: the mean image of the neuron channel is missing (the motion correction step saves it)."; };
    image.src = `${URL}/roi-fov.png`;
  }
  document.addEventListener("pre-tab", e => { if (e.detail === "groups" && !loaded) load(); });
})();
