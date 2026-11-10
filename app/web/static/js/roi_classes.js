// The ROI types tab of the Preprocessing page: the field of view with the accepted (iscell = 1) Suite2p ROIs outlined by their soma / dendrite label, a legend,
// zoom and pan (zoomable.js), and click-to-correct. What is shown is what the server saved (app/services/roi_classes.py); a click saves a manual label at once.
(function () {
  const root = document.getElementById("roi-panel");
  if (!root) return;
  const EXP_ID = JSON.parse(document.getElementById("page-data").textContent).expId;
  const URL = `/experiments/${EXP_ID}/preprocessing`;
  const $ = id => document.getElementById(id);
  const S = 2;  // on-screen pixels per field-of-view pixel
  const UNCLASSIFIED = {name: "unclassified", label: "Not classified", color: "#9aa8b7"};
  const base = $("roi-base"), ovl = $("roi-ovl"), stage = $("roi-stage"), image = new Image();
  let D = null, byRoi = new Map(), owner = null, W = 0, H = 0, selected = null, loaded = false, down = null;

  const classOf = name => D.classes.find(c => c.name === name) || UNCLASSIFIED;
  const colorOf = entry => classOf(entry.label).color;

  function outline(ctx, entry, width) {
    ctx.lineWidth = width;
    FovOutline.draw(ctx, entry.ypix, entry.xpix, W, S);
  }

  function drawOverlay() {
    const ctx = ovl.getContext("2d");
    ctx.clearRect(0, 0, ovl.width, ovl.height);
    for (const entry of D.rois) {
      ctx.strokeStyle = colorOf(entry);
      outline(ctx, entry, entry.manual ? 3 : 1.5);  // a bold outline: set by hand
    }
    if (selected !== null) {
      const entry = byRoi.get(selected);
      ctx.fillStyle = "rgba(255,255,255,0.28)";
      entry.ypix.forEach((y, i) => ctx.fillRect(entry.xpix[i] * S, y * S, S, S));
      ctx.strokeStyle = "#ffffff";
      outline(ctx, entry, 3);
    }
  }

  function drawLegend() {
    const names = [...D.classes, UNCLASSIFIED].filter(c => D.counts[c.name]);
    $("roi-legend").innerHTML = (names.length ? names : [UNCLASSIFIED]).map(c => `<span class="roi-chip"><i style="background:${c.color}"></i>${c.label} <b>${D.counts[c.name] || 0}</b></span>`).join("") +
      `<span class="roi-chip muted small">bold outline = set by hand (${D.n_manual})</span>`;
    const t = D.thresholds;
    $("roi-summary").textContent = `${D.rois.length} accepted ROIs (iscell = 1) of this field of view. ` + (D.auto_available
      ? `Automatic rule: soma when elongation <= ${t.soma_max_elongation} and solidity >= ${t.soma_min_solidity}; dendrite when elongation >= ${t.dendrite_min_elongation} or solidity <= ${t.dendrite_max_solidity}; otherwise (and under ${t.min_pixels} pixels) uncertain.`
      : "No automatic classification is saved yet: run it below.") + (D.warning ? " " + D.warning : "");
  }

  function drawDetail() {
    const box = $("roi-detail");
    if (selected === null) { box.innerHTML = '<p class="muted">Click an ROI in the field of view to see why it got its label, and to set it by hand.</p>'; return; }
    const e = byRoi.get(selected), f = e.features;
    const auto = e.auto ? `${classOf(e.auto).label} (${e.why})` : "not classified";
    box.innerHTML = `<h3>ROI ${e.roi} <span class="muted small">(cell ${e.cell} in the analyses)</span></h3>
      <p>Label: <b style="color:${colorOf(e)}">${classOf(e.label).label}</b>${e.manual ? " (set by hand)" : ""}</p>
      <p class="muted small">Automatic: ${auto}</p>
      ${f ? `<table class="roi-features"><tr><td>pixels</td><td>${f.npix}</td></tr><tr><td>elongation</td><td>${f.elongation.toFixed(2)}</td></tr><tr><td>solidity</td><td>${f.solidity.toFixed(2)}</td></tr><tr><td>max width (px)</td><td>${f.max_width.toFixed(1)}</td></tr><tr><td>length (px)</td><td>${f.length.toFixed(1)}</td></tr></table>` : ""}
      <div class="toolbar">${D.classes.filter(c => D.manual_labels.includes(c.name)).map(c => `<button class="secondary" data-set="${c.name}" style="border-color:${c.color}">${c.label}</button>`).join("")}
        <button class="secondary" data-set="" ${e.manual ? "" : "disabled"} title="Remove the label set by hand: the automatic one applies again">Use automatic</button></div>
      <p id="roi-save-status" class="muted small"></p>`;
    box.querySelectorAll("[data-set]").forEach(button => button.addEventListener("click", () => setManual(e.roi, button.dataset.set || null)));
  }

  function refresh() { drawOverlay(); drawLegend(); drawDetail(); }

  function recount() {
    D.counts = {}; D.n_manual = 0;
    for (const e of D.rois) { const k = e.label || "unclassified"; D.counts[k] = (D.counts[k] || 0) + 1; D.n_manual += e.manual ? 1 : 0; }
  }

  function setPayload(payload) {
    D = payload; byRoi = new Map(D.rois.map(e => [e.roi, e]));
    owner = new Int32Array(W * H).fill(-1);
    for (const e of D.rois) e.ypix.forEach((y, i) => { owner[y * W + e.xpix[i]] = e.roi; });
    if (selected !== null && !byRoi.has(selected)) selected = null;
    refresh();
  }

  async function request(path, options) {
    const res = await fetch(`${URL}/${path}`, options), body = await res.json();
    if (!res.ok) throw new Error(body.message || `Request failed (${res.status})`);
    return body;
  }

  async function setManual(roi, label) {
    const status = $("roi-save-status");
    status.textContent = "Saving...";
    try {
      const saved = await request(`roi-classes/${roi}`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({label})});
      Object.assign(byRoi.get(roi), {label: saved.label, manual: saved.manual});
      recount(); refresh();
      $("roi-save-status").textContent = "Saved.";
    } catch (error) { status.textContent = error.message; }
  }

  async function runAuto() {
    if (D && D.n_manual && !confirm(`Run the automatic classification again? It replaces everything saved, including the ${D.n_manual} label(s) you set by hand.`)) return;
    $("roi-run").disabled = true; $("roi-summary").textContent = "Classifying...";
    try { setPayload(await request("roi-classes/auto", {method: "POST"})); }
    catch (error) { $("roi-summary").textContent = error.message; }
    $("roi-run").disabled = false;
  }

  async function exportLabels() {
    const status = $("roi-export-status");
    status.textContent = "Exporting...";
    try {
      const r = await request("roi-classes/export", {method: "POST"});
      status.textContent = r.exported ? "Added " + r.exported + " label(s) set by hand to the dataset (" + r.records + " labelled ROIs from " + r.experiments + " experiment(s)): " + r.path : "No label set by hand in this experiment: nothing was added.";
    } catch (error) { status.textContent = error.message; }
  }

  function load() {
    loaded = true;
    image.onload = () => {
      W = image.naturalWidth; H = image.naturalHeight;
      for (const canvas of [base, ovl]) { canvas.width = W * S; canvas.height = H * S; }
      stage.style.width = W * S + "px"; stage.style.height = H * S + "px";
      base.getContext("2d").drawImage(image, 0, 0, W * S, H * S);
      request("roi-classes.json").then(payload => { setPayload(payload); stage.parentNode._zv.fit(); })
        .catch(error => { $("roi-summary").textContent = error.message; });
    };
    image.onerror = () => { $("roi-summary").textContent = "The field of view could not be loaded: the mean image of the neuron channel is missing (the motion correction step saves it)."; };
    image.src = `${URL}/roi-fov.png`;
  }

  const host = stage.parentNode;  // zoomable.js captures the pointer on its box, so the box gets the clicks
  host.addEventListener("pointerdown", e => { down = [e.clientX, e.clientY]; });
  host.addEventListener("click", e => {  // a click, not the end of a drag (zoomable.js pans on drag); not a click on its buttons
    if (!D || e.target.closest(".zv-bar") || (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4)) return;
    const box = ovl.getBoundingClientRect(), x = Math.floor((e.clientX - box.left) / box.width * W), y = Math.floor((e.clientY - box.top) / box.height * H);
    const roi = x >= 0 && y >= 0 && x < W && y < H ? owner[y * W + x] : -1;
    selected = roi >= 0 ? roi : null;
    refresh();
  });
  $("roi-run").addEventListener("click", runAuto);
  $("roi-export").addEventListener("click", exportLabels);

  // the tabs of the page: the ROI types are loaded when their tab is first opened
  document.querySelectorAll("[data-pre-tab]").forEach(button => button.addEventListener("click", () => showTab(button.dataset.preTab)));
  function showTab(name) {
    $("pre-tab-main").hidden = name !== "main"; $("pre-tab-roi").hidden = name !== "roi";
    document.querySelectorAll("[data-pre-tab]").forEach(b => b.classList.toggle("active", b.dataset.preTab === name));
    if (name === "roi") { if (!loaded) load(); else stage.parentNode._zv.fit(); }
    history.replaceState(null, "", name === "roi" ? "#roi" : location.pathname + location.search);  // a reload keeps the tab
    if (window.autoRefresh) name === "roi" ? window.autoRefresh.stop() : window.autoRefresh.start();  // the page reloads itself while a run is active: not while the ROI types are being labelled
  }
  if (location.hash === "#roi") showTab("roi");
})();
