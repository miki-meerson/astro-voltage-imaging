(function () {
  const EXP_ID = JSON.parse(document.getElementById("page-data").textContent).expId;
  const neuronSelect = document.getElementById("neuron-select");
  const significantOnly = document.getElementById("significant-only");
  const anatomyOnly = document.getElementById("anatomy-only");
  const cellView =document.getElementById("cell-view");
  const cellFrame = document.getElementById("cell-frame");
  const reviewSummary = document.getElementById("review-summary");
  const currentCellChip = document.getElementById("current-cell-chip");
  const currentReviewChip = document.getElementById("current-review-chip");
  const currentPeakSummary = document.getElementById("current-peak-summary");
  const prevCellButton = document.getElementById("prev-cell");
  const nextCellButton = document.getElementById("next-cell");
  const nextUnreviewedButton = document.getElementById("next-unreviewed");
  const editCurrentSpikesButton = document.getElementById("edit-current-spikes");
  const markReviewedButton = document.getElementById("mark-reviewed");
  const reviewState = {items: {}};
  function neuronOptions() {
    return Array.from(neuronSelect.options).filter(option => option.value);
  }
  function visibleNeuronOptions() {
    return neuronOptions().filter(option => !option.hidden);
  }
  function selectedNeuronOption() {
    return neuronSelect.selectedOptions[0]?.value ? neuronSelect.selectedOptions[0] : null;
  }
  function reviewInfoFor(option) {
    return option ? (reviewState.items[option.value] || null) : null;
  }
  function updateReviewSummary() {
    const total = neuronOptions().length;
    const reviewed = neuronOptions().filter(option => option.dataset.reviewed === "1").length;
    const staText = reviewState.resultTypes?.length
      ? ` STA: ${reviewState.resultTypes.map(row => {
          const chance = row.expected_by_chance === undefined ? "" :
            ` (by chance alone ~${row.expected_by_chance.toFixed(0)} of ${row.cells_tested} cells; ${row.p_vs_chance >= 0.05 ? "NOT above chance, p=" : "above chance, p="}${row.p_vs_chance < 0.001 ? "<0.001" : row.p_vs_chance.toFixed(2)})`;
          return `${row.result_type}: ${row.neurons} neurons / ${row.clusters} clusters${chance}`;
        }).join("; ")}.`
      : "";
    const staleText = reviewState.staleCells?.length ? ` WARNING: spikes of ${reviewState.staleCells.length} cell(s) were edited after the STA ran (cells ${reviewState.staleCells.slice(0, 12).join(", ")}${reviewState.staleCells.length > 12 ? ", ..." : ""}): their STA results are out of date.` : "";
    reviewSummary.textContent = `${reviewed} / ${total} cells reviewed. Review peaks cell by cell, then rerun STA maps.${staText}${staleText}`;
  }
  function updateCellReviewUi() {
    const option = selectedNeuronOption();
    const options = visibleNeuronOptions();
    const index = option ? options.findIndex(item => item.value === option.value) : -1;
    const info = reviewInfoFor(option);
    currentCellChip.textContent = option ? option.dataset.baseLabel : "No Cell Selected";
    currentReviewChip.textContent = option?.dataset.reviewed === "1" ? "Reviewed" : "Unreviewed";
    currentReviewChip.classList.toggle("completed", option?.dataset.reviewed === "1");
    currentPeakSummary.textContent = info
      ? `${info.n_peaks} peaks (${info.n_simple} simple, ${info.n_complex} complex, ${info.n_excluded} excluded)`
      : "";
    prevCellButton.disabled = index <= 0;
    nextCellButton.disabled = index < 0 || index >= options.length - 1;
    nextUnreviewedButton.disabled = !neuronOptions().some(item => item.dataset.reviewed !== "1");
    editCurrentSpikesButton.disabled = !option;
    markReviewedButton.disabled = !option;
    updateReviewSummary();
  }
  function decorateNeuronOption(option) {
    const sigText = (option.dataset.significant === "1" ? " | significant" : "") + (option.dataset.anatomy === "motion" ? " | anatomy (motion?)" : option.dataset.anatomy === "1" ? " | anatomy" : "");
    const reviewedText = option.dataset.reviewed === "1" ? " | reviewed" : "";
    option.textContent = `${option.dataset.baseLabel}${reviewedText}${sigText}`;
  }
  function applyNeuronFilter() {
    const only = significantOnly.checked, anatomy = anatomyOnly.checked;
    for (const option of neuronSelect.options) {
      if (!option.value) continue;
      option.hidden = (only && option.dataset.significant !== "1") || (anatomy && !option.dataset.anatomy);
    }
    if (neuronSelect.selectedOptions[0]?.hidden) {
      neuronSelect.value = "";
      cellView.style.display = "none";
      cellFrame.removeAttribute("src");
    }
    updateCellReviewUi();
  }
  significantOnly.addEventListener("change", applyNeuronFilter);
  anatomyOnly.addEventListener("change", applyNeuronFilter);
  fetch(`/experiments/${EXP_ID}/anatomy-clusters.json`).then(r => r.json()).then(d => {
    if (!d.available) { anatomyOnly.disabled = true; anatomyOnly.parentElement.title = d.message || "No anatomy results"; return; }
    const mark = {};
    (d.cells || []).forEach(c => { mark[c] = "1"; });
    ((d.excluded || {}).motion_flagged || []).forEach(c => { mark[c] = "motion"; });
    ((d.excluded || {}).no_motion_diagnostics || []).forEach(c => { mark[c] = "1"; });
    for (const option of neuronSelect.options) {
      if (!option.value) continue;
      const m = mark[option.dataset.cellId];
      if (m) option.dataset.anatomy = m; else delete option.dataset.anatomy;
      decorateNeuronOption(option);
    }
    applyNeuronFilter();
  }).catch(() => {});
  fetch(`/experiments/${EXP_ID}/sta-significant-cells/data`).then(r => r.json()).then(data => {
    const cells = data.cells || {};
    reviewState.resultTypes = data.result_types || [];
    reviewState.staleCells = data.stale_cells || [];
    for (const option of neuronSelect.options) {
      if (!option.value) continue;
      const count = Number(cells[option.dataset.cellId] || 0);
      option.dataset.significant = count > 0 ? "1" : "0";
      decorateNeuronOption(option);
    }
    applyNeuronFilter();
  }).catch(() => {});
  async function refreshCellReviewState() {
    try {
      const res = await fetch(`/experiments/${EXP_ID}/cell-review/data`);
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
      reviewState.items = data.review || {};
      for (const option of neuronOptions()) {
        const info = reviewState.items[option.value];
        option.dataset.reviewed = info?.reviewed ? "1" : "0";
        decorateNeuronOption(option);
      }
      updateCellReviewUi();
    } catch (error) {
      reviewSummary.textContent = error.message || "Review state failed to load.";
    }
  }
  function selectNeuron(neuronId, mode = "view") {
    neuronSelect.value = neuronId || "";
    if (!neuronSelect.value) {
      cellView.style.display = "none";
      cellFrame.removeAttribute("src");
      updateCellReviewUi();
      return;
    }
    const url = `/experiments/${EXP_ID}?neuron=${neuronSelect.value}#cell-view`;
    history.replaceState(null, "", url);
    cellFrame.src = mode === "spikes" ? `/neurons/${neuronSelect.value}/spikes?embedded=1` : `/neurons/${neuronSelect.value}?embedded=1`;
    cellView.style.display = "";
    updateCellReviewUi();
  }
  function selectRelativeCell(delta) {
    const options = visibleNeuronOptions();
    const current = selectedNeuronOption();
    const index = current ? options.findIndex(option => option.value === current.value) : -1;
    const next = options[Math.max(0, Math.min(options.length - 1, index + delta))];
    if (next) selectNeuron(next.value);
  }
  function selectNextUnreviewed() {
    const options = visibleNeuronOptions();
    const current = selectedNeuronOption();
    const start = current ? Math.max(0, options.findIndex(option => option.value === current.value) + 1) : 0;
    const ordered = options.slice(start).concat(options.slice(0, start));
    const next = ordered.find(option => option.dataset.reviewed !== "1");
    if (next) selectNeuron(next.value, "spikes");
  }
  neuronSelect.addEventListener("change", () => selectNeuron(neuronSelect.value));
  prevCellButton.addEventListener("click", () => selectRelativeCell(-1));
  nextCellButton.addEventListener("click", () => selectRelativeCell(1));
  nextUnreviewedButton.addEventListener("click", selectNextUnreviewed);
  editCurrentSpikesButton.addEventListener("click", () => {
    if (neuronSelect.value) selectNeuron(neuronSelect.value, "spikes");
  });
  markReviewedButton.addEventListener("click", async () => {
    if (!neuronSelect.value) return;
    markReviewedButton.disabled = true;
    try {
      const res = await fetch(`/neurons/${neuronSelect.value}/review/mark`, {method: "POST"});
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
      await refreshCellReviewState();
      selectRelativeCell(1);
    } catch (error) {
      reviewSummary.textContent = error.message || "Mark reviewed failed.";
    } finally {
      markReviewedButton.disabled = false;
    }
  });
  cellFrame.addEventListener("load", () => {
    const src = cellFrame.getAttribute("src") || "";
    if (src.includes(`/neurons/${neuronSelect.value}?embedded=1`)) refreshCellReviewState();
  });
  refreshCellReviewState();
  applyNeuronFilter();
})();
