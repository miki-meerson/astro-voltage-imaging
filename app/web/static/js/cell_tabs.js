// The tabs of the Cells page (movie and components, clusters, motion, spikes) and the result type filter of the clusters tab. A tab that was hidden has canvases without a size: showing it
// sends a resize so every plot of the page draws itself again at its real size.
(function () {
  const tabs = document.getElementById("cell-tabs");
  if (!tabs) return;
  const buttons = [...tabs.querySelectorAll("button[data-tab]")];
  function show(name) {
    buttons.forEach(b => b.classList.toggle("on", b.dataset.tab === name));
    document.querySelectorAll("[data-pane]").forEach(pane => { pane.hidden = pane.dataset.pane !== name; });
    window.dispatchEvent(new Event("resize"));
  }
  tabs.addEventListener("click", e => {
    const button = e.target.closest("button[data-tab]");
    if (!button) return;
    history.replaceState(null, "", `#${button.dataset.tab}`);
    show(button.dataset.tab);
  });
  const wanted = location.hash.slice(1);
  show(buttons.some(b => b.dataset.tab === wanted) ? wanted : buttons[0].dataset.tab);

  const type = document.getElementById("cluster-type");
  if (type) type.addEventListener("change", () => {
    document.querySelectorAll('[data-pane="clusters"] .card[data-result]').forEach(card => { card.hidden = !!type.value && card.dataset.result !== type.value; });
    window.dispatchEvent(new Event("resize"));
  });
})();
