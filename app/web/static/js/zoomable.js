// Zoom / pan / fit / 1:1 / fullscreen for every .zv box (images or the movie stage). Exposes zoomable(host) and sets host._zv = {fit}.
// ---- zoom / pan / fit / 1:1 / fullscreen for every .zv box -------------------------------------------------------
function zoomable(host) {
  const content = host.firstElementChild;
  const st = {k: 1, x: 0, y: 0, fit: true};
  const nat = () => content.tagName === "IMG" ? [content.naturalWidth, content.naturalHeight] : [content.offsetWidth, content.offsetHeight];
  const apply = () => { content.style.transform = `translate(${st.x}px,${st.y}px) scale(${st.k})`; };
  let autoH = true, setH = null;
  function sizeBox(w, h) {  // height follows the picture's aspect ratio at the available width, capped to the window
    if (!autoH || document.fullscreenElement === host || !host.clientWidth) return;
    setH = Math.max(160, Math.round(Math.min(host.clientWidth * h / w, window.innerHeight * parseFloat(host.dataset.cap || "0.8"))));
    host.style.height = setH + "px";
  }
  function fit() {
    const [w, h] = nat();
    if (!w || !h) return;
    sizeBox(w, h);
    const bw = host.clientWidth, bh = host.clientHeight;
    if (!bw || !bh) return;
    st.k = Math.min(bw / w, bh / h); st.x = (bw - w * st.k) / 2; st.y = (bh - h * st.k) / 2; st.fit = true; apply();
  }
  function zoomAt(f, cx, cy) {
    const nk = Math.max(0.05, Math.min(60, st.k * f)), r = nk / st.k;
    st.x = cx - (cx - st.x) * r; st.y = cy - (cy - st.y) * r; st.k = nk; st.fit = false; apply();
  }
  function actual() {
    const [w, h] = nat(); st.k = 1; st.x = (host.clientWidth - w) / 2; st.y = (host.clientHeight - h) / 2; st.fit = false; apply();
  }
  const bar = document.createElement("div"); bar.className = "zv-bar";
  [["Fit", "Fit to window (autoscale)", fit], ["1:1", "Actual size", actual],
   ["+", "Zoom in", () => zoomAt(1.4, host.clientWidth / 2, host.clientHeight / 2)],
   ["\u2212", "Zoom out", () => zoomAt(1 / 1.4, host.clientWidth / 2, host.clientHeight / 2)],
   ["\u26F6", "Full screen", () => { if (document.fullscreenElement === host) document.exitFullscreen(); else host.requestFullscreen && host.requestFullscreen(); }],
  ].forEach(([label, title, fn]) => { const b = document.createElement("button"); b.type = "button"; b.textContent = label; b.title = title; b.addEventListener("click", e => { e.stopPropagation(); fn(); }); bar.appendChild(b); });
  host.appendChild(bar);
  // plain wheel scrolls the page as usual; zoom with the buttons, or hold Ctrl while using the wheel
  host.addEventListener("wheel", e => { if (!e.ctrlKey) return; e.preventDefault(); const r = host.getBoundingClientRect(); zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top); }, {passive: false});
  let drag = null;
  host.addEventListener("pointerdown", e => { if (e.target.closest(".zv-bar")) return; drag = {x: e.clientX, y: e.clientY}; host.setPointerCapture(e.pointerId); });
  host.addEventListener("pointermove", e => { if (!drag) return; st.x += e.clientX - drag.x; st.y += e.clientY - drag.y; drag = {x: e.clientX, y: e.clientY}; st.fit = false; apply(); });
  host.addEventListener("pointerup", () => { drag = null; });
  host.addEventListener("dblclick", fit);
  new ResizeObserver(() => {
    if (setH !== null && Math.abs(host.clientHeight - setH) > 3 && document.fullscreenElement !== host) autoH = false;  // the user dragged the corner
    if (st.fit) fit();
  }).observe(host);
  if (content.tagName === "IMG") { content.addEventListener("load", fit); if (content.complete && content.naturalWidth) fit(); }
  host._zv = {fit};
}

document.querySelectorAll(".zv").forEach(zoomable);
