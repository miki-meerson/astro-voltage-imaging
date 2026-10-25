// Zoom and pan of a canvas plot, for plots the page draws itself: the mouse wheel zooms around the pointer (Shift: the horizontal axis only, Alt: the vertical axis only),
// dragging pans, a double click goes back to the whole plot (an image always zooms both axes together). A view is the range of each axis that is shown; several canvases can share one view, so they zoom together.
window.PlotView = (() => {
  const ZOOM = 1.25, MAX_ZOOM_OUT = 3;  // one wheel step; how far past the data an axis can be zoomed out (in data spans) when it is not held inside the data
  const DRAG_PIXELS = 4;  // a pointer that moved less than this is not yet a drag

  // base: {x: [lo, hi], y: [lo, hi]}, the whole plot; onChange(view): called after every change of the shown ranges (it redraws)
  function create(base, onChange) {
    return {base, x: base.x.slice(), y: base.y.slice(), onChange};
  }

  const shifted = (range, by, base, inside) => {  // `range` moved by `by`, kept inside `base` when asked
    let lo = range[0] + by, hi = range[1] + by;
    if (inside) { if (lo < base[0]) { hi += base[0] - lo; lo = base[0]; } if (hi > base[1]) { lo -= hi - base[1]; hi = base[1]; } }
    return [lo, hi];
  };
  const zoomed = (range, base, center, factor, inside) => {  // `range` scaled around `center`, no wider than the data (or MAX_ZOOM_OUT spans of it)
    const span = Math.min((range[1] - range[0]) * factor, (base[1] - base[0]) * (inside ? 1 : MAX_ZOOM_OUT)), lo = center - (center - range[0]) * span / (range[1] - range[0]);
    return shifted([lo, lo + span], 0, base, inside);
  };

  // area(width, height) -> {L, R, T, B}: the margins of the plot area inside the canvas. options.inside: both axes stay inside the data (an image); options.yDown: the vertical axis
  // grows downward (an image)
  function attach(view, canvas, area, options = {}) {
    let drag = null;
    const geometry = () => {
      const box = canvas.getBoundingClientRect(), a = area(box.width, box.height);
      return {box, a, w: Math.max(1, box.width - a.L - a.R), h: Math.max(1, box.height - a.T - a.B)};
    };
    const unit = (value, size) => Math.max(0, Math.min(1, value / size));  // where in the plot area (0..1) a pixel is
    canvas.style.touchAction = "none";
    canvas.addEventListener("wheel", e => {
      e.preventDefault();
      const {box, a, w, h} = geometry(), fx = unit(e.clientX - box.left - a.L, w), fy = unit(e.clientY - box.top - a.T, h), factor = e.deltaY < 0 ? 1 / ZOOM : ZOOM;
      const only = options.inside ? {} : {x: e.shiftKey, y: e.altKey};  // an image always zooms both axes together
      if (!only.y) view.x = zoomed(view.x, view.base.x, view.x[0] + fx * (view.x[1] - view.x[0]), factor, true);
      if (!only.x) view.y = zoomed(view.y, view.base.y, options.yDown ? view.y[0] + fy * (view.y[1] - view.y[0]) : view.y[1] - fy * (view.y[1] - view.y[0]), factor, !!options.inside);
      view.onChange(view);
    }, {passive: false});
    canvas.addEventListener("pointerdown", e => {
      drag = {x: e.clientX, y: e.clientY, vx: view.x.slice(), vy: view.y.slice(), moved: false};
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", e => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y, {w, h} = geometry();
      if (Math.hypot(dx, dy) > DRAG_PIXELS) drag.moved = true;
      if (!drag.moved) return;
      view.x = shifted(drag.vx, -dx / w * (drag.vx[1] - drag.vx[0]), view.base.x, true);
      view.y = shifted(drag.vy, (options.yDown ? -1 : 1) * dy / h * (drag.vy[1] - drag.vy[0]), view.base.y, !!options.inside);
      canvas.style.cursor = "grabbing";
      view.onChange(view);
    });
    const end = () => { drag = null; canvas.style.cursor = "grab"; };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);
    canvas.addEventListener("dblclick", () => { view.x = view.base.x.slice(); view.y = view.base.y.slice(); view.onChange(view); });
    canvas.style.cursor = "grab";
  }

  return {create, attach};
})();
