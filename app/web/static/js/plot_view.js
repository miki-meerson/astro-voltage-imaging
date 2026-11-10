// Zoom and pan of a canvas plot, for plots the page draws itself. Time and height zoom independently:
//   the mouse wheel over the plot zooms both axes around the pointer; over the left (height) axis only the height; over the bottom (time) axis only the time;
//   with Shift held it zooms the time only, with Alt the height only;
//   dragging with Shift held draws a box and zooms to it (as wide and as tall as you drew it: a thin box zooms one axis);
//   dragging pans; a double click goes back to the whole plot.
// An image (options.inside) keeps its proportions: the wheel zooms both axes together and there is no box. A view is the range of each axis that is shown; several canvases can
// share one view, so they zoom together.
window.PlotView = (() => {
  const ZOOM = 1.25, MAX_ZOOM_OUT = 3;  // the zoom of one notch of a mouse wheel (100 units of scroll); how far past the data an axis can be zoomed out (in data spans) when it is not held inside the data
  const LINE_UNITS = 33, PAGE_UNITS = 100, MAX_NOTCH_UNITS = 300;  // a scroll in lines or pages in units of pixels; the most one event can zoom
  const DRAG_PIXELS = 4;  // a pointer that moved less than this is not yet a drag
  const MIN_BOX_PIXELS = 6;  // a box narrower (or lower) than this does not zoom that axis

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

  // how far a wheel event scrolled, in pixels (positive: zoom out). A trackpad sends many small events for one swipe and a pinch arrives as a wheel event with Ctrl held, so the
  // zoom follows the size of the event instead of taking one fixed step each.
  const scrolled = e => Math.max(-MAX_NOTCH_UNITS, Math.min(MAX_NOTCH_UNITS, e.deltaY * (e.deltaMode === 1 ? LINE_UNITS : e.deltaMode === 2 ? PAGE_UNITS : 1)));

  // area(width, height) -> {L, R, T, B}: the margins of the plot area inside the canvas. options.inside: both axes stay inside the data and keep their proportions (an image);
  // options.yDown: the vertical axis grows downward (an image)
  function attach(view, canvas, area, options = {}) {
    let drag = null;
    const geometry = () => {
      const box = canvas.getBoundingClientRect(), a = area(box.width, box.height);
      return {box, a, w: Math.max(1, box.width - a.L - a.R), h: Math.max(1, box.height - a.T - a.B)};
    };
    const unit = (value, size) => Math.max(0, Math.min(1, value / size));  // where in the plot area (0..1) a pixel is
    const xAt = f => view.x[0] + f * (view.x[1] - view.x[0]);
    const yAt = f => options.yDown ? view.y[0] + f * (view.y[1] - view.y[0]) : view.y[1] - f * (view.y[1] - view.y[0]);
    const boxFrame = document.createElement("div");  // the box being drawn, over the canvas
    boxFrame.style.cssText = "position:absolute;display:none;pointer-events:none;border:1px dashed #fff;background:rgba(255,255,255,0.12)";
    canvas.parentElement.style.position = "relative";
    canvas.parentElement.appendChild(boxFrame);
    canvas.style.touchAction = "none";

    canvas.addEventListener("wheel", e => {
      e.preventDefault();
      const {box, a, w, h} = geometry(), px = e.clientX - box.left, py = e.clientY - box.top, fx = unit(px - a.L, w), fy = unit(py - a.T, h), factor = Math.pow(ZOOM, scrolled(e) / 100);
      const onHeightAxis = px < a.L, onTimeAxis = py > box.height - a.B;  // the wheel over an axis zooms that axis
      const timeOnly = !options.inside && (e.shiftKey || onTimeAxis), heightOnly = !options.inside && (e.altKey || onHeightAxis);
      if (!heightOnly) view.x = zoomed(view.x, view.base.x, xAt(fx), factor, true);
      if (!timeOnly) view.y = zoomed(view.y, view.base.y, yAt(fy), factor, !!options.inside);
      view.onChange(view);
    }, {passive: false});

    canvas.addEventListener("pointerdown", e => {
      drag = {x: e.clientX, y: e.clientY, vx: view.x.slice(), vy: view.y.slice(), moved: false, zoomBox: e.shiftKey && !options.inside};
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", e => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y, {box, a, w, h} = geometry();
      if (Math.hypot(dx, dy) > DRAG_PIXELS) drag.moved = true;
      if (!drag.moved) return;
      if (drag.zoomBox) {  // the box, kept inside the plot area
        const x0 = Math.max(a.L, Math.min(box.width - a.R, Math.min(drag.x, e.clientX) - box.left)), x1 = Math.max(a.L, Math.min(box.width - a.R, Math.max(drag.x, e.clientX) - box.left));
        const y0 = Math.max(a.T, Math.min(box.height - a.B, Math.min(drag.y, e.clientY) - box.top)), y1 = Math.max(a.T, Math.min(box.height - a.B, Math.max(drag.y, e.clientY) - box.top));
        Object.assign(boxFrame.style, {display: "block", left: `${canvas.offsetLeft + x0}px`, top: `${canvas.offsetTop + y0}px`, width: `${x1 - x0}px`, height: `${y1 - y0}px`});
        return;
      }
      view.x = shifted(drag.vx, -dx / w * (drag.vx[1] - drag.vx[0]), view.base.x, true);
      view.y = shifted(drag.vy, (options.yDown ? -1 : 1) * dy / h * (drag.vy[1] - drag.vy[0]), view.base.y, !!options.inside);
      canvas.style.cursor = "grabbing";
      view.onChange(view);
    });
    const end = e => {
      if (drag && drag.zoomBox && drag.moved && e.type === "pointerup") {  // zoom to the box: each axis only when the box is wide (tall) enough
        const {box, a, w, h} = geometry(), fx = [drag.x, e.clientX].map(v => unit(v - box.left - a.L, w)), fy = [drag.y, e.clientY].map(v => unit(v - box.top - a.T, h));
        if (Math.abs(fx[1] - fx[0]) * w >= MIN_BOX_PIXELS) view.x = shifted([xAt(Math.min(...fx)), xAt(Math.max(...fx))], 0, view.base.x, true);
        if (Math.abs(fy[1] - fy[0]) * h >= MIN_BOX_PIXELS) view.y = [yAt(Math.max(...fy)), yAt(Math.min(...fy))].sort((p, q) => p - q);
        view.onChange(view);
      }
      boxFrame.style.display = "none";
      drag = null; canvas.style.cursor = "grab";
    };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);
    canvas.addEventListener("dblclick", () => { view.x = view.base.x.slice(); view.y = view.base.y.slice(); view.onChange(view); });
    canvas.style.cursor = "grab";
  }

  return {create, attach};
})();
