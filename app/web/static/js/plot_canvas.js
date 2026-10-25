// What the pages that draw their own plots on a canvas share: sizing the canvas and writing a number on an axis.
window.PlotCanvas = (() => {
  // Size the canvas to its box at the screen's pixel ratio and clear it. Returns [the context (draw in CSS pixels), width, height].
  function setup(canvas) {
    const ratio = window.devicePixelRatio || 1, box = canvas.getBoundingClientRect();
    canvas.width = Math.round(box.width * ratio); canvas.height = Math.round(box.height * ratio);
    const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, box.width, box.height);
    return [ctx, box.width, box.height];
  }

  // A number on an axis or in a readout: no decimals from 100 up (and for whole numbers), one from 1 up, two below.
  const format = v => Math.abs(v) >= 100 || Number.isInteger(v) ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2);

  return {setup, format};
})();
