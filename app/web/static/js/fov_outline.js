// The outline of a set of field-of-view pixels (the sides of its pixels that touch a pixel outside the set), shared by the pages that draw ROIs and clusters on the field of view.
window.FovOutline = (() => {
  // ys, xs: the pixels; width: the width of the field of view in pixels; scale: canvas pixels per field-of-view pixel. Strokes the path with the context's current style.
  function draw(ctx, ys, xs, width, scale) {
    const inside = new Set(ys.map((y, i) => y * width + xs[i]));
    ctx.beginPath();
    ys.forEach((y, i) => {
      const x = xs[i];
      if (!inside.has((y - 1) * width + x)) { ctx.moveTo(x * scale, y * scale); ctx.lineTo((x + 1) * scale, y * scale); }
      if (!inside.has((y + 1) * width + x)) { ctx.moveTo(x * scale, (y + 1) * scale); ctx.lineTo((x + 1) * scale, (y + 1) * scale); }
      if (!inside.has(y * width + x - 1)) { ctx.moveTo(x * scale, y * scale); ctx.lineTo(x * scale, (y + 1) * scale); }
      if (!inside.has(y * width + x + 1)) { ctx.moveTo((x + 1) * scale, y * scale); ctx.lineTo((x + 1) * scale, (y + 1) * scale); }
    });
    ctx.stroke();
  }
  return {draw};
})();
