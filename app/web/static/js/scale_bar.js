// Lengths in micrometres, and the scale bar of every image. The size of one pixel of the recorded images (PIXEL_UM) comes from the page: the server reads it once per experiment from
// Experiment.xml (utils/movie_utils.py `read_pixel_size_um`) and writes it into <meta name="pixel-um">. An image binned by `bin` has pixels `bin` times as large. When the size could not be
// read the page says why and everything stays in pixels (UNIT "px"): no value is made up. The rules for the bar length are the same as in utils/scale_bar.py.
const ScaleBar = (() => {
  const meta = document.querySelector('meta[name="pixel-um"]');
  const PIXEL_UM = meta ? Number(meta.content) : null;
  const UNIT = PIXEL_UM ? "µm" : "px", AREA_UNIT = PIXEL_UM ? "µm²" : "px";
  const BAR_FRACTION = 0.2;

  // a length of `pixels` pixels of an image binned by `bin`, in micrometres (unchanged, in pixels, when the pixel size is not known)
  const length = (pixels, bin = 1) => PIXEL_UM ? pixels * bin * PIXEL_UM : pixels;
  // an area of `pixels` pixels of an image binned by `bin`, in square micrometres
  const area = (pixels, bin = 1) => PIXEL_UM ? pixels * (bin * PIXEL_UM) ** 2 : pixels;
  // micrometres per `bin`-fold binned pixel, null when not known
  const umPerPixel = (bin = 1) => PIXEL_UM ? PIXEL_UM * bin : null;

  function roundLength(widthUm) {  // 1, 2 or 5 x a power of ten, at most BAR_FRACTION of the width
    const target = widthUm * BAR_FRACTION, power = Math.pow(10, Math.floor(Math.log10(target)));
    return [5, 2, 1].map(step => step * power).find(v => v <= target * (1 + 1e-9));
  }
  const label = um => um >= 1000 ? `${um / 1000} mm` : `${um} µm`;

  // Draws the bar in the bottom left corner (`right`: bottom right) of a canvas of W x H css pixels whose pixels are `umPerCanvasPixel` micrometres wide. Draws nothing when the pixel size
  // is not known.
  function draw(ctx, W, H, umPerCanvasPixel, right = false) {
    if (!PIXEL_UM || !(umPerCanvasPixel > 0)) return;
    const um = roundLength(W * umPerCanvasPixel), len = um / umPerCanvasPixel, x = right ? W - 10 - len : 10, y = H - 12;
    ctx.save(); ctx.setLineDash([]); ctx.lineCap = "butt";
    for (const [style, width] of [["rgba(11,15,20,0.85)", 5], ["#ffffff", 2.5]]) { ctx.strokeStyle = style; ctx.lineWidth = width; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + len, y); ctx.stroke(); }
    ctx.font = "bold 11px sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.lineWidth = 3; ctx.strokeStyle = "rgba(11,15,20,0.85)"; ctx.fillStyle = "#ffffff";
    ctx.strokeText(label(um), x, y - 5); ctx.fillText(label(um), x, y - 5);
    ctx.restore();
  }

  return {PIXEL_UM, UNIT, AREA_UNIT, length, area, umPerPixel, roundLength, label, draw};
})();
