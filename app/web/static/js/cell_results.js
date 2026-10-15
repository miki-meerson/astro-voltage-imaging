// The change-from-baseline plot of every result-type card on the Cells tab, drawn from the data the page carries (#cards-data).
// Plots keep a dark background in both themes; text uses the page font; the sign colours come from the design tokens (--dep, --hyper).
(() => {
  const dataElement = document.getElementById("cards-data");
  if (!dataElement) return;
  const cards = JSON.parse(dataElement.textContent);
  const token = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const FONT = '12px "Segoe UI", system-ui, sans-serif';
  const PAD = {left: 52, right: 14, top: 12, bottom: 34};

  function niceTicks(lo, hi, count) {
    const raw = (hi - lo) / Math.max(1, count - 1);
    const power = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map(m => m * power).find(s => s >= raw) || raw;
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 0.5; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return ticks;
  }

  function extent(card) {
    const times = card.clusters.flatMap(c => c.t);
    return times.length ? [Math.min(...times), Math.max(...times)] : card.tested_window;
  }

  function draw(canvas, card) {
    const ratio = window.devicePixelRatio || 1, box = canvas.getBoundingClientRect();
    canvas.width = Math.round(box.width * ratio);
    canvas.height = Math.round(box.height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const width = box.width, height = box.height;
    const plot = {x0: PAD.left, y0: PAD.top, x1: width - PAD.right, y1: height - PAD.bottom};
    const [t0, t1] = extent(card);
    const peak = Math.max(0.005, ...card.clusters.flatMap(c => c.trace.map(Math.abs)));
    const limit = card.clusters.length ? peak * 1.15 : 0.05;
    const x = t => plot.x0 + (t - t0) / Math.max(t1 - t0, 1e-6) * (plot.x1 - plot.x0);
    const y = v => plot.y1 - (v + limit) / (2 * limit) * (plot.y1 - plot.y0);

    ctx.clearRect(0, 0, width, height);
    ctx.font = FONT;
    const [w0, w1] = card.tested_window;
    if (w0 > t0 || w1 < t1) {  // the plot is wider than the tested window: shade the tested frames
      ctx.fillStyle = "#16212c";
      ctx.fillRect(x(Math.max(w0, t0)), plot.y0, x(Math.min(w1, t1)) - x(Math.max(w0, t0)), plot.y1 - plot.y0);
      ctx.fillStyle = "#6f8296";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText("tested", x(Math.max(w0, t0)) + 4, plot.y0 + 3);
    }
    ctx.strokeStyle = "#26303a";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#aeb9c6";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const v of niceTicks(-limit, limit, 5)) {
      ctx.beginPath(); ctx.moveTo(plot.x0, y(v)); ctx.lineTo(plot.x1, y(v)); ctx.stroke();
      const pct = v * 100;
      ctx.fillText(`${Math.abs(pct) >= 10 || pct === 0 ? pct.toFixed(0) : pct.toFixed(1)}%`, plot.x0 - 8, y(v));
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const t of niceTicks(t0, t1, 5)) ctx.fillText(String(t), x(t), plot.y1 + 6);
    ctx.fillText("frames from the spike", (plot.x0 + plot.x1) / 2, plot.y1 + 20);
    if (t0 <= 0 && 0 <= t1) {
      ctx.strokeStyle = "#5d6a78";
      ctx.beginPath(); ctx.moveTo(x(0), plot.y0); ctx.lineTo(x(0), plot.y1); ctx.stroke();
    }
    if (!card.clusters.length) {
      ctx.fillStyle = "#7f8c9b";
      ctx.textBaseline = "middle";
      ctx.fillText("no significant clusters", (plot.x0 + plot.x1) / 2, (plot.y0 + plot.y1) / 2);
    }
    ctx.lineWidth = 2;
    for (const cluster of card.clusters) {
      ctx.strokeStyle = token(cluster.polarity === "depolarizing" ? "--dep" : "--hyper");
      ctx.beginPath();
      cluster.t.forEach((t, i) => (i ? ctx.lineTo(x(t), y(cluster.trace[i])) : ctx.moveTo(x(t), y(cluster.trace[i]))));
      ctx.stroke();
    }
  }

  const redraw = () => document.querySelectorAll("canvas.result-plot").forEach(canvas => draw(canvas, cards[Number(canvas.dataset.card)]));
  redraw();
  window.addEventListener("resize", redraw);
})();
