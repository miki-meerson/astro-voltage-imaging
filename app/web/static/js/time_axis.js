// Ticks of a time axis in seconds: every 100 ms when the axis spans 2 s or less, else the usual round steps of `nice` (the page's own tick function).
const TimeAxis = {
  nice(lo, hi, count) {  // round tick values between lo and hi, about `count` of them
    const span = hi - lo || 1, raw = span / count, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
    const step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag, out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.abs(v) < 1e-12 ? 0 : v);
    return out;
  },
  ticks(lo, hi, count, nice) {
    if (hi - lo > 2) return nice(lo, hi, count);
    const ticks = [];
    for (let i = Math.ceil(lo * 10 - 1e-9); i <= Math.floor(hi * 10 + 1e-9); i++) ticks.push(i / 10);
    return ticks;
  },
};
