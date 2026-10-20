// The Border masking panel of the Overview tab: the same frames twice, as the motion correction leaves them and with the border left out of the STA test.
(function () {
  const panel = document.getElementById("border-panel");
  if (!panel) return;
  const EXP_ID = JSON.parse(document.getElementById("page-data").textContent).expId;
  const $ = id => document.getElementById(id);
  const raw = $("bm-raw"), masked = $("bm-masked"), scrub = $("bm-scrub"), play = $("bm-play");
  let D = null, frames = null, frame = 0, timer = null, playing = true;

  function draw(index) {
    frame = Math.max(0, Math.min(D.frames - 1, index));
    const n = D.width * D.height, off = frame * n;
    const image = new ImageData(D.width, D.height);
    for (let i = 0; i < n; i++) { const v = frames[off + i]; image.data[4 * i] = image.data[4 * i + 1] = image.data[4 * i + 2] = v; image.data[4 * i + 3] = 255; }
    const m = D.margins_shown, innerW = D.width - m.left - m.right, innerH = D.height - m.top - m.bottom;
    [raw, masked].forEach(canvas => canvas.getContext("2d").putImageData(image, 0, 0));
    const ctx = masked.getContext("2d");
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, D.width, m.top); ctx.fillRect(0, D.height - m.bottom, D.width, m.bottom);
    ctx.fillRect(0, 0, m.left, D.height); ctx.fillRect(D.width - m.right, 0, m.right, D.height);
    [raw, masked].forEach(canvas => {  // the inner edge of the masked border, on both pictures
      const c = canvas.getContext("2d"); c.setLineDash([6, 4]); c.strokeStyle = "#ffd24a"; c.lineWidth = Math.max(1, D.width / 256);
      c.strokeRect(m.left, m.top, innerW, innerH); c.setLineDash([]);
    });
    scrub.value = frame;
    const [dy, dx] = D.shifts[frame];
    $("bm-readout").textContent = `frame ${frame + 1}/${D.frames} | ${(D.start_time_s + frame / D.fps).toFixed(2)} s | shift of this frame: dy ${dy >= 0 ? "+" : ""}${dy.toFixed(1)} px, dx ${dx >= 0 ? "+" : ""}${dx.toFixed(1)} px`;
  }
  function start() { clearInterval(timer); timer = setInterval(() => draw((frame + 1) % D.frames), 1000 / Math.min(D.fps, 30)); play.textContent = "Pause"; playing = true; }
  function stop() { clearInterval(timer); timer = null; play.textContent = "Play"; playing = false; }
  play.addEventListener("click", () => (playing ? stop() : start()));
  scrub.addEventListener("input", () => { stop(); draw(Number(scrub.value)); });

  const fail = message => { $("bm-summary").textContent = message; panel.querySelectorAll("input, figure, #bm-readout, button").forEach(el => { el.style.display = "none"; }); };
  fetch(`/experiments/${EXP_ID}/border-snippets.json`)
    .then(async res => { const body = await res.json(); if (!res.ok) throw new Error(body.message || `Not available (${res.status})`); return body; })
    .then(async d => {
      D = d;
      const parts = await Promise.all(Array.from({length: d.chunks}, (_, k) => fetch(`/experiments/${EXP_ID}/movie/chunk?kind=${d.kind}&chunk=${d.first_chunk + k}`).then(r => { if (!r.ok) throw new Error(`The movie chunk ${d.first_chunk + k} could not be read (${r.status}).`); return r.arrayBuffer(); })));
      frames = new Uint8Array(parts.reduce((s, p) => s + p.byteLength, 0));
      let at = 0; parts.forEach(p => { frames.set(new Uint8Array(p), at); at += p.byteLength; });
      raw.width = masked.width = d.width; raw.height = masked.height = d.height;
      scrub.max = d.frames - 1;
      const mp = d.margins_px;
      $("bm-summary").textContent = `Left out of the test: ${mp.top} px at the top, ${mp.bottom} at the bottom, ${mp.left} at the left and ${mp.right} at the right (the largest shift toward each side, rounded up, plus the interpolation margin). Shown: ${(d.frames / d.fps).toFixed(1)} s from t = ${d.start_time_s} s; the largest shift of the recording is at t = ${d.peak_time_s} s.`;
      draw(d.peak_frame); start();
    })
    .catch(err => fail(err.message || "Not available."));
})();
