(function () {
  const EXP_ID = JSON.parse(document.getElementById("page-data").textContent).expId;
  const KEEP_BEHIND = 1, KEEP_AHEAD = 3, PREFETCH = 2;  // chunks kept in memory around the current one, and chunks asked for ahead of the playing

  // The corrected movie is played from chunks of about 2 seconds that the page asks for one after the other, a little ahead of the playing,
  // so playing costs about one request every 2 seconds. Frames are drawn here; nothing is asked for per frame.
  function initPreprocessingMovieViewer(root, kind) {
    const canvas = root.querySelector("[data-movie-canvas]");
    const ctx = canvas.getContext("2d");
    const summary = root.querySelector("[data-movie-summary]");
    const message = root.querySelector("[data-movie-message]");
    const play = root.querySelector("[data-movie-play]");
    const scrub = root.querySelector("[data-movie-scrub]");
    const frameInput = root.querySelector("[data-movie-frame-input]");
    const readout = root.querySelector("[data-movie-readout]");
    const loadState = root.querySelector("[data-movie-load-state]");
    const state = {meta: null, image: null, chunks: new Map(), pending: new Map(), frame: 0, playing: false, started: 0, startFrame: 0, wanted: 0, timer: null};

    function setMessage(text) {
      message.textContent = text || "";
      message.style.display = text ? "grid" : "none";
    }
    function formatTime(seconds) {
      return seconds >= 60 ? `${(seconds / 60).toFixed(1)} min` : `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
    }
    const chunkOf = frame => Math.floor(frame / state.meta.chunk_frames);

    function evict(centre) {
      for (const index of [...state.chunks.keys()]) if (index < centre - KEEP_BEHIND || index > centre + KEEP_AHEAD) state.chunks.delete(index);
    }
    async function loadChunk(index) {
      if (state.chunks.has(index)) return;
      if (index < 0 || index * state.meta.chunk_frames >= state.meta.frames) return;
      if (!state.pending.has(index)) {
        const request = fetch(`/experiments/${EXP_ID}/movie/chunk?kind=${encodeURIComponent(kind)}&chunk=${index}`)
          .then(async response => {
            if (!response.ok) throw new Error((await response.json()).error || `The movie chunk ${index} could not be loaded.`);
            state.chunks.set(index, new Uint8Array(await response.arrayBuffer()));
          })
          .finally(() => state.pending.delete(index));
        state.pending.set(index, request);
      }
      await state.pending.get(index);
    }
    function prefetch(frame) {
      const centre = chunkOf(frame);
      evict(centre);
      for (let ahead = 1; ahead <= PREFETCH; ahead++) loadChunk(centre + ahead).catch(() => {});
    }
    function paint(frame) {
      const {width, height, chunk_frames: perChunk} = state.meta, size = width * height;
      const pixels = state.chunks.get(chunkOf(frame)), offset = (frame % perChunk) * size, data = state.image.data;
      for (let i = 0; i < size; i++) {
        data[4 * i] = data[4 * i + 1] = data[4 * i + 2] = pixels[offset + i];
        data[4 * i + 3] = 255;
      }
      ctx.putImageData(state.image, 0, 0);
      state.frame = frame;
      readout.textContent = `Frame ${frame + 1} / ${state.meta.frames} | ${formatTime(frame / state.meta.fps)}`;
      scrub.value = frame;
      frameInput.value = frame;
    }
    // show a frame: at once when its chunk is in memory, else after the chunk arrived (a short wait first, so dragging the slider does not ask for every chunk it passes)
    function show(frame) {
      frame = Math.max(0, Math.min(state.meta.frames - 1, Math.round(Number(frame) || 0)));
      state.wanted = frame;
      window.clearTimeout(state.timer);
      if (state.chunks.has(chunkOf(frame))) { paint(frame); prefetch(frame); return; }
      readout.textContent = `Frame ${frame + 1} / ${state.meta.frames}`;
      state.timer = window.setTimeout(async () => {
        loadState.textContent = "Loading...";
        try {
          await loadChunk(chunkOf(frame));
          if (state.wanted === frame) { paint(frame); prefetch(frame); setMessage(""); }
        } catch (error) {
          setMessage(error.message);
        }
        loadState.textContent = "";
      }, 120);
    }
    function stop() {
      state.playing = false;
      play.textContent = "Play";
    }
    function tick(now) {
      if (!state.playing) return;
      const next = Math.min(state.meta.frames - 1, state.startFrame + Math.floor((now - state.started) / 1000 * state.meta.fps));
      if (next !== state.frame) {
        if (!state.chunks.has(chunkOf(next))) {  // the next chunk is not here yet: wait for it, then go on from this frame
          loadState.textContent = "Loading...";
          loadChunk(chunkOf(next)).then(() => {
            loadState.textContent = "";
            state.started = performance.now();
            state.startFrame = next;
            window.requestAnimationFrame(tick);
          }).catch(error => { stop(); setMessage(error.message); loadState.textContent = ""; });
          return;
        }
        paint(next);
        prefetch(next);
      }
      if (next >= state.meta.frames - 1) stop();
      else window.requestAnimationFrame(tick);
    }
    play.addEventListener("click", () => {
      if (!state.meta) return;
      if (state.playing) { stop(); return; }
      if (state.frame >= state.meta.frames - 1) show(0);
      state.playing = true;
      play.textContent = "Pause";
      state.started = performance.now();
      state.startFrame = state.frame;
      window.requestAnimationFrame(tick);
    });
    scrub.addEventListener("input", () => { stop(); show(Number(scrub.value)); });
    frameInput.addEventListener("change", () => { stop(); show(Number(frameInput.value)); });

    async function load() {
      setMessage("Loading movie...");
      try {
        const response = await fetch(`/experiments/${EXP_ID}/movie/meta?kind=${encodeURIComponent(kind)}`);
        const meta = await response.json();
        if (!response.ok || !meta.available) throw new Error(meta.message || "Movie is unavailable.");
        state.meta = meta;
        canvas.width = meta.width;
        canvas.height = meta.height;
        state.image = ctx.createImageData(meta.width, meta.height);
        scrub.max = meta.frames - 1;
        frameInput.max = meta.frames - 1;
        summary.textContent = `${meta.file_name} | ${meta.frames} frames | ${meta.width} x ${meta.height} | ${Number(meta.fps).toFixed(1)} fps`;
        await loadChunk(0);
        setMessage("");
        show(0);
      } catch (error) {
        summary.textContent = error.message || "Movie is unavailable.";
        setMessage(error.message || "Movie is unavailable.");
      }
    }
    load();
  }

  function resizeCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.max(900, Math.round(rect.width * dpr));
    canvas.height = Math.max(320, Math.round(rect.height * dpr));
    return {width: canvas.width, height: canvas.height, dpr};
  }
  function drawMeanIntensity(payload) {
    const canvas = document.getElementById("intensity-canvas");
    const ctx = canvas.getContext("2d");
    const {width, height} = resizeCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    const pad = {left: 70, right: 22, top: 24, bottom: 54};
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const series = ["astro", "neuron"].map(kind => ({kind, color: kind === "astro" ? "#37c6a6" : "#4aa3ff", data: payload.series[kind]})).filter(s => s.data?.available);
    const allT = series.flatMap(s => s.data.time_s || []);
    const allY = series.flatMap(s => s.data.mean || []).filter(Number.isFinite);
    if (!series.length || !allT.length || !allY.length) {
      ctx.fillStyle = "#9aa8b7";
      ctx.font = "14px Segoe UI";
      ctx.fillText("Mean intensity data are unavailable.", pad.left, pad.top + 20);
      return;
    }
    const xMin = 0;
    const xMax = Math.max(...allT, 1);
    let yMin = Math.min(...allY);
    let yMax = Math.max(...allY);
    const yPad = (yMax - yMin) * 0.08 || 1;
    yMin -= yPad;
    yMax += yPad;
    const xScale = x => pad.left + ((x - xMin) / (xMax - xMin)) * plotW;
    const yScale = y => pad.top + (1 - ((y - yMin) / (yMax - yMin))) * plotH;
    ctx.fillStyle = "#0b0f14";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "#26303a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 5; i++) {
      const y = pad.top + (i / 5) * plotH;
      ctx.moveTo(pad.left, y);
      ctx.lineTo(width - pad.right, y);
    }
    for (let i = 0; i <= 6; i++) {
      const x = pad.left + (i / 6) * plotW;
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, height - pad.bottom);
    }
    ctx.stroke();
    ctx.strokeStyle = "#748292";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(pad.left, pad.top, plotW, plotH);
    ctx.fillStyle = "#9aa8b7";
    ctx.font = "12px Segoe UI";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i <= 5; i++) {
      const value = yMax - (i / 5) * (yMax - yMin);
      ctx.fillText(value.toFixed(0), pad.left - 10, pad.top + (i / 5) * plotH);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i <= 6; i++) {
      const value = xMin + (i / 6) * (xMax - xMin);
      const label = xMax >= 120 ? `${(value / 60).toFixed(1)}m` : `${value.toFixed(0)}s`;
      ctx.fillText(label, pad.left + (i / 6) * plotW, height - pad.bottom + 12);
    }
    ctx.save();
    ctx.translate(18, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("Mean intensity (a.u.)", 0, 0);
    ctx.restore();
    ctx.textAlign = "center";
    ctx.fillText("Time", pad.left + plotW / 2, height - 20);
    for (const s of series) {
      const t = s.data.time_s;
      const y = s.data.mean;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < t.length; i++) {
        if (!Number.isFinite(y[i])) continue;
        const px = xScale(t[i]);
        const py = yScale(y[i]);
        if (!started) {
          ctx.moveTo(px, py);
          started = true;
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.stroke();
    }
  }
  function formatMetricName(name) {
    return name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()).replace("Dff", "dF/F").replace("R2", "R2");
  }
  function renderQcMetrics(payload) {
    const box = document.getElementById("qc-metrics");
    box.innerHTML = "";
    for (const kind of ["astro", "neuron"]) {
      const qc = payload.series[kind]?.qc || {};
      const wanted = ["bleach_tau", "bleach_r2", "residual_slope_post_correction", "global_dff_std", "corrected_mean_intensity"];
      for (const key of wanted) {
        if (!(key in qc)) continue;
        const value = Number(qc[key]);
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = `${kind}: ${formatMetricName(key)} ${Number.isFinite(value) ? value.toPrecision(4) : qc[key]}`;
        box.appendChild(chip);
      }
    }
  }
  let meanIntensity = null;
  async function loadMeanIntensity() {
    const summary = document.getElementById("intensity-summary");
    try {
      const res = await fetch(`/experiments/${EXP_ID}/preprocessing/mean-intensity/data`);
      const data = await res.json();
      if (!res.ok || !data.available) throw new Error(data.message || "Mean intensity data are unavailable.");
      const available = Object.values(data.series || {}).filter(s => s.available);
      summary.textContent = available.map(s => `${s.file_name}: ${s.frames} frames, showing every ${s.display_step}`).join(" | ");
      meanIntensity = data;
      drawMeanIntensity(data);
      renderQcMetrics(data);
    } catch (error) {
      summary.textContent = error.message || "Mean intensity data are unavailable.";
      for (const id of ["intensity-canvas", "qc-metrics"]) document.getElementById(id).style.display = "none";  // nothing to draw
      document.querySelector(".intensity-legend").style.display = "none";
    }
  }
  document.querySelectorAll("[data-movie-viewer]").forEach(root => initPreprocessingMovieViewer(root, root.dataset.movieViewer));
  loadMeanIntensity();
  window.addEventListener("resize", () => { if (meanIntensity) drawMeanIntensity(meanIntensity); });
})();
