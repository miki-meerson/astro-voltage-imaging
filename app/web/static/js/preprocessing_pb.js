(function () {
  const EXP_ID = JSON.parse(document.getElementById("page-data").textContent).expId;
  function initPreprocessingMovieViewer(root, kind) {
    const frames = [root.querySelector("[data-frame-a]"), root.querySelector("[data-frame-b]")];
    const summary = root.querySelector("[data-movie-summary]");
    const message = root.querySelector("[data-movie-message]");
    const play = root.querySelector("[data-movie-play]");
    const scrub = root.querySelector("[data-movie-scrub]");
    const frameInput = root.querySelector("[data-movie-frame-input]");
    const readout = root.querySelector("[data-movie-readout]");
    const loadState = root.querySelector("[data-movie-load-state]");
    const state = {meta: null, frame: 0, playing: false, timer: null, requestId: 0, cache: new Map(), pending: new Map(), activeBuffer: 0};
    function setMessage(text) {
      message.textContent = text || "";
      message.style.display = text ? "grid" : "none";
    }
    function formatTime(frame) {
      const fps = state.meta?.fps || 1;
      const seconds = frame / Math.max(1e-9, fps);
      return seconds >= 60 ? `${(seconds / 60).toFixed(1)} min` : `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
    }
    function frameUrl(frame) {
      return `/experiments/${EXP_ID}/movie/frame?kind=${encodeURIComponent(kind)}&frame=${frame}&max_size=760`;
    }
    function updateReadout() {
      const total = state.meta?.frames || 0;
      readout.textContent = total ? `Frame ${state.frame + 1} / ${total} | ${formatTime(state.frame)}` : "Frame 0 / 0";
      scrub.value = state.frame;
      frameInput.value = state.frame;
    }
    function trimCache() {
      const keep = new Set();
      for (let i = -5; i <= 10; i++) {
        const frame = state.frame + i;
        if (frame >= 0 && frame < (state.meta?.frames || 0)) keep.add(frame);
      }
      for (const [frame, url] of state.cache.entries()) {
        if (!keep.has(frame)) {
          URL.revokeObjectURL(url);
          state.cache.delete(frame);
        }
      }
    }
    async function fetchFrame(frame, requestId) {
      if (state.cache.has(frame)) return state.cache.get(frame);
      if (state.pending.has(frame)) return state.pending.get(frame);
      const promise = fetch(frameUrl(frame), {cache: "force-cache"})
        .then(async res => {
          if (!res.ok) throw new Error(`Frame ${frame} failed (${res.status})`);
          const url = URL.createObjectURL(await res.blob());
          state.cache.set(frame, url);
          trimCache();
          return url;
        })
        .finally(() => state.pending.delete(frame));
      state.pending.set(frame, promise);
      const url = await promise;
      return requestId === state.requestId ? url : "";
    }
    async function showFrame(frame) {
      const total = state.meta?.frames || 0;
      if (!total) return;
      const requestId = ++state.requestId;
      state.frame = Math.max(0, Math.min(total - 1, Math.round(Number(frame) || 0)));
      updateReadout();
      loadState.textContent = state.cache.has(state.frame) ? "" : "Loading frame...";
      try {
        const url = await fetchFrame(state.frame, requestId);
        if (!url || requestId !== state.requestId) return;
        const next = 1 - state.activeBuffer;
        const nextImage = frames[next];
        const currentImage = frames[state.activeBuffer];
        if (nextImage.src !== url) {
          nextImage.src = url;
          if (nextImage.decode) {
            try { await nextImage.decode(); } catch (_) {}
          }
        }
        if (requestId !== state.requestId) return;
        nextImage.classList.add("active");
        currentImage.classList.remove("active");
        state.activeBuffer = next;
        loadState.textContent = "";
        setMessage("");
      } catch (error) {
        if (requestId === state.requestId) {
          loadState.textContent = "";
          setMessage(error.message || "Movie frame failed to load.");
        }
      }
    }
    function stop() {
      state.playing = false;
      play.textContent = "Play";
      window.clearTimeout(state.timer);
    }
    function tick() {
      window.clearTimeout(state.timer);
      if (!state.playing || !state.meta) return;
      const displayFps = Math.min(18, Math.max(4, state.meta.fps));
      state.timer = window.setTimeout(async () => {
        const next = state.frame + Math.max(1, Math.round(state.meta.fps / displayFps));
        if (next >= state.meta.frames) {
          stop();
          return;
        }
        await showFrame(next);
        tick();
      }, 1000 / displayFps);
    }
    play.addEventListener("click", () => {
      if (!state.meta?.frames) return;
      state.playing = !state.playing;
      play.textContent = state.playing ? "Pause" : "Play";
      if (state.playing) tick();
      else window.clearTimeout(state.timer);
    });
    scrub.addEventListener("input", () => {
      stop();
      showFrame(Number(scrub.value));
    });
    frameInput.addEventListener("change", () => {
      stop();
      showFrame(Number(frameInput.value));
    });
    async function loadMeta() {
      stop();
      state.requestId += 1;
      for (const url of state.cache.values()) URL.revokeObjectURL(url);
      state.cache.clear();
      state.pending.clear();
      frames.forEach((img, index) => {
        img.removeAttribute("src");
        img.classList.toggle("active", index === 0);
      });
      setMessage("Loading movie...");
      try {
        const res = await fetch(`/experiments/${EXP_ID}/movie/meta?kind=${encodeURIComponent(kind)}`);
        const data = await res.json();
        if (!res.ok || !data.available) throw new Error(data.message || "Movie is unavailable.");
        state.meta = data;
        state.frame = 0;
        state.activeBuffer = 0;
        scrub.max = Math.max(0, data.frames - 1);
        frameInput.max = Math.max(0, data.frames - 1);
        summary.textContent = `${data.file_name} | ${data.frames} frames | ${data.width} x ${data.height} | ${Number(data.fps).toFixed(1)} fps`;
        updateReadout();
        await showFrame(0);
      } catch (error) {
        summary.textContent = error.message || "Movie is unavailable.";
        setMessage(error.message || "Movie is unavailable.");
        updateReadout();
      }
    }
    loadMeta();
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
  async function loadMeanIntensity() {
    const summary = document.getElementById("intensity-summary");
    try {
      const res = await fetch(`/experiments/${EXP_ID}/preprocessing/mean-intensity/data`);
      const data = await res.json();
      if (!res.ok || !data.available) throw new Error(data.message || "Mean intensity data are unavailable.");
      const available = Object.values(data.series || {}).filter(s => s.available);
      summary.textContent = available.map(s => `${s.file_name}: ${s.frames} frames, showing every ${s.display_step}`).join(" | ");
      drawMeanIntensity(data);
      renderQcMetrics(data);
    } catch (error) {
      summary.textContent = error.message || "Mean intensity data are unavailable.";
    }
  }
  document.querySelectorAll("[data-movie-viewer]").forEach(root => initPreprocessingMovieViewer(root, root.dataset.movieViewer));
  loadMeanIntensity();
  window.addEventListener("resize", loadMeanIntensity);
})();
