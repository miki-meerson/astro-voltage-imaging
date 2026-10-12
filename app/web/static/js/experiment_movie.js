(function () {
  const EXP_ID = JSON.parse(document.getElementById("page-data").textContent).expId;
  const movieKind = document.getElementById("movie-kind");
  const movieFrames = [
    document.getElementById("movie-frame-a"),
    document.getElementById("movie-frame-b"),
  ];
  const movieMessage = document.getElementById("movie-message");
  const moviePlay = document.getElementById("movie-play");
  const movieScrub = document.getElementById("movie-scrub");
  const movieFrameInput = document.getElementById("movie-frame-input");
  const movieRate = document.getElementById("movie-rate");
  const movieReadout = document.getElementById("movie-readout");
  const movieSummary = document.getElementById("movie-summary");
  const movieLoadState = document.getElementById("movie-load-state");
  const movieState = {
    meta: null,
    frame: 0,
    playing: false,
    timer: null,
    requestId: 0,
    objectUrl: "",
    cache: new Map(),
    pending: new Map(),
    activeBuffer: 0,
  };
  function formatMovieTime(frame) {
    const fps = movieState.meta?.fps || 1;
    const seconds = frame / Math.max(1e-9, fps);
    return seconds >= 60 ? `${(seconds / 60).toFixed(1)} min` : `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  }
  function movieFrameUrl(frame) {
    return `/experiments/${EXP_ID}/movie/frame?kind=${encodeURIComponent(movieKind.value)}&frame=${frame}&max_size=820`;
  }
  function setMovieMessage(text) {
    movieMessage.textContent = text || "";
    movieMessage.style.display = text ? "grid" : "none";
  }
  function updateMovieReadout() {
    const total = movieState.meta?.frames || 0;
    movieReadout.textContent = total
      ? `Frame ${movieState.frame + 1} / ${total} | ${formatMovieTime(movieState.frame)}`
      : "Frame 0 / 0";
    movieScrub.value = movieState.frame;
    movieFrameInput.value = movieState.frame;
  }
  function trimMovieCache() {
    const keep = new Set();
    for (let i = -8; i <= 16; i++) {
      const frame = movieState.frame + i;
      if (frame >= 0 && frame < (movieState.meta?.frames || 0)) keep.add(frame);
    }
    for (const [frame, url] of movieState.cache.entries()) {
      if (!keep.has(frame)) {
        URL.revokeObjectURL(url);
        movieState.cache.delete(frame);
      }
    }
  }
  async function fetchMovieFrame(frame, requestId) {
    if (movieState.cache.has(frame)) return movieState.cache.get(frame);
    if (movieState.pending.has(frame)) return movieState.pending.get(frame);
    const promise = fetch(movieFrameUrl(frame), {cache: "force-cache"})
      .then(async res => {
        if (!res.ok) throw new Error(`Frame ${frame} failed (${res.status})`);
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        movieState.cache.set(frame, url);
        trimMovieCache();
        return url;
      })
      .finally(() => movieState.pending.delete(frame));
    movieState.pending.set(frame, promise);
    const url = await promise;
    if (requestId !== movieState.requestId) return "";
    return url;
  }
  async function swapMovieFrame(url, requestId) {
    const nextBuffer = 1 - movieState.activeBuffer;
    const nextImage = movieFrames[nextBuffer];
    const currentImage = movieFrames[movieState.activeBuffer];
    if (nextImage.src !== url) {
      nextImage.src = url;
      if (nextImage.decode) {
        try { await nextImage.decode(); } catch (_) {}
      }
    }
    if (requestId !== movieState.requestId) return;
    nextImage.classList.add("active");
    currentImage.classList.remove("active");
    movieState.activeBuffer = nextBuffer;
  }
  function preloadMovieFrames(center) {
    const total = movieState.meta?.frames || 0;
    const requestId = movieState.requestId;
    const step = Math.max(1, Math.round((movieState.meta?.fps || 10) / 12));
    for (let i = 1; i <= 2; i++) {
      const frame = Math.min(total - 1, center + i * step);
      if (frame > center && !movieState.cache.has(frame)) {
        fetchMovieFrame(frame, requestId).catch(() => {});
      }
    }
  }
  async function showMovieFrame(frame) {
    const total = movieState.meta?.frames || 0;
    if (!total) return;
    const requestId = ++movieState.requestId;
    movieState.frame = Math.max(0, Math.min(total - 1, Math.round(Number(frame) || 0)));
    updateMovieReadout();
    movieLoadState.textContent = movieState.cache.has(movieState.frame) ? "" : "Loading frame...";
    try {
      const url = await fetchMovieFrame(movieState.frame, requestId);
      if (requestId !== movieState.requestId || !url) return;
      await swapMovieFrame(url, requestId);
      setMovieMessage("");
      movieLoadState.textContent = "";
      preloadMovieFrames(movieState.frame);
    } catch (error) {
      if (requestId === movieState.requestId) {
        movieLoadState.textContent = "";
        setMovieMessage(error.message || "Movie frame failed to load.");
      }
    }
  }
  function stopMovie() {
    movieState.playing = false;
    moviePlay.textContent = "Play";
    window.clearTimeout(movieState.timer);
  }
  function scheduleMovieTick() {
    window.clearTimeout(movieState.timer);
    if (!movieState.playing || !movieState.meta) return;
    const displayFps = Math.min(24, Math.max(4, movieState.meta.fps * Number(movieRate.value || 1)));
    movieState.timer = window.setTimeout(async () => {
      const next = movieState.frame + Math.max(1, Math.round(movieState.meta.fps / displayFps));
      if (next >= movieState.meta.frames) {
        stopMovie();
        return;
      }
      await showMovieFrame(next);
      scheduleMovieTick();
    }, 1000 / displayFps);
  }
  function toggleMovie() {
    if (!movieState.meta?.frames) return;
    movieState.playing = !movieState.playing;
    moviePlay.textContent = movieState.playing ? "Pause" : "Play";
    if (movieState.playing) scheduleMovieTick();
    else window.clearTimeout(movieState.timer);
  }
  async function loadMovieMeta() {
    stopMovie();
    movieState.requestId += 1;
    for (const url of movieState.cache.values()) URL.revokeObjectURL(url);
    movieState.cache.clear();
    movieState.pending.clear();
    movieState.meta = null;
    movieState.frame = 0;
    movieState.activeBuffer = 0;
    movieFrames.forEach((img, index) => {
      img.removeAttribute("src");
      img.classList.toggle("active", index === 0);
    });
    movieSummary.textContent = "Loading movie metadata...";
    setMovieMessage("Loading movie...");
    try {
      const res = await fetch(`/experiments/${EXP_ID}/movie/meta?kind=${encodeURIComponent(movieKind.value)}`);
      const data = await res.json();
      if (!res.ok || !data.available) throw new Error(data.message || "Movie is unavailable.");
      movieState.meta = data;
      movieScrub.max = Math.max(0, data.frames - 1);
      movieFrameInput.max = Math.max(0, data.frames - 1);
      movieSummary.textContent = `${data.file_name} | ${data.frames} frames | ${data.width} x ${data.height} | ${Number(data.fps).toFixed(1)} fps${data.gevi_flipped ? " | GEVI sign flipped (bright = depolarization)" : ""}`;
      updateMovieReadout();
      await showMovieFrame(0);
    } catch (error) {
      movieSummary.textContent = error.message || "Movie is unavailable.";
      setMovieMessage(error.message || "Movie is unavailable.");
      updateMovieReadout();
    }
  }
  moviePlay.addEventListener("click", toggleMovie);
  movieKind.addEventListener("change", loadMovieMeta);
  movieRate.addEventListener("change", () => {
    if (movieState.playing) scheduleMovieTick();
  });
  movieScrub.addEventListener("input", () => {
    stopMovie();
    showMovieFrame(Number(movieScrub.value));
  });
  movieFrameInput.addEventListener("change", () => {
    stopMovie();
    showMovieFrame(Number(movieFrameInput.value));
  });
  loadMovieMeta();
})();
