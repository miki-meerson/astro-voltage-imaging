"""Corrected-movie data for the preprocessing page: the movie in chunks of frames, and the mean-intensity traces from the saved QC files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from app.services import caches, polarity
from utils.file_utils import read_json
from utils.paths_utils import PathKind, path_for

PB_KINDS = ("astro", "neuron")
CHUNK_FRAMES = 60  # frames sent in one piece: about 2 seconds of the recording; the page asks for the next piece while it plays the current one
MAX_SIDE = 512  # pixels on the longer side of the frames sent (larger movies are averaged down by a whole number of pixels)
CONTRAST_SAMPLE_FRAMES = 24
CONTRAST_PERCENTILES = (1.0, 99.7)


def pb_movie_path(source_path: str | Path | None, kind: str) -> Path | None:
    """The photobleaching corrected movie of a channel, or None for an unknown channel or experiment."""
    if not source_path or kind not in PB_KINDS:
        return None
    return path_for(PathKind.ASTRO_PB_MOVIE if kind == "astro" else PathKind.NEURON_PB_MOVIE, source_path)


def pb_qc_metrics(source_path: str | Path | None, kind: str) -> dict:
    """Photobleaching QC numbers of `kind`; {} when the QC file was not written (the photobleaching step produces it)."""
    if not source_path:
        return {}
    return read_json(path_for(PathKind.PB_QC_METRICS, source_path, pb_kind=kind), "The photobleaching QC file", default={})


def pb_qc_trace(source_path: str | Path | None, kind: str, max_points: int) -> dict | None:
    if not source_path:
        return None
    path = path_for(PathKind.PB_QC_TRACE, source_path, pb_kind=kind)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        frame_idx = np.asarray(data["frame_idx"], dtype=np.int64)
        time_s = np.asarray(data["time_sec"], dtype=float)
        corrected_mean = np.asarray(data["corrected_mean"], dtype=float)
    if corrected_mean.size == 0:
        return None
    step = max(1, int(np.ceil(corrected_mean.size / max_points)))
    keep = np.arange(0, corrected_mean.size, step, dtype=np.int64)
    finite = corrected_mean[np.isfinite(corrected_mean)]
    return {
        "available": True,
        "source": "qc_trace",
        "file_name": path.name,
        "frames": int(corrected_mean.size),
        "display_step": int(step),
        "frame": frame_idx[keep].astype(int).tolist(),
        "time_s": time_s[keep].astype(float).tolist(),
        "mean": corrected_mean[keep].astype(float).tolist(),
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "qc": pb_qc_metrics(source_path, kind),
    }


def _open_movie(path: Path) -> np.ndarray:
    """The movie as a (time, y, x) array that is read from the file only where it is indexed."""
    try:
        movie = np.asarray(tifffile.memmap(path))
    except ValueError:  # tifffile cannot memory-map this file (compressed or not contiguous): read it through its series
        with tifffile.TiffFile(path) as tif:
            movie = np.asarray(tif.series[0].asarray(out="memmap"))
    if movie.ndim == 2:
        movie = movie[np.newaxis]
    while movie.ndim > 3:
        movie = movie[:, 0]
    return movie


def _shown(frames: np.ndarray, flip: bool, block: int) -> np.ndarray:
    """Frames as they are shown: sign flipped when `flip` (astro: bright = depolarization) and averaged down by `block` pixels."""
    frames = np.asarray(frames, dtype=np.float32)
    if flip:
        frames = polarity.to_biological(frames)
    if block > 1:
        n, height, width = frames.shape[0], (frames.shape[1] // block) * block, (frames.shape[2] // block) * block
        frames = frames[:, :height, :width].reshape(n, height // block, block, width // block, block).mean(axis=(2, 4))
    return frames


def movie_info(path: Path, fps: float, flip: bool) -> dict:
    """{available, frames, height, width (as sent), fps, chunk_frames, low, high}: the black and white values are the same for every chunk, taken
    from a few frames spread over the movie. Small, so it is kept in memory per file."""
    key = (str(path), int(path.stat().st_mtime_ns), bool(flip))
    cached = caches.MOVIE_INFO.get(key)
    if cached is not None:
        return dict(cached)
    movie = _open_movie(path)
    n_frames, height, width = (int(v) for v in movie.shape)
    block = max(1, int(np.ceil(max(height, width) / MAX_SIDE)))
    sample = np.linspace(0, n_frames - 1, min(CONTRAST_SAMPLE_FRAMES, n_frames), dtype=int)
    values = _shown(movie[sample], flip, block)
    values = values[np.isfinite(values)]
    low, high = (float(v) for v in np.percentile(values, CONTRAST_PERCENTILES)) if values.size else (0.0, 1.0)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = 0.0, 1.0
    info = {"available": True, "frames": n_frames, "height": height // block, "width": width // block, "block": block, "fps": float(fps),
            "chunk_frames": CHUNK_FRAMES, "low": low, "high": high, "file_name": path.name, "gevi_flipped": bool(flip)}
    caches.MOVIE_INFO.put(key, dict(info))
    return info


def movie_chunk(path: Path, fps: float, flip: bool, chunk: int) -> bytes:
    """Frames [chunk * CHUNK_FRAMES, (chunk + 1) * CHUNK_FRAMES) of the movie as raw 8-bit values, frame after frame. Raises ValueError for a chunk
    beyond the end."""
    info = movie_info(path, fps, flip)
    start = int(chunk) * CHUNK_FRAMES
    if chunk < 0 or start >= info["frames"]:
        raise ValueError(f"The movie has no chunk {chunk} ({info['frames']} frames).")
    frames = _shown(_open_movie(path)[start:start + CHUNK_FRAMES], flip, info["block"])
    scaled = np.clip((frames - info["low"]) / (info["high"] - info["low"]), 0, 1)
    return (scaled * 255).astype(np.uint8).tobytes()


def mean_intensity_payload(source_path: str | Path | None, fps: float, max_points: int = 350) -> dict:
    """The mean intensity of the corrected movies over time, from the saved QC traces of the photobleaching step."""
    traces = {kind: path_for(PathKind.PB_QC_TRACE, source_path, pb_kind=kind) for kind in PB_KINDS} if source_path else {}
    existing = [path for path in traces.values() if path.exists()]
    if not existing:
        return {"available": False, "message": "The photobleaching QC traces were not found. The photobleaching correction step writes them: rerun that step."}
    cache_key = (str(source_path), sum(int(path.stat().st_mtime_ns) for path in existing), int(max_points))
    cached = caches.MEAN_INTENSITY.get(cache_key)
    if cached is not None:
        return dict(cached)
    series_payload = {kind: pb_qc_trace(source_path, kind, max_points) or {"available": False, "message": f"The {kind} QC trace was not found."} for kind in PB_KINDS}
    payload = {"available": True, "fps": float(fps), "series": series_payload}
    caches.MEAN_INTENSITY.put(cache_key, dict(payload))
    return payload
