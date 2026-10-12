"""Movie access for the preprocessing / experiment pages: file paths, metadata, frames as JPEG bytes, mean-intensity traces."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import numpy as np

from app.services import caches, polarity

def experiment_fps(source_path: str | Path) -> float:
    """Frame rate of the experiment's movies, from its Experiment.xml."""
    from utils.movie_utils import get_movie_parameters

    return float(get_movie_parameters(f"{Path(source_path).as_posix()}/", is_dir=True)["fps"])


def pb_movie_path(source_path: str | Path | None, kind: str) -> Path | None:
    if not source_path:
        return None
    names = {
        "astro": "astro_photobleaching_corrected.tif",
        "neuron": "neuron_photobleaching_corrected.tif",
    }
    name = names.get(kind)
    if name is None:
        return None
    return Path(source_path) / "pipeline_results" / "pb_corrected" / name


def pb_qc_metrics(source_path: str | Path | None, kind: str) -> dict:
    if not source_path:
        return {}
    path = Path(source_path) / "pipeline_results" / "qc" / f"photobleaching_qc_{kind}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def pb_qc_plot_path(source_path: str | Path | None, kind: str) -> Path | None:
    if kind not in {"astro", "neuron"} or not source_path:
        return None
    return Path(source_path) / "pipeline_results" / "qc" / f"photobleaching_qc_plot_{kind}.png"


def latest_preprocessing_params(conn, experiment_id: int) -> dict:
    run = conn.execute(
        """
        SELECT output_dir,parameters_json
        FROM analysis_runs
        WHERE experiment_id=? AND step='preprocessing' AND is_test=0
        ORDER BY id DESC LIMIT 1
        """,
        (experiment_id,),
    ).fetchone()
    if run is None:
        return {}
    state_path = Path(run["output_dir"] or "") / "preprocessing_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            params_path = Path(state.get("params_path") or "")
            if params_path.exists():
                return json.loads(params_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        return json.loads(run["parameters_json"] or "{}")
    except json.JSONDecodeError:
        return {}


def pb_qc_trace(source_path: str | Path | None, kind: str, max_points: int) -> dict | None:
    if not source_path:
        return None
    path = Path(source_path) / "pipeline_results" / "qc" / f"photobleaching_qc_trace_{kind}.npz"
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


def movie_series_shape(series) -> tuple[int, int, int]:
    shape = tuple(int(v) for v in getattr(series, "shape", ()) or ())
    axes = str(getattr(series, "axes", "") or "").upper()
    if len(shape) == 2:
        return 1, shape[0], shape[1]
    if len(shape) < 3:
        raise ValueError(f"Unsupported movie shape: {shape}")
    if axes and len(axes) == len(shape):
        t_axis = next((i for i, axis in enumerate(axes) if axis in {"T", "I", "Q", "Z"}), 0)
        y_axis = axes.find("Y")
        x_axis = axes.find("X")
        if y_axis >= 0 and x_axis >= 0:
            return shape[t_axis], shape[y_axis], shape[x_axis]
    return shape[0], shape[-2], shape[-1]


def movie_metadata(path: Path, fps: float) -> dict:
    stat_key = int(path.stat().st_mtime_ns)
    cache_key = (str(path), stat_key)
    cached = caches.MOVIE_META.get(cache_key)
    if cached is not None:
        return dict(cached)
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        n_frames, height, width = movie_series_shape(series)
        dtype = str(series.dtype)
        axes = str(getattr(series, "axes", "") or "")
    payload = {
        "available": True,
        "frames": int(n_frames),
        "height": int(height),
        "width": int(width),
        "fps": float(fps),
        "duration_s": float(n_frames / max(fps, 1e-9)),
        "dtype": dtype,
        "axes": axes,
        "mtime_ns": stat_key,
        "file_name": path.name,
    }
    caches.MOVIE_META.put(cache_key, dict(payload))
    return payload


def movie_array_payload(path: Path, invert_sign: bool = False) -> dict:
    import tifffile

    stat_key = int(path.stat().st_mtime_ns)
    cache_key = (str(path), stat_key, bool(invert_sign))
    cached = caches.MOVIE_ARRAY.get(cache_key)
    if cached is not None:
        return cached

    try:
        movie = np.asarray(tifffile.memmap(path))
        axes = ""
    except Exception:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            axes = str(getattr(series, "axes", "") or "").upper()
            movie = np.asarray(series.asarray(out="memmap"))
    if movie.ndim == 2:
        movie = movie[np.newaxis, :, :]
    elif movie.ndim >= 3:
        if axes and len(axes) == movie.ndim:
            t_axis = next((i for i, axis in enumerate(axes) if axis in {"T", "I", "Q", "Z"}), 0)
            y_axis = axes.find("Y")
            x_axis = axes.find("X")
            if y_axis >= 0 and x_axis >= 0:
                movie = np.moveaxis(movie, (t_axis, y_axis, x_axis), (0, 1, 2))
        while movie.ndim > 3:
            movie = movie[:, 0]

    n_frames = int(movie.shape[0])
    sample_frames = np.linspace(0, max(0, n_frames - 1), min(24, n_frames), dtype=int) if n_frames else np.array([], dtype=int)
    samples = []
    for idx in sample_frames:
        frame = np.asarray(movie[int(idx)])
        if invert_sign:
            frame = polarity.to_biological(frame)
        y_step = max(1, frame.shape[0] // 220)
        x_step = max(1, frame.shape[1] // 220)
        samples.append(np.asarray(frame[::y_step, ::x_step], dtype=np.float32).ravel())
    if samples:
        values = np.concatenate(samples)
        values = values[np.isfinite(values)]
    else:
        values = np.empty(0, dtype=np.float32)
    if values.size:
        lo, hi = (float(v) for v in np.nanpercentile(values, [1, 99.7]))
    else:
        lo, hi = 0.0, 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    payload = {"movie": movie, "mtime_ns": stat_key, "contrast": (lo, hi), "invert_sign": bool(invert_sign)}
    caches.MOVIE_ARRAY.put(cache_key, payload)
    return payload


def read_movie_frame(path: Path, frame_index: int, invert_sign: bool = False) -> tuple[np.ndarray, int, tuple[float, float]]:
    payload = movie_array_payload(path, invert_sign)
    movie = payload["movie"]
    frame_index = min(max(0, int(frame_index)), max(0, int(movie.shape[0]) - 1))
    frame = np.asarray(movie[frame_index])
    if invert_sign:
        frame = polarity.to_biological(frame)
    return frame, int(payload["mtime_ns"]), payload["contrast"]


def movie_frame_jpeg(path: Path, frame_index: int, max_size: int, contrast: str, invert_sign: bool = False):
    from PIL import Image

    frame, mtime_ns, contrast_limits = read_movie_frame(path, frame_index, invert_sign)
    cache_key = (str(path), mtime_ns, int(frame_index), int(max_size), str(contrast), bool(invert_sign))
    cached = caches.MOVIE_JPEG.get(cache_key)
    if cached is not None:
        return cached

    img = np.asarray(frame, dtype=np.float32)
    if contrast == "full":
        finite = img[np.isfinite(img)]
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    else:
        lo, hi = contrast_limits
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((img - lo) / (hi - lo), 0, 1)
    scaled = (scaled * 255).astype(np.uint8)
    image = Image.fromarray(scaled, mode="L")
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
    bio = BytesIO()
    image.save(bio, format="JPEG", quality=82, optimize=False)
    data = bio.getvalue()
    caches.MOVIE_JPEG.put(cache_key, data)
    return data


def _mean_series(path: Path, fps: float, max_points: int) -> dict:
    """Mean intensity of a movie over time (spatially subsampled, at most ~max_points time points)."""
    movie = movie_array_payload(path, invert_sign=False)["movie"]
    n_frames = int(movie.shape[0])
    step = max(1, int(np.ceil(n_frames / max_points)))
    frame_idx = np.arange(0, n_frames, step, dtype=np.int64)
    y_step = max(1, int(movie.shape[1]) // 64)
    x_step = max(1, int(movie.shape[2]) // 64)
    means = []
    for start in range(0, len(frame_idx), 96):
        idx = frame_idx[start:start + 80]
        chunk = np.asarray(movie[idx, ::y_step, ::x_step], dtype=np.float32)
        means.extend(np.nanmean(chunk, axis=(1, 2)).astype(float).tolist())
    values = np.asarray(means, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "available": True,
        "file_name": path.name,
        "frames": n_frames,
        "display_step": int(step),
        "frame": frame_idx.astype(int).tolist(),
        "time_s": (frame_idx / max(fps, 1e-9)).astype(float).tolist(),
        "mean": values.astype(float).tolist(),
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
    }


def mean_intensity_payload(source_path: str | Path | None, fps: float, max_points: int = 350) -> dict:
    paths = {kind: pb_movie_path(source_path, kind) for kind in ("astro", "neuron")}
    existing = [path for path in paths.values() if path is not None and path.exists()]
    if not existing:
        return {"available": False, "message": "Photobleaching corrected movies were not found."}
    stat_key = sum(int(path.stat().st_mtime_ns) for path in existing)
    cache_key = (str(source_path), stat_key, int(max_points))
    cached = caches.MEAN_INTENSITY.get(cache_key)
    if cached is not None:
        return dict(cached)

    series_payload: dict[str, dict] = {}
    for kind, path in paths.items():
        trace = pb_qc_trace(source_path, kind, max_points)
        if trace is not None:
            series_payload[kind] = trace
            continue
        if path is None or not path.exists():
            series_payload[kind] = {"available": False, "message": f"{kind} movie was not found."}
            continue
        series_payload[kind] = {**_mean_series(path, fps, max_points), "qc": pb_qc_metrics(source_path, kind)}
    payload = {"available": True, "fps": float(fps), "series": series_payload}
    caches.MEAN_INTENSITY.put(cache_key, dict(payload))
    return payload
