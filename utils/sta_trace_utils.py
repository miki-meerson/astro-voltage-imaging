from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt


def significant_cluster_specs_from_npz(npz_path: str | Path, cluster_colors: dict | None = None) -> list[dict]:
    with np.load(npz_path, allow_pickle=True) as d:
        sig = np.asarray(d["significant_cluster_mask"], dtype=bool)
        effect = np.asarray(d["observed_effect_movie"], dtype=float)
        # trace values are dF/F-like: use the raw effect, not the tested z-statistic (sign test below is unaffected)
        trace_effect = np.asarray(d["observed_effect_raw_movie"], dtype=float) if "observed_effect_raw_movie" in d else effect
        crop = tuple(np.asarray(d["anatomy_crop_bounds"], dtype=int))
        pre = int(np.asarray(d["pre_frames"]))
        post = int(np.asarray(d["post_frames"]))
        spatial_bin = int(np.asarray(d["spatial_bin"]))
        peaks = np.asarray(d["peaks"], dtype=int)
        summary = np.asarray(d["cluster_summary"], dtype=object)
        analysis_window = tuple(np.asarray(d["analysis_window"], dtype=int))
    if not sig.any() or summary.size == 0:
        return []
    sig_rows = summary[np.asarray(summary[:, 6], dtype=bool)]
    if sig_rows.size == 0:
        return []

    allowed_signs = set(cluster_colors or {"positive": None, "negative": None})
    mask_crop = sig.any(axis=0)
    effect_trace = np.asarray(trace_effect[:, mask_crop].mean(axis=1), dtype=np.float32)
    effect_t_rel = np.arange(int(analysis_window[0]), int(analysis_window[1]) + 1, dtype=int)[: effect_trace.size]
    specs = []
    for row in sig_rows:
        sign = str(row[1]).lower()
        if sign not in allowed_signs:
            sign = "positive" if np.nanmean(effect[sig]) >= 0 else "negative"
        specs.append(
            {
                "component_id": int(row[0]),
                "sign": sign,
                "mask_crop": mask_crop,
                "crop": crop,
                "pre": pre,
                "post": post,
                "spatial_bin": spatial_bin,
                "peaks": peaks,
                "effect_trace": effect_trace,
                "effect_t_rel": effect_t_rel,
            }
        )
    return specs


def crop_mask_to_fullres(mask_crop: np.ndarray, crop: tuple[int, int, int, int], spatial_bin: int, image_shape: tuple[int, int]) -> np.ndarray:
    y0, y1, x0, x1 = crop
    spatial_bin = int(spatial_bin)
    binned_shape = (int(np.ceil(image_shape[0] / spatial_bin)), int(np.ceil(image_shape[1] / spatial_bin)))
    full_binned = np.zeros(binned_shape, dtype=bool)
    h = min(mask_crop.shape[0], y1 - y0)
    w = min(mask_crop.shape[1], x1 - x0)
    full_binned[y0 : y0 + h, x0 : x0 + w] = np.asarray(mask_crop, dtype=bool)[:h, :w]
    if spatial_bin <= 1:
        return full_binned[: image_shape[0], : image_shape[1]]
    return np.repeat(np.repeat(full_binned, spatial_bin, axis=0), spatial_bin, axis=1)[: image_shape[0], : image_shape[1]]


def spike_triggered_dff_trace(trace: np.ndarray, peaks: np.ndarray, pre: int, post: int, baseline_frames: int | None = None) -> np.ndarray:
    y = np.asarray(trace, dtype=float)
    baseline_frames = max(1, int(pre // 2 if baseline_frames is None else baseline_frames))
    trials = []
    for peak in np.asarray(peaks, dtype=int):
        a, b = int(peak - pre), int(peak + post + 1)
        if a < 0 or b > y.size:
            continue
        seg = y[a:b]
        f0 = np.nanmean(seg[:baseline_frames])
        trials.append((seg - f0) / (abs(f0) + 1e-6))
    return np.nanmean(np.vstack(trials), axis=0) if trials else np.full(pre + post + 1, np.nan)


def cluster_dff_trace_from_movie(movie: np.ndarray, mask_full: np.ndarray, peaks: np.ndarray, pre: int, post: int) -> np.ndarray:
    yy, xx = np.where(np.asarray(mask_full, dtype=bool))
    if yy.size == 0:
        return np.full(pre + post + 1, np.nan)
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    local_mask = np.asarray(mask_full[y0:y1, x0:x1], dtype=bool)
    baseline_frames = max(1, int(pre) // 2)
    trials = []
    for peak in np.asarray(peaks, dtype=int):
        a, b = int(peak - pre), int(peak + post + 1)
        if a < 0 or b > movie.shape[0]:
            continue
        seg = -np.asarray(movie[a:b, y0:y1, x0:x1], dtype=np.float32)[:, local_mask]
        f0 = np.nanmean(seg[:baseline_frames], axis=0, keepdims=True)
        trials.append(np.nanmean((seg - f0) / (np.abs(f0) + 1e-6), axis=1))
    return np.nanmean(np.vstack(trials), axis=0) if trials else np.full(pre + post + 1, np.nan)


def cluster_raw_trace_from_movie(
    movie: np.ndarray,
    mask_full: np.ndarray,
    frame_chunk: int = 512,
    frame_indices: np.ndarray | None = None,
) -> np.ndarray:
    yy, xx = np.where(np.asarray(mask_full, dtype=bool))
    if yy.size == 0:
        n = int(movie.shape[0]) if frame_indices is None else np.asarray(frame_indices).size
        return np.full(n, np.nan, dtype=np.float32)
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    local_mask = np.asarray(mask_full[y0:y1, x0:x1], dtype=bool)
    if frame_indices is None:
        frame_indices = np.arange(int(movie.shape[0]), dtype=int)
    else:
        frame_indices = np.asarray(frame_indices, dtype=int)
    out = np.empty(frame_indices.size, dtype=np.float32)
    regular_step = None
    if frame_indices.size > 1:
        diffs = np.diff(frame_indices)
        if np.all(diffs == diffs[0]):
            regular_step = int(diffs[0])
    for start in range(0, frame_indices.size, int(frame_chunk)):
        stop = min(frame_indices.size, start + int(frame_chunk))
        if regular_step is not None:
            first = int(frame_indices[start])
            last = int(frame_indices[stop - 1]) + 1
            chunk = -np.asarray(movie[first:last:regular_step, y0:y1, x0:x1], dtype=np.float32)
        else:
            idx = frame_indices[start:stop]
            chunk = -np.asarray(movie[idx, y0:y1, x0:x1], dtype=np.float32)
        out[start:stop] = chunk[:, local_mask].mean(axis=1)
    return out


def dff_from_lowest_abs(trace: np.ndarray, fraction: float = 0.08) -> tuple[np.ndarray, float]:
    y = np.asarray(trace, dtype=np.float32).ravel()
    finite = np.flatnonzero(np.isfinite(y))
    if finite.size == 0:
        return np.full_like(y, np.nan, dtype=np.float32), np.nan
    n_base = max(1, int(np.ceil(float(fraction) * finite.size)))
    base_idx = finite[np.argsort(np.abs(y[finite]))[:n_base]]
    f0 = float(np.nanmean(y[base_idx]))
    if not np.isfinite(f0) or abs(f0) <= 1e-6:
        return np.full_like(y, np.nan, dtype=np.float32), f0
    return ((y - f0) / (abs(f0) + 1e-6)).astype(np.float32), f0


def lowpass_centered(trace: np.ndarray, sample_rate_hz: float, cutoff_hz: float = 0.5, order: int = 2) -> np.ndarray:
    y = np.asarray(trace, dtype=np.float32).ravel()
    out = np.full_like(y, np.nan, dtype=np.float32)
    finite = np.isfinite(y)
    if finite.sum() < max(8, int(order) * 4):
        return out
    nyquist = 0.5 * float(sample_rate_hz)
    if nyquist <= 0 or float(cutoff_hz) >= nyquist:
        centered = y - np.nanmean(y)
        return centered.astype(np.float32)

    filled = y.copy()
    filled[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), y[finite])
    sos = butter(int(order), float(cutoff_hz) / nyquist, btype="lowpass", output="sos")
    filtered = sosfiltfilt(sos, filled)
    filtered = filtered - np.nanmean(filtered)
    out[finite] = filtered[finite]
    return out.astype(np.float32)
