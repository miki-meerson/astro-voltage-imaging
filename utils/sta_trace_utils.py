from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.distance import squareform

from utils.sta_stats_utils import filter_valid_peaks


def pixels_in_crop(ys: np.ndarray, xs: np.ndarray, crop: dict) -> list:
    """The pixels (ys, xs) that fall inside a crop box {y0, x0, height, width}, as [[y, x], ...] in the crop's own coordinates."""
    ys, xs = np.asarray(ys) - int(crop["y0"]), np.asarray(xs) - int(crop["x0"])
    keep = (ys >= 0) & (ys < int(crop["height"])) & (xs >= 0) & (xs < int(crop["width"]))
    return np.stack([ys[keep], xs[keep]], axis=1).tolist()


def crop_bounds(center_yx: tuple[int, int], radius: int, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """(y_min, y_max, x_min, x_max) of the square of half-size `radius` around a cell centre, cut at the image edges (max is exclusive)."""
    y, x = center_yx
    height, width = image_shape
    return (
        max(0, int(y) - int(radius)),
        min(height, int(y) + int(radius) + 1),
        max(0, int(x) - int(radius)),
        min(width, int(x) + int(radius) + 1),
    )


def event_triggered_mean(x: np.ndarray, centers: np.ndarray, rels: np.ndarray) -> np.ndarray | None:
    """x: (T,), centers: (K,) frame indices, rels: relative frames -> mean over the centers of x[center + rel], over the centers whose whole window lies inside x (the STA job's
    rule, filter_valid_peaks); None when no center does."""
    rels = np.asarray(rels, dtype=int)
    inside = filter_valid_peaks(centers, len(x), -int(rels.min()), int(rels.max()))
    return x[inside[:, None] + rels[None, :]].mean(axis=0) if inside.size else None


def lag_corr_matrix(traces: np.ndarray, lag: int) -> np.ndarray:
    """Correlation of every trace (a row) with every other, the second one shifted by `lag` frames (only the frames both have are compared); 0 for a flat trace."""
    a, b = (traces[:, lag:], traces[:, : traces.shape[1] - lag]) if lag >= 0 else (traces[:, :lag], traces[:, -lag:])
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    num = a @ b.T
    den = np.sqrt((a ** 2).sum(1))[:, None] * np.sqrt((b ** 2).sum(1))[None, :]
    return num / np.maximum(den, 1e-12)


def correlation_linkage(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(corr, linkage) of the traces (rows): their correlation matrix (1 on the diagonal; a flat trace has no shape to compare, so no correlation) and the hierarchical clustering
    (average linkage on 1 - correlation) of them. Needs at least two traces."""
    corr = lag_corr_matrix(np.asarray(traces, dtype=float), 0)
    np.fill_diagonal(corr, 1.0)
    distance = np.clip(1 - corr, 0, 2)
    return corr, linkage(squareform((distance + distance.T) / 2, checks=False), method="average")


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
    """The mean of the pixels of `mask_full` (full-resolution boolean image) in every frame of `movie` (or only the frames `frame_indices`), as the movie is given: for the astro
    movie read with read_astro_movie that is the raw intensity in the app's polarity (depolarization up). NaN for an empty mask."""
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
            chunk = np.asarray(movie[first:last:regular_step, y0:y1, x0:x1], dtype=np.float32)
        else:
            idx = frame_indices[start:stop]
            chunk = np.asarray(movie[idx, y0:y1, x0:x1], dtype=np.float32)
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
