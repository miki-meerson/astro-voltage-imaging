"""The global trends of an experiment: the slow signals that all of the field shares (the frame-mean intensity and the image shifts) and where in the
field each one is strong. Computed by the STA job from the movie it analyses (the movie, spike-independent) and saved as global_trends.npz, so the
Hypotheses tab only reads it.

For every series (the frame mean first, then the motion series) each pixel gets one number: its amplitude, the standard deviation of that pixel's
relative change that the series explains. Amplitude = |cov(pixel, series)| / |pixel mean| in percent per standard deviation of the series, read
from one pass over the movie (the series are z-scored, so the covariance is the regression slope). No spike enters it.
"""
from __future__ import annotations

import numpy as np

GLOBAL_SERIES = "global_mean"  # name of the frame-mean trace (the first series)
MEAN_FLOOR = 0.05  # a pixel whose mean is below this fraction of the median pixel mean is divided by that floor (dark border pixels would blow up)
SERIES_FOR_MAPS = ("dy", "dx", "shift_mag", "reg_corr")  # the motion series that also get a map (jerk is too spiky to carry a slow signal)


def zscore(x: np.ndarray) -> np.ndarray | None:
    """The series with mean 0 and sd 1, or None when it is constant."""
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return None if not np.isfinite(sd) or sd == 0 else (x - x.mean()) / sd


def compute_global_trends(movie: np.ndarray, motion_series: dict | None, chunk_values: int = 4_000_000) -> dict:
    """The traces and amplitude maps of a (T, H, W) movie in the biological sign (negative baseline, up = depolarization).

    motion_series: {name: (T,) series} aligned to the movie's frames (None when the motion test could not be built: only the frame mean is used).
    Returns names, traces (n_series, T), amplitude (n_series, H, W) and mean_image (H, W).
    Traces: the frame mean as % change from its median (up = depolarization), the motion series as they are."""
    frames, height, width = (int(v) for v in movie.shape)
    step = max(8, chunk_values // (height * width))
    mean_trace = np.empty(frames, dtype=np.float64)
    for start in range(0, frames, step):
        mean_trace[start:start + step] = np.asarray(movie[start:start + step], dtype=np.float64).reshape(-1, height * width).mean(axis=1)
    reference = np.median(mean_trace)
    series = {GLOBAL_SERIES: (mean_trace - reference) / abs(reference) * 100.0}
    for name in SERIES_FOR_MAPS:
        if motion_series and name in motion_series and len(motion_series[name]) == frames:
            series[name] = np.asarray(motion_series[name], dtype=np.float64)
    scores = {name: zscore(values) for name, values in series.items()}
    names = [name for name in series if scores[name] is not None]
    z = np.stack([scores[name] for name in names], axis=1)  # (T, K)
    pixel_sum = np.zeros(height * width, dtype=np.float64)
    pixel_cov = np.zeros((height * width, len(names)), dtype=np.float64)
    for start in range(0, frames, step):
        block = np.asarray(movie[start:start + step], dtype=np.float64).reshape(-1, height * width)
        pixel_sum += block.sum(axis=0)
        pixel_cov += block.T @ z[start:start + step]
    pixel_mean = pixel_sum / frames
    scale = np.maximum(np.abs(pixel_mean), MEAN_FLOOR * np.median(np.abs(pixel_mean)))
    amplitude = (np.abs(pixel_cov / frames) / scale[:, None] * 100.0).T.reshape(len(names), height, width)  # the series are z-scored: cov = slope
    return {
        "names": np.asarray(names),
        "traces": np.stack([series[name] for name in names]).astype(np.float32),
        "amplitude": amplitude.astype(np.float32),
        "mean_image": pixel_mean.reshape(height, width).astype(np.float32),
    }
