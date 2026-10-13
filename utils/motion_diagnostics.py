"""Spike-triggered motion diagnostics for STA maps.

Question answered per cell: are this cell's spikes time-locked to motion / registration quality?
For each motion series (applied registration shifts, jerk, per-frame registration quality) we compute the
spike-triggered average over the same window as the STA effect and test it with the SAME circular-shift
surrogates used for the maps. Diagnostic only: it never changes the map statistics.

Limitation: the shifts are the motion that was *corrected*. Residual motion after registration is only
visible indirectly through `reg_corr` (correlation of each frame with the mean image).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from utils import globals_utils as g

SERIES_NAMES = ("dy", "dx", "shift_mag", "jerk", "reg_corr")


def cell_motion_status(summary: pd.DataFrame) -> dict[int, str]:
    """{cell id: MOTION_OK | MOTION_SUSPECTED | MOTION_NOT_TESTED} from the STA summary csv. The one place that says whether a cell
    counts as motion-suspected: every consumer (anatomy summary, group test, pages) uses it and none opens the per-cell files.
    Raises ValueError when the csv has no status column (rerun STA Maps)."""
    if g.MOTION_STATUS_COLUMN not in summary:
        raise ValueError(f"The STA summary csv has no '{g.MOTION_STATUS_COLUMN}' column. It is written by the STA Maps step: rerun STA Maps for this experiment.")
    cells = summary[g.MOTION_STATUS_COLUMN]
    per_cell = cells.groupby(summary["cell_id"]).first()  # every row of a cell carries the same status; empty rows are skipped
    return {int(c): str(s) if pd.notna(s) else g.MOTION_NOT_TESTED for c, s in per_cell.items()}


def load_shifts(shifts_path: str | Path) -> np.ndarray | None:
    """The registration shifts (frames, 2) from the motion-correction .mat file (path_for(PathKind.MC_SHIFTS, experiment)); None when absent."""
    path = Path(shifts_path)
    if not path.exists():
        return None
    return np.asarray(loadmat(path)[g.MOTION_SHIFTS_MAT_KEY], dtype=np.float32)


def registration_quality(movie: np.ndarray) -> np.ndarray:
    """Per-frame Pearson correlation of a (T, H, W) movie with its own mean image."""
    flat = np.asarray(movie, dtype=np.float32).reshape(len(movie), -1)
    mean = flat.mean(axis=0)
    mean = mean - mean.mean()
    flat = flat - flat.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(flat, axis=1) * np.linalg.norm(mean) + 1e-12
    return ((flat @ mean) / denom).astype(np.float32)


def build_motion_series(shifts: np.ndarray, reg_corr: np.ndarray | None = None) -> dict[str, np.ndarray]:
    shifts = np.asarray(shifts, dtype=np.float32)
    jerk = np.r_[0.0, np.linalg.norm(np.diff(shifts, axis=0), axis=1)].astype(np.float32)
    series = {
        "dy": shifts[:, 0],
        "dx": shifts[:, 1],
        "shift_mag": np.linalg.norm(shifts, axis=1).astype(np.float32),
        "jerk": jerk,
    }
    if reg_corr is not None:
        series["reg_corr"] = np.asarray(reg_corr, dtype=np.float32)
    return series


def motion_sta_stats(
    series: dict[str, np.ndarray],
    peaks: np.ndarray,
    permutation_shifts: np.ndarray,
    pre: int,
    post: int,
    baseline_window: tuple[int, int],
    analysis_window: tuple[int, int],
    flag_alpha: float = 0.01,
) -> dict:
    """Event-triggered average of each motion series vs circular-shift null.

    peaks are frame indices of valid spikes (pre+1 <= peak < n_frames-post). Shifts follow the STA convention:
    surrogate centers = (valid position + shift) mod n_valid, valid centers = arange(pre+1, n_frames-post).
    Returns arrays over the analysis window, baseline-subtracted like the STA effect.
    """
    names = [n for n in SERIES_NAMES if n in series]
    n_frames = len(series[names[0]])
    n_valid = n_frames - post - pre - 1
    rels = np.arange(-pre, post + 1)
    base = np.flatnonzero((rels >= baseline_window[0]) & (rels <= baseline_window[1]))
    ana = np.flatnonzero((rels >= analysis_window[0]) & (rels <= analysis_window[1]))
    pos = np.asarray(peaks, dtype=int) - (pre + 1)
    pos = pos[(pos >= 0) & (pos < n_valid)]
    shifts = np.r_[0, np.asarray(permutation_shifts, dtype=int)]  # row 0 = observed

    observed = np.full((len(names), ana.size), np.nan, dtype=np.float32)
    null_mean = np.full_like(observed, np.nan)
    null_sd = np.full_like(observed, np.nan)
    stat = np.full(len(names), np.nan, dtype=np.float32)
    p = np.full(len(names), np.nan, dtype=np.float32)
    z = np.full(len(names), np.nan, dtype=np.float32)
    if pos.size:
        centers = ((pos[None, :] + shifts[:, None]) % n_valid) + (pre + 1)  # (1+P, K)
        for i, name in enumerate(names):
            x = np.asarray(series[name], dtype=np.float32)
            sta = x[centers[:, :, None] + rels[None, None, :]].mean(axis=1)  # (1+P, rels)
            effect = (sta - sta[:, base].mean(axis=1, keepdims=True))[:, ana]
            s = np.abs(effect).max(axis=1)
            observed[i] = effect[0]
            null_mean[i] = effect[1:].mean(axis=0)
            null_sd[i] = effect[1:].std(axis=0)
            stat[i] = s[0]
            p[i] = (1 + np.sum(s[1:] >= s[0])) / len(s)
            z[i] = (s[0] - s[1:].mean()) / (s[1:].std() + 1e-12)
    min_p = float(np.nanmin(p)) if np.isfinite(p).any() else float("nan")
    return {
        "series_names": np.asarray(names),
        "rels_analysis": rels[ana],
        "observed": observed,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "stat": stat,
        "p": p,
        "z": z,
        "min_p": np.float32(min_p),
        "flag": np.bool_(np.isfinite(min_p) and min_p <= flag_alpha),
        "flag_alpha": np.float32(flag_alpha),
        # smallest p the permutation count can resolve; if p_floor > flag_alpha the flag can never trigger
        "p_floor": np.float32(1.0 / (len(shifts))),
        "n_peaks_used": np.int32(pos.size),
    }


def align_shifts_to_analysed_frames(shifts: np.ndarray, n_frames: int, trimmed: int = 0) -> np.ndarray | None:
    """Rows of `shifts` that belong to the analysed (photobleaching-corrected) movie.

    Bad segments are removed outside the app before motion correction, so the shift rows already match the cleaned movie. The only
    difference left is the first `trimmed` frames the photobleaching step dropped (qc/photobleaching_qc_astro.json).
    Returns shifts when the lengths match, shifts[trimmed:] when that makes them match, else None.
    """
    shifts = np.asarray(shifts)
    n = int(n_frames)
    if len(shifts) == n:
        return shifts
    t = max(int(trimmed), 0)
    if t and len(shifts) == n + t:
        return shifts[t:]
    return None


def cuts_from_bad_segments(bad, trimmed: int = 0) -> list[int]:
    """Positions (in analysed frames) where removed bad segments leave a discontinuity. Overlapping segments are merged."""
    merged: list[list[int]] = []
    for a, b in sorted((int(x[0]), int(x[1])) for x in bad or []):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    removed, cuts = 0, []
    for a, b in merged:
        pos = a - removed - int(trimmed)
        if pos > 0:
            cuts.append(pos)
        removed += b - a
    return cuts
