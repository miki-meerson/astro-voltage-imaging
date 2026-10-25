"""Spike-triggered motion diagnostics for STA maps.

Question answered per cell: are this cell's spikes time-locked to motion / registration quality?
For each motion series (applied registration shifts, jerk, per-frame registration quality) we compute the
spike-triggered average over the same window as the STA effect and test it with the SAME random-frame
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


def load_patch_extent(shifts_path: str | Path) -> np.ndarray | None:
    """The per-frame extremes of the patch shifts (frames, 4) = (dy max, dy min, dx max, dx min) that a non-rigid motion correction saved next to the shifts;
    None when the file is absent or the correction was rigid."""
    path = Path(shifts_path)
    if not path.exists():
        return None
    saved = loadmat(path)
    return np.asarray(saved[g.MOTION_PATCH_EXTENT_MAT_KEY], dtype=np.float32) if g.MOTION_PATCH_EXTENT_MAT_KEY in saved else None


def patch_shift_summary(x_shifts: list, y_shifts: list) -> tuple[np.ndarray, np.ndarray]:
    """(shifts (frames, 2) = (dy, dx) averaged over the patches, extent (frames, 4) = (dy max, dy min, dx max, dx min) over the patches) of the per-frame patch
    shifts of a non-rigid motion correction (one array of patch shifts per frame, as CaImAn keeps them)."""
    dy, dx = np.asarray(y_shifts, dtype=np.float64).reshape(len(y_shifts), -1), np.asarray(x_shifts, dtype=np.float64).reshape(len(x_shifts), -1)
    return (np.stack([dy.mean(axis=1), dx.mean(axis=1)], axis=1).astype(np.float32),
            np.stack([dy.max(axis=1), dy.min(axis=1), dx.max(axis=1), dx.min(axis=1)], axis=1).astype(np.float32))


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
    surrogate_centers: np.ndarray,
    pre: int,
    post: int,
    baseline_window: tuple[int, int],
    analysis_window: tuple[int, int],
    flag_alpha: float,
) -> dict:
    """Event-triggered average of each motion series vs the random-frame null of the STA maps.

    peaks are frame indices of valid spikes; surrogate_centers (n_surrogates, len(peaks)) are the frames of the chance copies, the same
    ones the maps use (random valid spike frames, splice-free when the maps exclude spikes near splices).
    Returns arrays over the analysis window, baseline-subtracted like the STA effect. Per series: stat = max |effect| over the window, p = rank
    among the copies and z. The cell gets ONE p, `p_family`, corrected for having several series: each copy (and the real spikes) is
    scored by its largest z over the series, and p_family is the rank of the real score among the copies. `flag` = p_family <= flag_alpha.
    """
    names = [n for n in SERIES_NAMES if n in series]
    rels = np.arange(-pre, post + 1)
    base = np.flatnonzero((rels >= baseline_window[0]) & (rels <= baseline_window[1]))
    ana = np.flatnonzero((rels >= analysis_window[0]) & (rels <= analysis_window[1]))
    peaks = np.asarray(peaks, dtype=int)
    centers = np.vstack([peaks[None, :], np.asarray(surrogate_centers, dtype=int).reshape(len(surrogate_centers), peaks.size)])  # row 0 = observed

    observed = np.full((len(names), ana.size), np.nan, dtype=np.float32)
    null_mean = np.full_like(observed, np.nan)
    null_sd = np.full_like(observed, np.nan)
    stat = np.full(len(names), np.nan, dtype=np.float32)
    p = np.full(len(names), np.nan, dtype=np.float32)
    z = np.full(len(names), np.nan, dtype=np.float32)
    z_copies = np.full((len(names), len(centers)), np.nan, dtype=np.float32)  # z of the real spikes (column 0) and of every copy, per series
    if peaks.size:
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
            z_copies[i] = (s - s[1:].mean()) / (s[1:].std() + 1e-12)
            z[i] = z_copies[i, 0]
    p_family = float("nan")
    if np.isfinite(z_copies).any():
        strongest = np.nanmax(z_copies, axis=0)
        p_family = float((1 + np.sum(strongest[1:] >= strongest[0])) / len(strongest))
    return {
        "series_names": np.asarray(names),
        "rels_analysis": rels[ana],
        "observed": observed,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "stat": stat,
        "p": p,
        "z": z,
        "p_family": np.float32(p_family),
        "flag": np.bool_(np.isfinite(p_family) and p_family <= flag_alpha),
        "flag_alpha": np.float32(flag_alpha),
        # smallest p the permutation count can resolve; if p_floor > flag_alpha the flag can never trigger
        "p_floor": np.float32(1.0 / len(centers)),
        "n_peaks_used": np.int32(peaks.size),
    }


def align_shifts_to_analysed_frames(shifts: np.ndarray, n_frames: int, trimmed: int = 0, bad_segments=()) -> np.ndarray | None:
    """Rows of `shifts` that belong to the analysed (photobleaching-corrected) movie.

    The shifts file either already matches the cleaned movie (the bad segments were removed before motion correction) or has one row per
    frame of the uncleaned movie, in which case the rows inside `bad_segments` ([start, end) ranges, qc/bad_segments.json) are dropped. In
    both cases the first `trimmed` frames the photobleaching step dropped (qc/photobleaching_qc_astro.json) are dropped last.
    Returns the matching rows (`shifts` itself when it matches already), or None when no reading of the file gives `n_frames` rows.
    """
    shifts = np.asarray(shifts)
    n, t = int(n_frames), max(int(trimmed), 0)
    candidates = [shifts]
    if len(bad_segments):
        keep = np.ones(len(shifts), dtype=bool)
        for start, end in bad_segments:
            keep[int(start):int(end)] = False
        candidates.append(shifts[keep])
    for rows in candidates:
        if len(rows) == n:
            return rows
        if t and len(rows) == n + t:
            return rows[t:]
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
