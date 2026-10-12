"""Statistics helpers for STA cluster-permutation maps.

All functions are pure numpy/scipy so they can be unit tested without movies.
Maps are (..., H, W) arrays; time-resolved maps are (T, H, W) and permutation stacks (P, T, H, W).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

NULL_STANDARDIZATION_MODES = ("none", "per_pixel_z")


def smooth_spatial(maps: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian smoothing over the last two (spatial) axes only; sigma in analysis pixels."""
    if not sigma_px or sigma_px <= 0:
        return maps
    sigma = [0.0] * (maps.ndim - 2) + [float(sigma_px), float(sigma_px)]
    return gaussian_filter(np.asarray(maps, dtype=np.float32), sigma=sigma, mode="nearest")


def null_sd_map(perm_effects: np.ndarray, floor_quantile: float = 0.05) -> np.ndarray:
    """Per-pixel null standard deviation from permutation effects (P, T, H, W) -> (H, W).

    Variance over permutations is averaged over time, then square-rooted. The result is floored at the
    `floor_quantile` of the map (and a tiny constant) so near-zero-variance pixels cannot blow up z-scores.
    """
    var = np.var(np.asarray(perm_effects, dtype=np.float32), axis=0)
    sd = np.sqrt(var.mean(axis=0))
    floor = max(float(np.quantile(sd, floor_quantile)), 1e-12)
    return np.maximum(sd, floor).astype(np.float32)


def standardize(maps: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Divide (..., H, W) maps by a per-pixel sd map (H, W)."""
    return (np.asarray(maps, dtype=np.float32) / sd).astype(np.float32)


def per_sign_thresholds(perm_effects: np.ndarray, voxel_alpha: float) -> tuple[float, float]:
    """(positive, negative) voxel thresholds for the opt-in `per_sign_null` rule; both returned as positive magnitudes.

    One-sided quantiles of the SIGNED surrogate values: a voxel is a candidate depolarizing voxel when
    value >= q(1 - voxel_alpha) and a candidate hyperpolarizing voxel when value <= q(voxel_alpha). Each sign then
    has a null per-voxel false-positive rate of exactly voxel_alpha, even when the surrogate distribution is skewed
    (the legacy single quantile of |value| gives skewed noise very unequal rates per sign). Note this is voxel_alpha
    PER SIGN, i.e. twice the legacy per-sign rate in the symmetric case (legacy: voxel_alpha split over both signs).
    """
    pos = float(np.quantile(perm_effects, 1.0 - float(voxel_alpha)))
    neg = float(-np.quantile(perm_effects, float(voxel_alpha)))
    return pos, neg


def max_mass_by_sign(records: list[dict]) -> tuple[float, float]:
    """(max positive-cluster mass, max negative-cluster mass) of a cluster-record list (0.0 when a sign has none)."""
    pos = max((float(r["mass"]) for r in records if r["sign"] == "positive"), default=0.0)
    neg = max((float(r["mass"]) for r in records if r["sign"] == "negative"), default=0.0)
    return pos, neg


def per_sign_p_values(mass: float, sign: str, null_pos: np.ndarray, null_neg: np.ndarray) -> tuple[float, float]:
    """(p_sign, p_corrected) for one cluster under `per_sign_null`.

    p_sign is the usual max-mass permutation p (with +1 correction) against the null of the cluster's OWN sign.
    p_corrected = min(1, 2 * p_sign): Bonferroni over the two signs, so the cell-level false-positive rate
    stays <= cluster_alpha when significance is p_corrected <= cluster_alpha.
    """
    null = np.asarray(null_pos if sign == "positive" else null_neg)
    p_sign = float((1 + np.sum(null >= mass)) / (len(null) + 1))
    return p_sign, min(1.0, 2.0 * p_sign)


def min_footprint_in_analysis_px(min_footprint_fullres_px: float, spatial_bin: int) -> int:
    """Convert a minimum footprint in full-resolution px^2 to analysis (binned) px^2. 0 disables."""
    if not min_footprint_fullres_px or min_footprint_fullres_px <= 0:
        return 0
    return max(1, int(np.ceil(float(min_footprint_fullres_px) / max(1, int(spatial_bin)) ** 2)))


def cluster_extent(mask3d: np.ndarray) -> tuple[int, int]:
    """(spatial footprint in px, number of frames spanned) of a boolean (T, H, W) cluster mask."""
    mask3d = np.asarray(mask3d, dtype=bool)
    return int(mask3d.any(axis=0).sum()), int(mask3d.any(axis=(1, 2)).sum())


def passes_extent(mask3d: np.ndarray, min_footprint_px: int = 0, min_frames: int = 0) -> bool:
    if not min_footprint_px and not min_frames:
        return True
    footprint, frames = cluster_extent(mask3d)
    return footprint >= int(min_footprint_px) and frames >= int(min_frames)


def cluster_trace_wide(
    sta_movie: np.ndarray,
    pre_frames: int,
    post_frames: int,
    baseline_window: tuple[int, int],
    mask2d: np.ndarray,
    half_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean effect trace of a cluster footprint over a window wider than the tested one (display only).

    sta_movie is the saved observed STA, rel = -pre_frames .. +post_frames. The trace is
    STA(rel) - baseline_map (baseline_map = mean over `baseline_window`, per pixel), averaged over `mask2d`:
    the same quantity as the effect movie, so inside the tested window it equals the usual cluster trace.
    The window is -half_window .. +half_window, clipped to what the file contains. Frames outside the tested
    analysis window are descriptive only: they are not part of the cluster test.
    """
    rels = np.arange(-int(pre_frames), int(post_frames) + 1)
    base = np.flatnonzero((rels >= baseline_window[0]) & (rels <= baseline_window[1]))
    baseline_map = np.asarray(sta_movie[base], dtype=np.float32).mean(axis=0)
    sel = np.flatnonzero((rels >= -int(half_window)) & (rels <= int(half_window)))
    mask2d = np.asarray(mask2d, dtype=bool)
    trace = (np.asarray(sta_movie[sel], dtype=np.float32)[:, mask2d] - baseline_map[mask2d]).mean(axis=1)
    return rels[sel], trace.astype(np.float32)
