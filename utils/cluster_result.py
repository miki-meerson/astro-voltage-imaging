"""Read one saved cluster-permutation result (astro_sta_cluster_permutation_<type>.npz): the significant clusters, their 2D footprints and the
wide dF/F trace of each. The ONE implementation, used by the cell page (app/services/visualizations.py, which adds an in-memory cache) and by
utils/anatomy_summary.py (the experiment-level anatomy clusters summary). No Flask, no app imports."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.clusters_utils import cluster_mask2d_from_cluster, extract_clusters, saved_cluster_args
from utils.sta_stats_utils import cluster_trace_wide

# Cluster traces are DISPLAYED over -N..+N frames around the spike (clipped to the saved STA window). Only the analysis window
# (default -10..+10) is tested; the rest is descriptive and shaded differently.
TRACE_HALF_WINDOW_FRAMES = 50


def cluster_center_fullres(mask2d: np.ndarray, crop, spatial_bin: int) -> tuple[float, float]:
    """Centre (y, x) of a 2D cluster footprint in full-resolution pixel coordinates (crop = (y0, y1, x0, x1) of the anatomy crop or None)."""
    ys, xs = np.nonzero(mask2d)
    y0, x0 = (int(crop[0]), int(crop[2])) if crop is not None else (0, 0)
    b = int(spatial_bin)
    # bin units -> full-resolution pixel coordinates (pixel centres)
    return (y0 + float(ys.mean()) + 0.5) * b - 0.5, (x0 + float(xs.mean()) + 0.5) * b - 0.5


def cluster_offset_from_roi(mask2d: np.ndarray, crop, spatial_bin: int, roi_yx: tuple[float, float]) -> tuple[float, float]:
    """(dy, dx) from the ROI centre to the centre of a cluster footprint, in full-resolution pixels."""
    cy, cx = cluster_center_fullres(mask2d, crop, spatial_bin)
    return cy - roi_yx[0], cx - roi_yx[1]


def cluster_basic_fields(cluster: dict, window_start: int) -> dict:
    """Sign (as saved), p-value, mass and the first/last frame of a cluster relative to the spike (window_start = first frame of the tested window)."""
    frames = np.flatnonzero(np.asarray(cluster["mask3d"], dtype=bool).any(axis=(1, 2)))
    return {
        "sign": str(cluster.get("sign", "")), "p": float(cluster["p_value"]), "mass": float(cluster["mass"]),
        "first_rel": int(window_start + frames.min()), "last_rel": int(window_start + frames.max()),
    }


def load_cluster_result(path: str | Path):
    """(effect, t_rel, sig, metadata) of a saved cluster result.

    effect: observed effect movie (z-statistic that was tested); t_rel: frame numbers of the tested window; sig: list of (cluster dict, 2D mask)
    for the significant clusters (cluster["wide_trace"] = (t, dF/F trace)); metadata: spatial_bin,
    behavioral_group, anatomy_crop_bounds, trace_effect, tested_window. Signs are as saved (STA convention, see app/services/polarity.py).
    """
    with np.load(path, allow_pickle=True) as d:
        effect = np.asarray(d["observed_effect_movie"], dtype=np.float32)
        # traces are drawn on a dF/F axis: use the raw effect, not the z-statistic that was tested
        trace_effect = np.asarray(d["observed_effect_raw_movie"], dtype=np.float32) if "observed_effect_raw_movie" in d else effect
        aw = np.asarray(d["analysis_window"], dtype=int)
        wide = dict(  # kept only for this call (can be large at bin 1)
            sta=np.asarray(d["observed_sta_movie"], dtype=np.float32),
            pre=int(np.asarray(d["pre_frames"]).item()),
            post=int(np.asarray(d["post_frames"]).item()),
            baseline_window=tuple(int(v) for v in np.asarray(d["baseline_window"]).ravel()),
        )
        clusters = extract_clusters(effect_movie=effect, **saved_cluster_args(d))  # honours per_sign_null when the file says so
        metadata = {
            "spatial_bin": int(np.asarray(d["spatial_bin"]).item()),
            "behavioral_group": str(np.asarray(d["behavioral_group"]).item()) if "behavioral_group" in d else None,
            "anatomy_crop_bounds": tuple(np.asarray(d["anatomy_crop_bounds"], dtype=int).tolist()) if "anatomy_crop_bounds" in d else None,
            "trace_effect": trace_effect,
            "tested_window": (int(aw[0]), int(aw[1])),
        }
    t_rel = np.arange(int(aw[0]), int(aw[1]) + 1)
    sig = []
    for cluster in clusters:
        if not cluster.get("is_significant", False):
            continue
        mask = cluster_mask2d_from_cluster(cluster)
        if mask is not None and mask.any():
            if mask.shape == wide["sta"].shape[1:]:
                cluster["wide_trace"] = cluster_trace_wide(
                    wide["sta"], wide["pre"], wide["post"], wide["baseline_window"], mask, TRACE_HALF_WINDOW_FRAMES
                )
            sig.append((cluster, mask))
    del wide
    return effect, t_rel, sig, metadata
