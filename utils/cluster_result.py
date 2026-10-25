"""Read one saved cluster-permutation result (astro_sta_cluster_permutation_<type>.npz): the significant clusters, their 2D footprints and the
wide dF/F trace of each. The ONE implementation, used by the cell page (app/services/visualizations.py, which caches it in memory) and by
utils/cluster_map_summary.py (the experiment-level cluster summary). No Flask, no app imports."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.clusters_utils import cluster_mask2d_from_cluster, extract_clusters, mask_mean_trace, saved_cluster_args
from utils.sta_stats_utils import cluster_trace_wide

# Cluster traces are DISPLAYED over -N..+N frames around the spike (clipped to the saved STA window). Only the analysis window
# (default -10..+10) is tested; the rest is descriptive and shaded differently.
TRACE_HALF_WINDOW_FRAMES = 50


def cluster_pixels_fullres(mask2d: np.ndarray, crop, spatial_bin: int) -> tuple[np.ndarray, np.ndarray]:
    """Full-resolution pixel coordinates (ys, xs) covered by a 2D cluster footprint given in bin units (crop = (y0, y1, x0, x1) of the
    anatomy crop in bin units, or None). Each bin is a block of spatial_bin x spatial_bin pixels. The one bin -> pixel mapping."""
    ys, xs = np.nonzero(mask2d)
    y0, x0 = (int(crop[0]), int(crop[2])) if crop is not None else (0, 0)
    b = int(spatial_bin)
    block_y, block_x = np.meshgrid(np.arange(b), np.arange(b), indexing="ij")
    full_y = (y0 + ys)[:, None] * b + block_y.ravel()[None, :]
    full_x = (x0 + xs)[:, None] * b + block_x.ravel()[None, :]
    return full_y.ravel(), full_x.ravel()


def cluster_center_fullres(mask2d: np.ndarray, crop, spatial_bin: int) -> tuple[float, float]:
    """Centre (y, x) of a 2D cluster footprint in full-resolution pixel coordinates (crop = (y0, y1, x0, x1) of the anatomy crop or None)."""
    ys, xs = cluster_pixels_fullres(mask2d, crop, spatial_bin)
    return float(ys.mean()), float(xs.mean())


def cluster_offset_from_roi(mask2d: np.ndarray, crop, spatial_bin: int, roi_yx: tuple[float, float]) -> tuple[float, float]:
    """(dy, dx) from the ROI centre to the centre of a cluster footprint, in full-resolution pixels."""
    cy, cx = cluster_center_fullres(mask2d, crop, spatial_bin)
    return cy - roi_yx[0], cx - roi_yx[1]


def tested_time_axis(analysis_window, n_steps: int, time_bin: int = 1) -> np.ndarray:
    """The time (frames relative to the spike) of each step of a saved effect movie: every frame of the tested window, or the centre of each time bin."""
    return int(analysis_window[0]) + time_bin * np.arange(n_steps) + (time_bin - 1) / 2


def cluster_basic_fields(cluster: dict, window_start: int, time_bin: int = 1) -> dict:
    """Sign (as saved), p-value, mass and the first/last frame of a cluster relative to the spike (window_start = first frame of the tested window;
    time_bin = frames per step of the effect movie, so the last frame is the end of the last bin)."""
    frames = np.flatnonzero(np.asarray(cluster["mask3d"], dtype=bool).any(axis=(1, 2)))
    return {
        "sign": str(cluster.get("sign", "")), "p": float(cluster["p_value"]), "mass": float(cluster["mass"]),
        "first_rel": int(window_start + time_bin * frames.min()), "last_rel": int(window_start + time_bin * frames.max() + time_bin - 1),
    }


def read_result_header(path: str | Path) -> dict:
    """The small keys of a saved cluster result that describe its movie (never the permutations): peaks (the kept spikes), pre_frames, post_frames, baseline_window, spatial_bin,
    analysis_window, time_bin and crop (the anatomy crop as (y0, y1, x0, x1) in bins, None for a result of the whole field)."""
    with np.load(path, allow_pickle=True) as d:
        return {
            "peaks": np.asarray(d["peaks"], dtype=np.int32),
            "pre_frames": int(np.asarray(d["pre_frames"]).item()), "post_frames": int(np.asarray(d["post_frames"]).item()),
            "baseline_window": [int(v) for v in np.asarray(d["baseline_window"]).ravel()],
            "spatial_bin": int(np.asarray(d["spatial_bin"]).item()), "analysis_window": [int(v) for v in np.asarray(d["analysis_window"]).ravel()],
            "time_bin": int(np.asarray(d["time_bin_frames"]).item()) if "time_bin_frames" in d else 1,
            "crop": tuple(int(v) for v in np.asarray(d["anatomy_crop_bounds"]).ravel()) if "anatomy_crop_bounds" in d else None,
        }


def cluster_records(effect: np.ndarray, t_rel: np.ndarray, clusters: list, meta: dict) -> list[dict]:
    """The significant clusters of a loaded result (the four values load_cluster_result returns) as records: the basic fields (sign as saved, p, mass, first and last frame),
    `mask` (the 2D footprint, in bin units), `t` and `trace` (the wide dF/F trace over the footprint, else the mean of the tested effect over it). A cluster whose footprint does
    not fit the effect movie is left out."""
    records = []
    for cluster, mask in clusters:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != effect.shape[1:]:
            continue
        wide = cluster.get("wide_trace")  # displayed over a wider window than the tested one
        t, trace = wide if wide is not None else (t_rel, mask_mean_trace(meta["trace_effect"], mask))
        records.append({**cluster_basic_fields(cluster, meta["tested_window"][0], meta["time_bin"]), "mask": mask, "t": np.asarray(t), "trace": np.asarray(trace).ravel()})
    return records


def load_cluster_result(path: str | Path):
    """(effect, t_rel, sig, metadata) of a saved cluster result.

    effect: observed effect movie (z-statistic that was tested); t_rel: time of each of its steps (frame numbers; bin centres when binned); sig: list of (cluster dict, 2D mask)
    for the significant clusters (cluster["wide_trace"] = (t, dF/F trace)); metadata: spatial_bin,
    behavioral_group, anatomy_crop_bounds, trace_effect, tested_window, time_bin (frames per step; 1 in files without binning). Signs are as saved (STA convention, see app/services/polarity.py).
    """
    head = read_result_header(path)
    time_bin = head["time_bin"]
    with np.load(path, allow_pickle=True) as d:
        effect = np.asarray(d["observed_effect_movie"], dtype=np.float32)
        # traces are drawn on a dF/F axis: use the raw effect, not the z-statistic that was tested
        trace_effect = np.asarray(d["observed_effect_raw_movie"], dtype=np.float32) if "observed_effect_raw_movie" in d else effect
        wide = dict(  # kept only for this call (can be large at bin 1)
            sta=np.asarray(d["observed_sta_movie"], dtype=np.float32), pre=head["pre_frames"], post=head["post_frames"], baseline_window=tuple(head["baseline_window"]),
        )
        clusters = extract_clusters(effect_movie=effect, **saved_cluster_args(d))  # honours per_sign_null when the file says so
        metadata = {
            "spatial_bin": head["spatial_bin"],
            "behavioral_group": str(np.asarray(d["behavioral_group"]).item()) if "behavioral_group" in d else None,
            "anatomy_crop_bounds": head["crop"],
            "trace_effect": trace_effect,
            "tested_window": tuple(head["analysis_window"]),
            "time_bin": time_bin,
        }
    t_rel = tested_time_axis(head["analysis_window"], effect.shape[0], time_bin)
    sig = []
    for cluster in clusters:
        if not cluster.get("is_significant", False):
            continue
        mask = cluster_mask2d_from_cluster(cluster)
        if mask is not None and mask.any():
            if mask.shape == wide["sta"].shape[1:]:
                cluster["wide_trace"] = cluster_trace_wide(
                    wide["sta"], wide["pre"], wide["post"], wide["baseline_window"], mask,
                    max(wide["pre"], wide["post"]) if time_bin > 1 else TRACE_HALF_WINDOW_FRAMES,  # a slow hypothesis shows its whole saved window
                )
            sig.append((cluster, mask))
    del wide
    return effect, t_rel, sig, metadata
