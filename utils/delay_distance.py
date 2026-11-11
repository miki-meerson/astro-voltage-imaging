"""The delay between the response of a cluster and the neuron's own peak, and the distances between a cluster and its neuron that the delay is compared with (the panel "Delay and
distance" of the Hypotheses tab, app/services/cluster_map.py `cluster_delays`). Pure functions of arrays: no files, no Flask.

Delay = time of the cluster's response peak minus time of the neuron's peak, both in the spike-triggered means (frames relative to the spike), so a positive delay means the
cluster peaks after the neuron. A peak is the extremum of its trace inside a window, in the direction of the response (up for a depolarizing cluster and for the neuron's calcium,
down for a hyperpolarizing cluster), refined below one frame by the parabola through the peak frame and its two neighbours. A peak on the edge of its window is not a clear peak
(the trace was still rising or falling): it is flagged and the page can leave it out.

The distances (full-resolution pixels; the recordings carry no pixel size) are several ways to say how far a cluster is from the neuron, because a dendrite is long and thin and
the distance between centres says little about how far along it the cluster is:
    center-center    between the centre of the ROI (of the nearest part for a merged ROI) and the centre of the cluster;
    nearest-nearest  between the nearest pixels of the ROI and of the cluster (0 when they touch or overlap);
    center-nearest   from the centre of the cluster to the nearest pixel of the ROI;
    along            along the ROI's long axis (its first principal axis), from the ROI's centre to the cluster's centre;
    across           across it (the distance from the long axis).
`along` and `across` need one ROI with a defined shape: they are None for a merged ROI (parts far apart) and for a ROI of fewer than three pixels."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import linregress, spearmanr

DISTANCE_MEASURES = ("center-center", "nearest-nearest", "center-nearest", "along", "across")
MIN_POINTS = 3  # fewest points to give a correlation and a slope


def peak_time(trace, frames, window: tuple[int, int], direction: int) -> tuple[float, bool]:
    """(frame of the peak, whether the peak is on the edge of the window) of `trace` (values at `frames`, consecutive and increasing) inside the frames of `window` (both ends
    included): the highest value for direction +1, the lowest for -1. The frame is refined below one frame by a parabola through the peak and its neighbours (as long as it
    is not on the edge). Raises ValueError when no frame of the trace is inside the window."""
    values, frames = np.asarray(trace, dtype=float) * direction, np.asarray(frames, dtype=int)
    inside = np.flatnonzero((frames >= window[0]) & (frames <= window[1]))
    if not inside.size:
        raise ValueError(f"No frame of the trace ({frames[0]}..{frames[-1]}) is inside the window {tuple(window)}.")
    k = int(inside[np.argmax(values[inside])])
    on_edge = k in (inside[0], inside[-1])
    if on_edge:
        return float(frames[k]), True
    before, here, after = values[k - 1], values[k], values[k + 1]
    curvature = before - 2 * here + after
    return float(frames[k] + (0.5 * (before - after) / curvature if curvature else 0.0)), False


def distance_measures(roi_ys, roi_xs, cluster_ys, cluster_xs, origin, single_shape: bool = True) -> tuple[dict[str, float | None], dict[str, list | None]]:
    """(distances, paths) of DISTANCE_MEASURES between a ROI (its pixels) and a cluster (its pixels). `paths` has, for every measure, the points [[y, x], ...] the distance is measured
    between, for the page to draw (`along` goes from the ROI's centre to the foot of the cluster's centre on the long axis, `across` from there to the cluster's centre).
    `origin` is the (y, x) centre the `center-center` distance starts from (the ROI's centre, or the nearest part's for a merged ROI); `single_shape` False (a merged ROI) leaves
    `along` and `across` out (None in both)."""
    roi = np.column_stack([np.asarray(roi_ys, dtype=float), np.asarray(roi_xs, dtype=float)])
    cluster = np.column_stack([np.asarray(cluster_ys, dtype=float), np.asarray(cluster_xs, dtype=float)])
    center, origin = cluster.mean(axis=0), np.asarray(origin, dtype=float)
    between = cdist(roi, cluster)
    nearest_roi, nearest_cluster = np.unravel_index(between.argmin(), between.shape)
    to_center = cdist(roi, center[None])[:, 0]
    nearest_to_center = int(to_center.argmin())
    distances = {"center-center": float(np.hypot(*(center - origin))), "nearest-nearest": float(between.min()), "center-nearest": float(to_center.min()), "along": None, "across": None}
    paths = {"center-center": [origin, center], "nearest-nearest": [roi[nearest_roi], cluster[nearest_cluster]], "center-nearest": [roi[nearest_to_center], center], "along": None, "across": None}
    if single_shape and len(roi) >= MIN_POINTS:
        centre = roi.mean(axis=0)
        axis = np.linalg.svd(roi - centre, full_matrices=False)[2][0]  # the direction the ROI is longest in
        offset = center - centre
        along = float(offset @ axis)
        foot = centre + along * axis
        distances["along"], distances["across"] = abs(along), float(np.hypot(*(offset - along * axis)))
        paths["along"], paths["across"] = [centre, foot], [foot, center]
    return distances, {measure: None if path is None else [[float(y), float(x)] for y, x in path] for measure, path in paths.items()}


def delay_distance_stats(distances, delays) -> dict | None:
    """How the delay (seconds) goes with the distance (pixels): {n, rho, p (Spearman), slope (seconds per pixel), intercept (seconds), slope_p (linear fit)}, or None with fewer
    than MIN_POINTS points or when the distances do not vary (nothing to correlate)."""
    x, y = np.asarray(distances, dtype=float), np.asarray(delays, dtype=float)
    if len(x) < MIN_POINTS or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rho, p = spearmanr(x, y)
    fit = linregress(x, y)
    return {"n": int(len(x)), "rho": float(rho), "p": float(p), "slope": float(fit.slope), "intercept": float(fit.intercept), "slope_p": float(fit.pvalue)}
