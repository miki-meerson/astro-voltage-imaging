"""Are the significant anatomy clusters of an experiment more alike than clusters found in fake spike trains?

Plain version: every cell has 200 saved "fake" versions of its STA (its spike times shifted in time, which breaks any real link
between spikes and the movie). We run the very same cluster search on each fake, so fake clusters are chosen exactly like real
ones. Then we build many fake groups (one fake per cell) and ask how often a fake group looks as striking as the real one.

Numbers compared (all from clusters of different cells, tested window only, because that is the part the fakes cover):
  cells_with_cluster   how many cells have a significant cluster
  one_sidedness        how lopsided the polarity is (share of the more common polarity; 0.5 = even mix)
  similarity_same_time average correlation between clusters' traces
  similarity_shift3    the same, but allowing one trace to be shifted up to +-3 frames
  distance_px          average distance of cluster centre from the cell's ROI centre (full-resolution pixels)
"""
from __future__ import annotations

import numpy as np

from utils.cluster_result import cluster_offset_from_roi
from utils.clusters_utils import cluster_mask2d_from_cluster, extract_clusters, saved_cluster_args

MAX_SHIFT = 3
STATS = ("cells_with_cluster", "one_sidedness", "similarity_same_time", "similarity_shift3", "distance_px")
# which direction counts as "more striking than chance" for each number
DIRECTION = {"cells_with_cluster": "high", "one_sidedness": "high", "similarity_same_time": "high", "similarity_shift3": "high", "distance_px": "low"}


def _rows_from_effect(effect, d, null_max, crop, spatial_bin, roi_center, null_max_neg=None):
    """Significant-cluster rows of one effect movie. null_max is the pooled null or, when the result was made with
    per_sign_null, the positive-sign null with null_max_neg the negative-sign one: the same rule as the saved result."""
    args = saved_cluster_args(d)
    args["null_max"] = null_max
    if args.get("per_sign_null"):
        args["null_max_neg"] = null_max_neg
    clusters = extract_clusters(effect_movie=effect, **args)
    rows = []
    for c in clusters:
        if not c.get("is_significant", False):
            continue
        mask2d = np.asarray(cluster_mask2d_from_cluster(c), dtype=bool)
        dy, dx = cluster_offset_from_roi(mask2d, crop, spatial_bin, roi_center)
        rows.append({
            "sign": 1 if str(c.get("sign", "")).lower() == "positive" else -1,  # +1 depolarizing, -1 hyperpolarizing
            "dist": float(np.hypot(dy, dx)),
            "trace": effect[:, mask2d].mean(axis=1).astype(float).round(5).tolist(),
        })
    return rows


def cell_records(npz_path, roi_center) -> dict:
    """{'real': [rows], 'fake': [[rows] for each of the cell's fakes]} for one cell (anatomy configuration)."""
    with np.load(npz_path, allow_pickle=True) as d:
        effect = np.asarray(d["observed_effect_movie"], dtype=np.float32)
        perm = d["perm_effects"]
        args = saved_cluster_args(d)  # per_sign_null results: null_max / null_max_neg are the two per-sign nulls
        null_max, null_max_neg = args["null_max"], args.get("null_max_neg")
        crop = np.asarray(d["anatomy_crop_bounds"], dtype=int).ravel()
        b = int(np.asarray(d["spatial_bin"]).item())
        real = _rows_from_effect(effect, d, null_max, crop, b, roi_center, null_max_neg)
        fake = []
        for i in range(perm.shape[0]):
            # a fake is judged against the other fakes, never against itself (leave-one-out, per sign when per_sign_null)
            others = np.delete(null_max, i)
            others_neg = np.delete(null_max_neg, i) if null_max_neg is not None else None
            fake.append(_rows_from_effect(np.asarray(perm[i], dtype=np.float32), d, others, crop, b, roi_center, others_neg))
    return {"real": real, "fake": fake}


def _lag_corr_matrix(M: np.ndarray, lag: int) -> np.ndarray:
    a, b = (M[:, lag:], M[:, : M.shape[1] - lag]) if lag >= 0 else (M[:, :lag], M[:, -lag:])
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    num = a @ b.T
    den = np.sqrt((a ** 2).sum(1))[:, None] * np.sqrt((b ** 2).sum(1))[None, :]
    return num / np.maximum(den, 1e-12)


def group_stats(rows: list[dict]) -> dict:
    """The five numbers for one group of clusters. rows carry 'cell' too. NaN when too few clusters to compare."""
    out = {k: float("nan") for k in STATS}
    out["cells_with_cluster"] = float(len({r["cell"] for r in rows}))
    if not rows:
        return out
    signs = np.array([r["sign"] for r in rows])
    out["one_sidedness"] = float(max((signs > 0).mean(), (signs < 0).mean()))
    out["distance_px"] = float(np.mean([r["dist"] for r in rows]))
    cell_of = np.array([r["cell"] for r in rows])
    if len(rows) >= 2:
        M = np.array([r["trace"] for r in rows], dtype=float)
        pair = (cell_of[:, None] != cell_of[None, :]) & np.triu(np.ones((len(rows), len(rows)), bool), 1)
        if pair.any():
            c0 = _lag_corr_matrix(M, 0)
            best = np.max([_lag_corr_matrix(M, L) for L in range(-MAX_SHIFT, MAX_SHIFT + 1)], axis=0)
            # for a pair (i, j) the matrix at lag L correlates i shifted one way with j; the lag set is symmetric, so this is the best shift either way
            out["similarity_same_time"] = float(c0[pair].mean())
            out["similarity_shift3"] = float(best[pair].mean())
    return out


def group_test(records: dict[int, dict], n_draws: int = 3000, seed: int = 0) -> dict:
    """records: {cell: {'real': rows, 'fake': [rows,...]}}. Returns real values, fake percentiles and one-sided p-values."""
    cells = sorted(records)
    real_rows = [dict(r, cell=c) for c in cells for r in records[c]["real"]]
    real = group_stats(real_rows)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in STATS}
    n_fake = min(len(records[c]["fake"]) for c in cells)
    for _ in range(n_draws):
        pick = rng.integers(n_fake, size=len(cells))
        rows = [dict(r, cell=c) for c, i in zip(cells, pick) for r in records[c]["fake"][int(i)]]
        s = group_stats(rows)
        for k in STATS:
            draws[k].append(s[k])
    result = {"n_cells": len(cells), "n_real_clusters": len(real_rows), "n_draws": n_draws, "n_fakes_per_cell": n_fake, "stats": {}}
    fake_rows = [r for c in cells for f in records[c]["fake"] for r in f]
    result["polarity_in_fakes"] = {
        "n_fake_clusters": len(fake_rows), "n_fake_results": int(sum(len(records[c]["fake"]) for c in cells)),
        "hyperpolarizing_share_fake": float(np.mean([r["sign"] < 0 for r in fake_rows])) if fake_rows else None,
        "hyperpolarizing_share_real": float(np.mean([r["sign"] < 0 for r in real_rows])) if real_rows else None,
    }
    for k in STATS:
        v = np.array(draws[k], dtype=float)
        v = v[np.isfinite(v)]
        if not np.isfinite(real[k]) or v.size < 50:
            result["stats"][k] = {"real": None if not np.isfinite(real[k]) else float(real[k]), "valid_draws": int(v.size), "p": None}
            continue
        more = (v >= real[k]) if DIRECTION[k] == "high" else (v <= real[k])
        result["stats"][k] = {
            "real": float(real[k]), "fake_median": float(np.median(v)), "fake_low": float(np.percentile(v, 2.5)), "fake_high": float(np.percentile(v, 97.5)),
            "p": float((1 + more.sum()) / (v.size + 1)), "direction": DIRECTION[k], "valid_draws": int(v.size),
        }
    return result
