"""STA result summaries read from the cluster outputs (significant cells per result type, anatomy clusters in a crop)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.services import caches
from app.services.run_views import latest_completed_run
from app.services.runs import run_dir

STA_RESULT_TYPES = [
    ("all", "All"),
    ("simple", "Simple"),
    ("complex", "Complex"),
    ("anatomy", "Anatomy"),
    ("behavioral", "Behavioral"),
]


def sta_significant_summary(settings, conn, experiment_id: int) -> dict:
    run = latest_completed_run(conn, experiment_id, "sta")
    if run is None:
        return {"cells": {}, "result_types": []}
    run_id = int(run["id"])
    sta_dir = run_dir(settings, run_id, bool(run["is_test"])) / "sta_algorithm_home" / "sta-analysis"
    summary_csv = sta_dir / "sta_maps_batch_summary.csv"
    stat_key = int(summary_csv.stat().st_mtime_ns) if summary_csv.exists() else 0
    size_key = int(summary_csv.stat().st_size) if summary_csv.exists() else 0
    cache_key = (run_id, stat_key, size_key)
    cached = caches.STA_SIGNIFICANT.get(cache_key)
    if cached is not None:
        return cached
    if not sta_dir.exists():
        return {"cells": {}, "result_types": []}
    try:
        if summary_csv.exists():
            import pandas as pd

            result_df = pd.read_csv(summary_csv).rename(columns={"spike_group": "result_type"})
        else:
            from utils.db_utils import collect_cluster_result_metadata

            result_df = collect_cluster_result_metadata(sta_dir)
    except Exception:
        return {"cells": {}, "result_types": []}
    if result_df.empty or "cell_id" not in result_df or "n_significant_clusters" not in result_df:
        return {"cells": {}, "result_types": []}
    if "version" in result_df:
        result_df = result_df.sort_values("version").groupby(["cell_id", "result_type"], as_index=False).tail(1)
    result_df["n_significant_clusters"] = result_df["n_significant_clusters"].fillna(0).astype(int)
    counts = result_df.groupby("cell_id")["n_significant_clusters"].sum()
    significant = result_df[result_df["n_significant_clusters"] > 0]
    result_type_rows = []
    if not significant.empty:
        by_type = (
            significant.groupby("result_type", as_index=False)
            .agg(
                neurons=("cell_id", "nunique"),
                clusters=("n_significant_clusters", "sum"),
            )
            .sort_values(["clusters", "neurons", "result_type"], ascending=[False, False, True])
        )
        alpha = None
        try:
            alpha = float(json.loads((sta_dir / "sta_maps_used_parameters.json").read_text(encoding="utf-8")).get("cluster_alpha"))
        except Exception:
            pass
        tested = result_df
        if "skip_reason" in tested:
            tested = tested[tested["skip_reason"].isna()]
        tested_by_type = tested.groupby("result_type")["cell_id"].nunique().to_dict()
        result_type_rows = []
        for row in by_type.itertuples():
            item = {"result_type": str(row.result_type), "neurons": int(row.neurons), "clusters": int(row.clusters)}
            n_tested = int(tested_by_type.get(str(row.result_type), 0))
            if alpha and n_tested:
                # each cell's test is built to give a false positive in at most `alpha` of cells, so this many cells are expected by luck alone
                from scipy.stats import binom

                item.update({
                    "cells_tested": n_tested, "alpha": alpha, "expected_by_chance": n_tested * alpha,
                    "p_vs_chance": float(binom.sf(int(row.neurons) - 1, n_tested, alpha)),
                })
            result_type_rows.append(item)
    result = {
        "cells": {int(cell_id): int(count) for cell_id, count in counts.items() if int(count) > 0},
        "result_types": result_type_rows,
        "sta_run_id": run_id,
        "source": "sta_maps_batch_summary.csv" if summary_csv.exists() else "cluster_npz_files",
    }
    caches.STA_SIGNIFICANT.put(cache_key, result)
    return result


def anatomy_clusters_in_crop(sta_npz_path: Path, crop: dict) -> list[dict]:
    """Significant anatomy clusters of this cell, in the pixel coordinates of the cropped STA movie. [] when unavailable.

    Reads only the small keys of the anatomy result (never the permutations), so it is fast.
    """
    from utils.cluster_context import significant_clusters

    path = sta_npz_path.with_name("astro_sta_cluster_permutation_anatomy.npz")
    if not path.exists():
        return []
    out = []
    with np.load(path, allow_pickle=True) as d:
        if "anatomy_crop_bounds" not in d.files:
            return []
        ay0, _, ax0, _ = (int(v) for v in np.asarray(d["anatomy_crop_bounds"]).ravel())
        b = int(np.asarray(d["spatial_bin"]).item()) if "spatial_bin" in d.files else 1
        aw = np.asarray(d["analysis_window"], dtype=int)
        for c in significant_clusters(d, limit=None):
            m3 = np.asarray(c["mask3d"], dtype=bool)
            frames = np.flatnonzero(m3.any(axis=(1, 2)))
            ys, xs = np.nonzero(m3.any(axis=0))
            pix = set()
            for y, x in zip(ys.tolist(), xs.tolist()):  # bin units -> full-resolution pixels (blocks), then into the crop
                for dy in range(b):
                    for dx in range(b):
                        yy, xx = (ay0 + y) * b + dy - int(crop["y0"]), (ax0 + x) * b + dx - int(crop["x0"])
                        if 0 <= yy < int(crop["height"]) and 0 <= xx < int(crop["width"]):
                            pix.add((yy, xx))
            if not pix:
                continue
            arr = np.array(sorted(pix))
            out.append({
                "sign": str(c.get("sign", "")), "p": float(c.get("p_value", float("nan"))), "mass": float(c["mass"]),
                "first_rel": int(aw[0] + frames.min()), "last_rel": int(aw[0] + frames.max()),
                "centroid": [float(arr[:, 0].mean()), float(arr[:, 1].mean())],
                "pixels": arr.tolist(),
            })
    return out
