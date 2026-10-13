"""STA result summaries read from the cluster outputs (significant cells per result type, anatomy clusters in a crop)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from app.db import neuron_repository
from app.services import caches
from app.services.runs import run_dir
from utils import globals_utils as g
from utils.cluster_context import significant_clusters
from utils.cluster_result import cluster_basic_fields, cluster_pixels_fullres
from utils.db_utils import collect_cluster_result_metadata
from utils.paths_utils import PathKind, path_for, sta_dir_of_run
from utils.sta_trace_utils import pixels_in_crop

STA_RESULT_TYPES = [
    ("all", "All"),
    ("simple", "Simple"),
    ("complex", "Complex"),
    ("anatomy", "Anatomy"),
    ("behavioral", "Behavioral"),
]


def sta_significant_summary(settings, experiment_id: int) -> dict:
    run = neuron_repository.latest_completed_run(settings, experiment_id, "sta")
    if run is None:
        return {"cells": {}, "result_types": []}
    run_id = int(run["id"])
    sta_dir = sta_dir_of_run(run_dir(settings, run_id, bool(run["is_test"])))
    summary_csv = path_for(PathKind.STA_SUMMARY_CSV, sta_dir)
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
            result_df = pd.read_csv(summary_csv).rename(columns={"spike_group": "result_type"})
        else:
            result_df = collect_cluster_result_metadata(sta_dir)
    except (OSError, ValueError):  # an unreadable or empty summary file: nothing to count
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
            used = json.loads(path_for(PathKind.STA_USED_PARAMETERS_JSON, sta_dir).read_text(encoding="utf-8"))
        except FileNotFoundError:  # the chance estimate is an extra: shown only when the job saved its parameters
            used = {}
        if used.get("cluster_alpha") is not None:
            alpha = float(used["cluster_alpha"])
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
                item.update({
                    "cells_tested": n_tested, "alpha": alpha, "expected_by_chance": n_tested * alpha,
                    "p_vs_chance": float(binom.sf(int(row.neurons) - 1, n_tested, alpha)),
                })
            result_type_rows.append(item)
    result = {
        "cells": {int(cell_id): int(count) for cell_id, count in counts.items() if int(count) > 0},
        "result_types": result_type_rows,
        "sta_run_id": run_id,
        "source": g.STA_SUMMARY_CSV if summary_csv.exists() else "cluster_npz_files",
    }
    caches.STA_SIGNIFICANT.put(cache_key, result)
    return result


def anatomy_clusters_in_crop(anatomy_npz: Path, crop: dict) -> list[dict]:
    """Significant anatomy clusters of this cell (read from its anatomy result file), in the pixel coordinates of the cropped STA movie.

    Reads only the small keys of the anatomy result (never the permutations), so it is fast.
    """
    out = []
    with np.load(anatomy_npz, allow_pickle=True) as d:
        ay0, _, ax0, _ = (int(v) for v in np.asarray(d["anatomy_crop_bounds"]).ravel())
        b = int(np.asarray(d["spatial_bin"]).item())
        window_start = int(np.asarray(d["analysis_window"], dtype=int)[0])
        for c in significant_clusters(d, limit=None):
            ys, xs = cluster_pixels_fullres(np.asarray(c["mask3d"], dtype=bool).any(axis=0), (ay0, 0, ax0, 0), b)
            pixels = np.unique(np.array(pixels_in_crop(ys, xs, crop)).reshape(-1, 2), axis=0)  # sorted, no duplicates
            if not len(pixels):
                continue
            out.append({
                **cluster_basic_fields(c, window_start),
                "centroid": [float(pixels[:, 0].mean()), float(pixels[:, 1].mean())],
                "pixels": pixels.tolist(),
            })
    return out
