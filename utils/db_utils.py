from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import re

from utils import globals_utils as g
from utils.paths_utils import PathKind, cell_dirs, path_for

CLUSTER_RESULT_WINDOW_KEYS = [
    "pre_window",
    "post_window",
    "analysis_window",
    "baseline_window",
]

CLUSTER_RESULT_ALGORITHM_KEYS = [
    "voxel_alpha",
    "cluster_alpha",
    "spatial_bin",
    "anatomy_spatial_bin",
    "anatomy_radius_binned",
    "anatomy_source_radius_binned",
    "anatomy_source_spatial_bin",
    "anatomy_fullres_radius_px",
    "min_spikes_for_cluster",
    "rng_seed",
    "pre_seconds",
    "post_seconds",
    "baseline_seconds",
    "analysis_half_window_seconds",
    "analysis_half_window_frames",
    "cluster_connectivity",
    "min_cluster_size",
    "min_cluster_mass",
    "merge_similar_clusters",
    "merge_max_distance_px",
    "merge_min_trace_corr",
    "null_standardization",
    "per_sign_null",
    "smooth_sigma_px",
    "smooth_sigma_fullres_px",
    "min_cluster_footprint_px",
    "min_cluster_frames",
    "significance_statistic",
]

CLUSTER_RESULT_HPARAM_KEYS = [
    *CLUSTER_RESULT_ALGORITHM_KEYS,
    *CLUSTER_RESULT_WINDOW_KEYS,
]


def parse_cluster_result_filename(path: str | Path) -> dict[str, int | str] | None:
    path = Path(path)
    prefix = g.RESULT_PREFIX
    if not path.stem.startswith(prefix):
        return None

    suffix = path.stem.removeprefix(prefix)
    if suffix.startswith("_"):
        suffix = suffix[1:]

    version = 0
    m = re.search(r"_v(\d+)$", suffix)
    if m:
        version = int(m.group(1))
        suffix = suffix[:m.start()]

    return {"result_type": suffix or g.ALL_GROUP, "version": version}


def _npz_metadata_value(npz_data, key: str, default=np.nan):
    if key not in npz_data:
        return default
    value = np.asarray(npz_data[key])
    value = value.item() if value.shape == () else value.tolist()
    return tuple(value) if isinstance(value, list) else value


def read_cluster_result_metadata(npz_path: str | Path) -> dict[str, Any] | None:
    npz_path = Path(npz_path)
    parsed = parse_cluster_result_filename(npz_path)
    if parsed is None:
        return None

    with np.load(npz_path, allow_pickle=True) as d:
        row = {
            "cell_id": int(np.asarray(d["cell_id"]).item()),
            "file_name": npz_path.name,
            "path": npz_path,
            **parsed,
        }
        row.update({key: _npz_metadata_value(d, key) for key in CLUSTER_RESULT_HPARAM_KEYS})
        row["n_permutations"] = int(len(d["permutation_positions"]))
        row["n_peaks"] = int(len(d["peaks"]))
        is_sig = [bool(c[6]) for c in np.asarray(d["cluster_summary"], dtype=object)]  # columns as written by save_cluster_result; 6 = is_significant
        row["n_significant_clusters"] = int(sum(is_sig))
        row["has_significant_cluster"] = bool(any(is_sig))
        return row


def collect_cluster_result_metadata(
    sta_dir: str | Path,
    cell_ids: list[int] | None = None,
) -> pd.DataFrame:
    sta_dir = Path(sta_dir)
    folders = (
        [path_for(PathKind.CELL_DIR, sta_dir, cell=int(cell_id)) for cell_id in cell_ids]
        if cell_ids is not None
        else list(cell_dirs(sta_dir).values())
    )

    rows = []
    for cell_dir in folders:
        for npz_path in sorted(cell_dir.glob(g.RESULT_GLOB)):
            row = read_cluster_result_metadata(npz_path)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)

