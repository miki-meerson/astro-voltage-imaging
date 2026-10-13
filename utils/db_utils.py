from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import re

from utils import globals_utils as g
from utils.paths_utils import PathKind, cell_dirs, path_for

DEFAULT_NEURON_DATASET_NAME = "neuron_dataset.pkl"

NEURON_INDEX_COLUMNS = [
    "cage",
    "mouse_name",
    "recording_date",
    "behavior",
    "recording_name",
    "neuron_id",
]

CLUSTER_INDEX_COLUMNS = NEURON_INDEX_COLUMNS + ["cluster_id"]

NEURON_VALUE_COLUMNS = [
    "all_peak_frames",
    "simple_peak_frames",
    "complex_peak_frames",
]

CLUSTER_VALUE_COLUMNS = [
    "sign",
    "is_significant",
    "p_value",
    "disqualify_latency",
    "latency_shared_peak_frame_for_kinetics",
    "all_mean_trace",
    "simple_mean_trace",
    "complex_mean_trace",
]

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


def _missing_required_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [c for c in required if c not in df.columns]


def ensure_neuron_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        if list(df.index.names) != NEURON_INDEX_COLUMNS:
            raise ValueError(
                f"Neuron index must be {NEURON_INDEX_COLUMNS}, got {list(df.index.names)}"
            )
        out = df.sort_index()
        missing_values = _missing_required_columns(out, NEURON_VALUE_COLUMNS)
        if missing_values:
            raise ValueError(f"Neuron dataset missing required value columns: {missing_values}")
        return out
    missing = [c for c in NEURON_INDEX_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot set neuron index. Missing columns: {missing}")
    out = df.set_index(NEURON_INDEX_COLUMNS).sort_index()
    missing_values = _missing_required_columns(out, NEURON_VALUE_COLUMNS)
    if missing_values:
        raise ValueError(f"Neuron dataset missing required value columns: {missing_values}")
    return out


def ensure_cluster_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        if list(df.index.names) != CLUSTER_INDEX_COLUMNS:
            raise ValueError(
                f"Cluster index must be {CLUSTER_INDEX_COLUMNS}, got {list(df.index.names)}"
            )
        out = df.sort_index()
        missing_values = _missing_required_columns(out, CLUSTER_VALUE_COLUMNS)
        if missing_values:
            raise ValueError(f"Cluster dataset missing required value columns: {missing_values}")
        return out
    missing = [c for c in CLUSTER_INDEX_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot set cluster index. Missing columns: {missing}")
    out = df.set_index(CLUSTER_INDEX_COLUMNS).sort_index()
    missing_values = _missing_required_columns(out, CLUSTER_VALUE_COLUMNS)
    if missing_values:
        raise ValueError(f"Cluster dataset missing required value columns: {missing_values}")
    return out

def load_params_dataset(params_dataset_path):
    params_df = pd.read_pickle(params_dataset_path)
    if not isinstance(params_df.index, pd.MultiIndex):
        params_df = params_df.set_index(CLUSTER_INDEX_COLUMNS)
    params_df = params_df.sort_index()
    params_reset = params_df.reset_index()

    return params_reset

def load_neuron_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(Path(path))
    return ensure_neuron_multiindex(df)


def load_cluster_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(Path(path))
    return ensure_cluster_multiindex(df)


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
        row["n_permutations"] = int(len(d["permutation_shifts"]))
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


def _infer_neuron_dataset_path_from_cluster_path(cluster_dataset_path: Path) -> Path:
    return Path(cluster_dataset_path).with_name(DEFAULT_NEURON_DATASET_NAME)


def load_normalized_dataset(
    cluster_dataset_path: Path,
    *,
    neuron_dataset_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_path = Path(cluster_dataset_path)
    neuron_path = (
        Path(neuron_dataset_path)
        if neuron_dataset_path is not None
        else _infer_neuron_dataset_path_from_cluster_path(cluster_path)
    )
    if not neuron_path.exists():
        raise FileNotFoundError(
            f"Missing required neuron dataset: {neuron_path}"
        )

    neuron_df = load_neuron_dataset(neuron_path)
    cluster_df = load_cluster_dataset(cluster_path)
    return neuron_df, cluster_df


def load_saved_npz(npz_path):
    with np.load(npz_path, allow_pickle=True) as d:
        effect = np.asarray(d["observed_effect_movie"], dtype=float)
        post_pre = np.asarray(d["observed_post_pre_map"], dtype=float)
        aw = np.asarray(d["analysis_window"], dtype=int)
        t_rel = np.arange(int(aw[0]), int(aw[1]) + 1)
        if len(t_rel) != effect.shape[0]:
            raise ValueError(f"{npz_path}: analysis_window mismatch")
        return {
            "effect": effect,
            "post_pre": post_pre,
            "sta_movie": np.asarray(d["observed_sta_movie"], dtype=np.float32),
            "pre_frames": int(np.asarray(d["pre_frames"]).item()),
            "post_frames": int(np.asarray(d["post_frames"]).item()),
            "baseline_frames": int(np.asarray(d["baseline_frames"]).item()),
            "peaks": np.asarray(d["peaks"], dtype=np.int32),
            "analysis_window": aw,
            "baseline_window": np.asarray(d["baseline_window"], dtype=int),
            "threshold": float(np.asarray(d["cluster_threshold"]).item()),
            "null_max": np.asarray(d["null_max_cluster_mass"], dtype=float),
            "cluster_alpha": float(np.asarray(d["cluster_alpha"]).item()),
            "cluster_connectivity": int(np.asarray(d["cluster_connectivity"]).item()),
            "min_cluster_size": int(np.asarray(d["min_cluster_size"]).item()),
            "min_cluster_mass": float(np.asarray(d["min_cluster_mass"]).item()),
            "merge_similar_clusters": bool(np.asarray(d["merge_similar_clusters"]).item()),
            "merge_max_distance_px": float(np.asarray(d["merge_max_distance_px"]).item()),
            "merge_min_trace_corr": float(np.asarray(d["merge_min_trace_corr"]).item()),
            "spatial_bin": int(np.asarray(d["spatial_bin"]).item()),
            # dF/F-like effect for traces; observed_effect_movie is the TESTED statistic (z in per_pixel_z mode); the raw one is saved only when they differ
            "trace_effect": np.asarray(d["observed_effect_raw_movie"] if "observed_effect_raw_movie" in d else d["observed_effect_movie"], dtype=float),
            "min_footprint_px": int(np.asarray(d["min_cluster_footprint_px"]).item()),
            "min_frames": int(np.asarray(d["min_cluster_frames"]).item()),
            "significance_statistic": str(np.asarray(d["significance_statistic"]).item()),
            # opt-in per-sign rule (see docs/statistics.md); for such files "threshold" / "null_max" above are the
            # positive-sign values and the negative-sign ones are here (saved only when per_sign_null is on)
            "per_sign_null": bool(np.asarray(d["per_sign_null"]).item()),
            "threshold_neg": float(np.asarray(d["cluster_threshold_neg"]).item()) if "cluster_threshold_neg" in d else None,
            "null_max_pos": np.asarray(d["null_max_cluster_mass_pos"], dtype=float) if "null_max_cluster_mass_pos" in d else None,
            "null_max_neg": np.asarray(d["null_max_cluster_mass_neg"], dtype=float) if "null_max_cluster_mass_neg" in d else None,
            "t_rel": t_rel,
        }
