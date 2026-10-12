from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import re


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
    prefix = "astro_sta_cluster_permutation"
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

    return {"result_type": suffix or "all", "version": version}


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
            "cell_id": (
                int(np.asarray(d["cell_id"]).item())
                if "cell_id" in d
                else int(npz_path.parent.name.split("_", 1)[1])
            ),
            "file_name": npz_path.name,
            "path": npz_path,
            **parsed,
        }
        row.update({key: _npz_metadata_value(d, key) for key in CLUSTER_RESULT_HPARAM_KEYS})
        row["n_permutations"] = (
            int(len(d["permutation_shifts"]))
            if "permutation_shifts" in d
            else int(len(d["null_max_cluster_mass"])) if "null_max_cluster_mass" in d else np.nan
        )
        row["n_peaks"] = int(len(d["peaks"])) if "peaks" in d else np.nan
        if "cluster_summary" in d:
            clusters = np.asarray(d["cluster_summary"], dtype=object)
            is_sig = [
                bool(c.get("is_significant", False)) if isinstance(c, dict) else bool(c[6])
                for c in clusters
            ]
            row["n_significant_clusters"] = int(sum(is_sig))
            row["has_significant_cluster"] = bool(any(is_sig))
        else:
            row["n_significant_clusters"] = 0
            row["has_significant_cluster"] = False
        return row


def collect_cluster_result_metadata(
    sta_dir: str | Path,
    cell_ids: list[int] | None = None,
) -> pd.DataFrame:
    sta_dir = Path(sta_dir)
    cell_dirs = (
        [sta_dir / f"cell_{int(cell_id)}" for cell_id in cell_ids]
        if cell_ids is not None
        else sorted(
            [p for p in sta_dir.glob("cell_*") if p.is_dir()],
            key=lambda p: int(p.name.split("_", 1)[1]),
        )
    )

    rows = []
    for cell_dir in cell_dirs:
        for npz_path in sorted(cell_dir.glob("astro_sta_cluster_permutation*.npz")):
            row = read_cluster_result_metadata(npz_path)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def _infer_neuron_dataset_path_from_cluster_path(cluster_dataset_path: Path) -> Path:
    return Path(cluster_dataset_path).with_name(DEFAULT_NEURON_DATASET_NAME)


def load_sta_metadata(cell_dir: str | Path) -> dict:
    cell_dir = Path(cell_dir)
    neuron_sta = np.load(cell_dir / "neuron_sta.npz", allow_pickle=True)
    peak_params = np.load(cell_dir / "peak_detection_params.npz", allow_pickle=True)
    return {
        "pre_frames": int(np.asarray(neuron_sta["pre_frames"]).item()),
        "post_frames": int(np.asarray(neuron_sta["post_frames"]).item()),
        "sta_trace": np.asarray(neuron_sta["sta"]),
        "threshold_factor": float(np.asarray(peak_params["threshold_factor"]).item()),
        "min_distance": int(np.asarray(peak_params["min_distance"]).item()),
    }


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
            "baseline_frames": int(np.asarray(d["baseline_frames"]).item()) if "baseline_frames" in d else None,
            "peaks": np.asarray(d["peaks"], dtype=np.int32) if "peaks" in d else np.empty(0, dtype=np.int32),
            "analysis_window": aw,
            "baseline_window": np.asarray(d["baseline_window"], dtype=int),
            "threshold": float(np.asarray(d["cluster_threshold"]).item()),
            "null_max": np.asarray(d["null_max_cluster_mass"], dtype=float),
            "cluster_alpha": float(np.asarray(d["cluster_alpha"]).item()),
            "cluster_connectivity": int(np.asarray(d["cluster_connectivity"]).item()) if "cluster_connectivity" in d else 1,
            "min_cluster_size": int(np.asarray(d["min_cluster_size"]).item()) if "min_cluster_size" in d else 1,
            "min_cluster_mass": float(np.asarray(d["min_cluster_mass"]).item()) if "min_cluster_mass" in d else 0.0,
            "merge_similar_clusters": bool(np.asarray(d["merge_similar_clusters"]).item()) if "merge_similar_clusters" in d else False,
            "merge_max_distance_px": float(np.asarray(d["merge_max_distance_px"]).item()) if "merge_max_distance_px" in d else 0.0,
            "merge_min_trace_corr": float(np.asarray(d["merge_min_trace_corr"]).item()) if "merge_min_trace_corr" in d else 1.0,
            "spatial_bin": int(np.asarray(d["spatial_bin"]).item()),
            # dF/F-like effect for traces; observed_effect_movie is the TESTED statistic (z in per_pixel_z mode)
            "trace_effect": np.asarray(d["observed_effect_raw_movie"] if "observed_effect_raw_movie" in d else d["observed_effect_movie"], dtype=float),
            "min_footprint_px": int(np.asarray(d["min_cluster_footprint_px"]).item()) if "min_cluster_footprint_px" in d else 0,
            "min_frames": int(np.asarray(d["min_cluster_frames"]).item()) if "min_cluster_frames" in d else 0,
            "significance_statistic": str(np.asarray(d["significance_statistic"]).item()) if "significance_statistic" in d else "effect",
            # opt-in per-sign rule (see docs/statistics.md); for such files "threshold" / "null_max" above are the
            # positive-sign values and the negative-sign ones are here
            "per_sign_null": bool(np.asarray(d["per_sign_null"]).item()) if "per_sign_null" in d else False,
            "threshold_neg": float(np.asarray(d["cluster_threshold_neg"]).item()) if "cluster_threshold_neg" in d else None,
            "null_max_pos": np.asarray(d["null_max_cluster_mass_pos"], dtype=float) if "null_max_cluster_mass_pos" in d else None,
            "null_max_neg": np.asarray(d["null_max_cluster_mass_neg"], dtype=float) if "null_max_cluster_mass_neg" in d else None,
            "t_rel": t_rel,
        }
