from __future__ import annotations

from pathlib import Path
import json
import shutil

import numpy as np

from app.db.core import tx
from app.services.run_views import latest_completed_run
from app.services.runs import run_dir
from app.settings import Settings


def _as_json_array(value) -> str:
    arr = np.asarray(value)
    return json.dumps(arr.ravel().tolist())


def _jsonable(value):
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _jsonable(value.item())
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _row_json_array(row, name: str) -> str:
    value = getattr(row, name, None)
    if value is None:
        return "[]"
    try:
        if bool(np.isscalar(value)) and not np.isfinite(value):
            return "[]"
    except TypeError:
        pass
    return json.dumps(_jsonable(value))


def _row_int(row, name: str):
    value = getattr(row, name, None)
    arr = np.asarray(value)
    if arr.shape != ():
        return None
    value = arr.item()
    try:
        if value is None or not np.isfinite(value):
            return None
    except TypeError:
        if value is None:
            return None
    return int(value)


def _row_float(row, name: str):
    value = getattr(row, name, None)
    arr = np.asarray(value)
    if arr.shape != ():
        return None
    value = arr.item()
    try:
        if value is None or not np.isfinite(value):
            return None
    except TypeError:
        if value is None:
            return None
    return float(value)


def _load_npz_arrays(path: Path) -> dict:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as d:
        return {key: np.asarray(d[key]) for key in d.files}


def _copy_source_artifact(src: Path, dest_root: Path) -> str:
    if not src.exists():
        return ""
    dest_root.mkdir(parents=True, exist_ok=True)
    dst = dest_root / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return str(dst)


def _remove_confirmed_artifacts(settings: Settings, canonical_id: str, pipeline_neuron_id: str) -> None:
    base = (settings.project_root / "runtime" / "scientific_database").resolve()
    target = (base / canonical_id / f"cell_{pipeline_neuron_id}").resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return
    if target.exists():
        shutil.rmtree(target)


def unconfirm_neuron(settings: Settings, neuron_id: int) -> bool:
    with tx(settings) as conn:
        row = conn.execute(
            """
            SELECT cn.id,cn.canonical_id,cn.pipeline_neuron_id
            FROM neurons n
            JOIN confirmed_neurons cn
              ON cn.experiment_id=n.experiment_id
             AND cn.pipeline_neuron_id=n.pipeline_neuron_id
            WHERE n.id=?
            """,
            (neuron_id,),
        ).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM confirmed_neurons WHERE id=?", (int(row["id"]),))
        _remove_confirmed_artifacts(settings, str(row["canonical_id"]), str(row["pipeline_neuron_id"]))
        return True


def confirm_neuron(settings: Settings, neuron_id: int, reviewer_note: str = "") -> int:
    from utils.clusters_utils import cluster_mask2d_from_cluster, extract_clusters, saved_cluster_args
    from utils.db_utils import collect_cluster_result_metadata

    with tx(settings) as conn:
        n = conn.execute(
            """
            SELECT n.*,e.*
            FROM neurons n
            JOIN experiments e ON e.id=n.experiment_id
            WHERE n.id=?
            """,
            (neuron_id,),
        ).fetchone()
        if n is None:
            raise ValueError("Neuron not found.")

        spike_run = latest_completed_run(conn, int(n["experiment_id"]), "spike")
        behavior_run = latest_completed_run(conn, int(n["experiment_id"]), "behavior")
        sta_run = latest_completed_run(conn, int(n["experiment_id"]), "sta")
        if sta_run is None:
            raise ValueError("Confirming a neuron requires a completed STA Maps run.")

        conn.execute(
            """
            INSERT INTO confirmed_neurons(
              experiment_id,neuron_row_id,pipeline_neuron_id,canonical_id,cage,mouse_name,
              recording_date,behavior,fov,z_plane,zoom,source_path,spike_run_id,behavior_run_id,
              sta_run_id,reviewer_note,updated_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(experiment_id,pipeline_neuron_id) DO UPDATE SET
              neuron_row_id=excluded.neuron_row_id,
              spike_run_id=excluded.spike_run_id,
              behavior_run_id=excluded.behavior_run_id,
              sta_run_id=excluded.sta_run_id,
              reviewer_note=excluded.reviewer_note,
              updated_at=CURRENT_TIMESTAMP
            """,
            (
                int(n["experiment_id"]),
                int(n["id"]),
                str(n["pipeline_neuron_id"]),
                n["canonical_id"],
                n["cage"],
                n["mouse_name"],
                n["recording_date"],
                n["behavior"],
                n["fov"],
                n["z_plane"],
                n["zoom"],
                n["source_path"],
                int(spike_run["id"]) if spike_run else None,
                int(behavior_run["id"]) if behavior_run else None,
                int(sta_run["id"]),
                reviewer_note,
            ),
        )
        confirmed_id = int(conn.execute(
            "SELECT id FROM confirmed_neurons WHERE experiment_id=? AND pipeline_neuron_id=?",
            (int(n["experiment_id"]), str(n["pipeline_neuron_id"])),
        ).fetchone()["id"])

        conn.execute("DELETE FROM confirmed_neuron_peaks WHERE confirmed_neuron_id=?", (confirmed_id,))
        conn.execute("DELETE FROM confirmed_cell_metrics WHERE confirmed_neuron_id=?", (confirmed_id,))
        for old in conn.execute("SELECT id FROM confirmed_sta_results WHERE confirmed_neuron_id=?", (confirmed_id,)):
            conn.execute("DELETE FROM confirmed_sta_results WHERE id=?", (old["id"],))

        permanent_root = (
            settings.project_root
            / "runtime"
            / "scientific_database"
            / str(n["canonical_id"])
            / f"cell_{n['pipeline_neuron_id']}"
        )

        if spike_run is not None:
            spike_out = run_dir(settings, int(spike_run["id"]), bool(spike_run["is_test"]))
            cell_dir = spike_out / f"cell_{n['pipeline_neuron_id']}"
            labels_path = cell_dir / "simple_complex_spike_labels.npz"
            peaks_path = cell_dir / "simple_complex_peaks_for_analysis.npz"
            params_path = cell_dir / "peak_detection_params.npz"
            labels = _load_npz_arrays(labels_path)
            peaks = _load_npz_arrays(peaks_path)
            params = _load_npz_arrays(params_path)

            all_peaks = labels.get("peaks_all", peaks.get("peaks_all", peaks.get("all_peak_frames", [])))
            simple_peaks = labels.get("simple_peaks", labels.get("simple_peak_frames", labels.get("peaks_simple", peaks.get("simple_peak_frames", []))))
            complex_peaks = labels.get("complex_peaks", labels.get("complex_peak_frames", labels.get("peaks_complex", peaks.get("complex_peak_frames", []))))
            source_artifact = _copy_source_artifact(labels_path if labels_path.exists() else peaks_path, permanent_root / "spikes")
            conn.execute(
                """
                INSERT INTO confirmed_neuron_peaks(
                  confirmed_neuron_id,all_peak_frames_json,simple_peak_frames_json,complex_peak_frames_json,
                  n_all_peaks,n_simple_peaks,n_complex_peaks,peak_detection_params_json,source_artifact_path
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    confirmed_id,
                    _as_json_array(all_peaks),
                    _as_json_array(simple_peaks),
                    _as_json_array(complex_peaks),
                    int(np.asarray(all_peaks).size),
                    int(np.asarray(simple_peaks).size),
                    int(np.asarray(complex_peaks).size),
                    json.dumps({k: _jsonable(v) for k, v in params.items()}),
                    source_artifact,
                ),
            )

        sta_out = run_dir(settings, int(sta_run["id"]), bool(sta_run["is_test"]))
        sta_dir = sta_out / "sta_algorithm_home" / "sta-analysis"
        result_df = collect_cluster_result_metadata(sta_dir, [int(n["pipeline_neuron_id"])])
        if result_df.empty:
            raise ValueError("No STA result files were found for this neuron.")
        result_df = result_df.sort_values("version").groupby("result_type", as_index=False).tail(1)

        for row in result_df.itertuples():
            source_npz = Path(row.path)
            copied_npz = _copy_source_artifact(source_npz, permanent_root / "sta")
            with np.load(source_npz, allow_pickle=True) as d:
                effect = np.asarray(d["observed_effect_movie"], dtype=np.float32)
                # stored traces are dF/F-like: use the raw effect, not the tested z-statistic
                trace_effect = np.asarray(d["observed_effect_raw_movie"], dtype=np.float32) if "observed_effect_raw_movie" in d else effect
                aw = np.asarray(d["analysis_window"], dtype=int)
                params = {
                    key: _jsonable(np.asarray(d[key]).item() if np.asarray(d[key]).shape == () else np.asarray(d[key]))
                    for key in d.files
                    if key
                    in {
                        "pre_window",
                        "post_window",
                        "analysis_window",
                        "baseline_window",
                        "voxel_alpha",
                        "cluster_alpha",
                        "spatial_bin",
                        "anatomy_spatial_bin",
                        "anatomy_radius_binned",
                        "anatomy_source_radius_binned",
                        "anatomy_source_spatial_bin",
                        "anatomy_fullres_radius_px",
                        "cluster_connectivity",
                        "min_cluster_size",
                        "min_cluster_mass",
                        "min_spikes_for_cluster",
                        "merge_similar_clusters",
                        "merge_max_distance_px",
                        "merge_min_trace_corr",
                        "null_standardization",
                        "per_sign_null",
                        "cluster_threshold_pos",
                        "cluster_threshold_neg",
                        "smooth_sigma_px",
                        "smooth_sigma_fullres_px",
                        "min_cluster_footprint_px",
                        "min_cluster_frames",
                        "significance_statistic",
                        "n_permutations",
                        "rng_seed",
                        "pre_seconds",
                        "post_seconds",
                        "baseline_seconds",
                        "analysis_half_window_seconds",
                        "analysis_half_window_frames",
                        "result_types",
                        "movie_shape",
                        "n_valid_centers",
                    }
                }
                clusters = extract_clusters(effect_movie=effect, **saved_cluster_args(d))  # honours per_sign_null when the file says so
            t_rel = np.arange(int(aw[0]), int(aw[1]) + 1)
            cur = conn.execute(
                """
                INSERT INTO confirmed_sta_results(
                  confirmed_neuron_id,result_type,version,n_peaks,n_significant_clusters,has_significant_cluster,
                  pre_window,post_window,analysis_window_json,baseline_window_json,spatial_bin,voxel_alpha,
                  cluster_alpha,n_permutations,parameters_json,source_npz_path
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    confirmed_id,
                    str(row.result_type),
                    int(row.version),
                    _row_int(row, "n_peaks"),
                    int(row.n_significant_clusters),
                    int(bool(row.has_significant_cluster)),
                    _row_int(row, "pre_window"),
                    _row_int(row, "post_window"),
                    _row_json_array(row, "analysis_window"),
                    _row_json_array(row, "baseline_window"),
                    _row_int(row, "spatial_bin"),
                    _row_float(row, "voxel_alpha"),
                    _row_float(row, "cluster_alpha"),
                    _row_int(row, "n_permutations"),
                    json.dumps(params),
                    copied_npz,
                ),
            )
            sta_result_id = int(cur.lastrowid)
            masks = {}
            for cluster in clusters:
                mask = cluster_mask2d_from_cluster(cluster)
                if mask is None:
                    continue
                cluster_id = int(cluster.get("cluster_id", len(masks) + 1))
                masks[f"cluster_{cluster_id}_mask2d"] = np.asarray(mask, dtype=bool)
                trace = np.asarray(trace_effect[:, mask].mean(axis=1), dtype=np.float32)
                yyxx = np.argwhere(mask)
                centroid_y = float(np.mean(yyxx[:, 0])) if yyxx.size else None
                centroid_x = float(np.mean(yyxx[:, 1])) if yyxx.size else None
                conn.execute(
                    """
                    INSERT INTO confirmed_sta_clusters(
                      confirmed_sta_result_id,cluster_id,sign,is_significant,p_value,mass,size_voxels,
                      peak_value,centroid_y,centroid_x,area_px,mean_trace_json,t_rel_json,mask_artifact_path
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        sta_result_id,
                        cluster_id,
                        str(cluster.get("sign", "")),
                        int(bool(cluster.get("is_significant", False))),
                        float(cluster["p_value"]) if "p_value" in cluster else None,
                        float(cluster.get("mass", 0.0)),
                        int(cluster.get("size", 0)),
                        float(cluster.get("peak_value", 0.0)),
                        centroid_y,
                        centroid_x,
                        int(mask.sum()),
                        _as_json_array(trace),
                        _as_json_array(t_rel),
                        str(permanent_root / "sta" / f"{source_npz.stem}_cluster_masks.npz"),
                    ),
                )
            if masks:
                (permanent_root / "sta").mkdir(parents=True, exist_ok=True)
                np.savez_compressed(permanent_root / "sta" / f"{source_npz.stem}_cluster_masks.npz", **masks)

        return confirmed_id
