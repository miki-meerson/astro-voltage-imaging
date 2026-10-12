"""Database access for confirmed cells (the confirmed_* tables). Plain arguments in, plain data out."""
from __future__ import annotations

from collections.abc import Callable

from app.db.core import connect, tx
from app.settings import Settings

# columns of one confirmed_sta_results row / confirmed_sta_clusters row, in insert order (the dicts passed to save_confirmed_cell use these names)
STA_RESULT_COLUMNS = (
    "result_type", "version", "n_peaks", "n_significant_clusters", "has_significant_cluster", "pre_window", "post_window",
    "analysis_window_json", "baseline_window_json", "spatial_bin", "voxel_alpha", "cluster_alpha", "n_permutations", "parameters_json", "source_npz_path",
)
STA_CLUSTER_COLUMNS = (
    "cluster_id", "sign", "is_significant", "p_value", "mass", "size_voxels", "peak_value", "centroid_y", "centroid_x", "area_px",
    "mean_trace_json", "t_rel_json", "mask_artifact_path",
)
PEAKS_COLUMNS = (
    "all_peak_frames_json", "simple_peak_frames_json", "complex_peak_frames_json", "n_all_peaks", "n_simple_peaks", "n_complex_peaks",
    "peak_detection_params_json", "source_artifact_path",
)


def confirmation_inputs(settings: Settings, neuron_id: int) -> dict | None:
    """What confirming a cell needs from the database: the cell with its experiment's metadata and the id of the latest completed spike /
    behavior / STA run ({'neuron': {...}, 'spike_run': row or None, 'behavior_run': ..., 'sta_run': ...}); None when the cell is unknown."""
    with connect(settings) as conn:
        neuron = conn.execute(
            """
            SELECT n.id,n.experiment_id,n.pipeline_neuron_id,e.canonical_id,e.cage,e.mouse_name,e.recording_date,e.behavior,e.fov,e.z_plane,e.zoom,e.source_path
            FROM neurons n
            JOIN experiments e ON e.id=n.experiment_id
            WHERE n.id=?
            """,
            (neuron_id,),
        ).fetchone()
        if neuron is None:
            return None
        runs = {}
        for step in ("spike", "behavior", "sta"):
            row = conn.execute(
                """
                SELECT * FROM analysis_runs
                WHERE experiment_id=? AND step=? AND is_test=0 AND status='completed'
                ORDER BY id DESC LIMIT 1
                """,
                (int(neuron["experiment_id"]), step),
            ).fetchone()
            runs[f"{step}_run"] = dict(row) if row is not None else None
    return {"neuron": dict(neuron), **runs}


def save_confirmed_cell(settings: Settings, neuron: dict, spike_run_id: int | None, behavior_run_id: int | None, sta_run_id: int,
                        reviewer_note: str, peaks: dict | None, sta_results: list[dict]) -> int:
    """Store (or replace) the confirmation of a cell in one transaction and return its confirmed_neurons id.

    neuron: the 'neuron' dict of confirmation_inputs. peaks: PEAKS_COLUMNS -> value, or None when the experiment has no spike run.
    sta_results: one dict per STA result type with STA_RESULT_COLUMNS and a "clusters" list of dicts with STA_CLUSTER_COLUMNS.
    """
    with tx(settings) as conn:
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
                int(neuron["experiment_id"]), int(neuron["id"]), str(neuron["pipeline_neuron_id"]), neuron["canonical_id"], neuron["cage"],
                neuron["mouse_name"], neuron["recording_date"], neuron["behavior"], neuron["fov"], neuron["z_plane"], neuron["zoom"],
                neuron["source_path"], spike_run_id, behavior_run_id, sta_run_id, reviewer_note,
            ),
        )
        confirmed_id = int(conn.execute(
            "SELECT id FROM confirmed_neurons WHERE experiment_id=? AND pipeline_neuron_id=?",
            (int(neuron["experiment_id"]), str(neuron["pipeline_neuron_id"])),
        ).fetchone()["id"])

        conn.execute("DELETE FROM confirmed_neuron_peaks WHERE confirmed_neuron_id=?", (confirmed_id,))
        conn.execute("DELETE FROM confirmed_cell_metrics WHERE confirmed_neuron_id=?", (confirmed_id,))
        for old in conn.execute("SELECT id FROM confirmed_sta_results WHERE confirmed_neuron_id=?", (confirmed_id,)).fetchall():
            conn.execute("DELETE FROM confirmed_sta_results WHERE id=?", (old["id"],))

        if peaks is not None:
            conn.execute(
                f"INSERT INTO confirmed_neuron_peaks(confirmed_neuron_id,{','.join(PEAKS_COLUMNS)}) VALUES(?{',?' * len(PEAKS_COLUMNS)})",
                (confirmed_id, *(peaks[name] for name in PEAKS_COLUMNS)),
            )
        for result in sta_results:
            cur = conn.execute(
                f"INSERT INTO confirmed_sta_results(confirmed_neuron_id,{','.join(STA_RESULT_COLUMNS)}) VALUES(?{',?' * len(STA_RESULT_COLUMNS)})",
                (confirmed_id, *(result[name] for name in STA_RESULT_COLUMNS)),
            )
            result_id = int(cur.lastrowid)
            for cluster in result["clusters"]:
                conn.execute(
                    f"INSERT INTO confirmed_sta_clusters(confirmed_sta_result_id,{','.join(STA_CLUSTER_COLUMNS)}) VALUES(?{',?' * len(STA_CLUSTER_COLUMNS)})",
                    (result_id, *(cluster[name] for name in STA_CLUSTER_COLUMNS)),
                )
        return confirmed_id


def delete_confirmed_cell(settings: Settings, neuron_id: int, remove_files: Callable[[str, str], None]) -> bool:
    """Remove the confirmation of a cell. `remove_files(canonical_id, pipeline_neuron_id)` runs inside the same transaction, after the rows are
    deleted, so an error while removing the saved files undoes the deletion. False when the cell has no confirmation."""
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
        remove_files(str(row["canonical_id"]), str(row["pipeline_neuron_id"]))
        return True
