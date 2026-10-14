"""Database access for experiments: the registry (add, scan, remove, reset), the experiment pages (notes, favorites, runs, cells) and the cell look-ups.

Every function takes plain arguments (and the Settings that locate the database) and returns plain data (dicts, lists, numbers). The services,
the pages and the pipelines never open the database themselves.
"""
from __future__ import annotations

import json
import sqlite3

from app.db.core import connect, rows, tx
from app.settings import Settings

# workflow stages of an experiment (experiments.workflow_stage)
STAGE_PREPROCESSING_PENDING = "preprocessing_pending"
STAGE_AWAITING_SUITE2P = "awaiting_suite2p"
STAGE_ANALYSIS_READY = "analysis_ready"
# review states (experiments.preprocessing_review_state / suite2p_review_state)
REVIEW_UNREVIEWED = "unreviewed"
REVIEW_CONFIRMED = "confirmed"

# the run statuses a run can be cancelled/reset from (analysis_runs.status)
RUN_CANCELLABLE_STATUSES = ("pending", "queued", "running", "completed", "failed", "blocked")


def _in_list(values) -> str:
    return ",".join("?" for _ in values)


# ---- the experiment list and single experiments -----------------------------------------------------------------------------------------------
def experiment_list(settings: Settings, favorites_only: bool) -> tuple[list[dict], dict | None]:
    """(experiments with their cell count, newest scan-history row or None), newest recording first."""
    with connect(settings) as conn:
        experiments = rows(conn, """
            SELECT e.*, COALESCE(n.neuron_count, 0) AS neuron_count
            FROM experiments e
            LEFT JOIN (SELECT experiment_id, COUNT(*) AS neuron_count FROM neurons GROUP BY experiment_id) n ON n.experiment_id=e.id
            WHERE (? = 0 OR e.is_favorite = 1)
            ORDER BY substr(e.recording_date, 7, 4) DESC,
                     substr(e.recording_date, 4, 2) DESC,
                     substr(e.recording_date, 1, 2) DESC,
                     e.canonical_id ASC
        """, (1 if favorites_only else 0,))
        last = conn.execute("SELECT * FROM scan_history ORDER BY id DESC LIMIT 1").fetchone()
    return experiments, (dict(last) if last else None)


def experiment_row(settings: Settings, experiment_id: int) -> dict | None:
    """The experiments row of an experiment as a dict (None when it is unknown)."""
    with connect(settings) as conn:
        row = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
    return dict(row) if row is not None else None


def experiment_exists(settings: Settings, experiment_id: int) -> bool:
    with connect(settings) as conn:
        return conn.execute("SELECT 1 FROM experiments WHERE id=?", (experiment_id,)).fetchone() is not None


def experiment_has_runs(settings: Settings, experiment_id: int) -> bool:
    with connect(settings) as conn:
        return conn.execute("SELECT 1 FROM analysis_runs WHERE experiment_id=? LIMIT 1", (experiment_id,)).fetchone() is not None


def workflow_stage_of(settings: Settings, experiment_id: int) -> str | None:
    """The workflow stage of an experiment (None when it is unknown)."""
    with connect(settings) as conn:
        row = conn.execute("SELECT workflow_stage FROM experiments WHERE id=?", (experiment_id,)).fetchone()
    return row["workflow_stage"] if row is not None else None


def experiment_source_path(settings: Settings, experiment_id: int) -> str | None:
    """The experiment folder of an experiment (None when it is unknown)."""
    with connect(settings) as conn:
        row = conn.execute("SELECT source_path FROM experiments WHERE id=?", (experiment_id,)).fetchone()
    return row["source_path"] if row is not None else None


def set_favorite(settings: Settings, experiment_id: int, favorite: bool) -> bool:
    """Mark or unmark an experiment as favorite. False when the experiment is unknown."""
    with tx(settings) as conn:
        updated = conn.execute("UPDATE experiments SET is_favorite=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (1 if favorite else 0, experiment_id)).rowcount
    return bool(updated)


def save_validation(settings: Settings, experiment_id: int, state: str, report: dict) -> None:
    """Store a fresh validation report of an experiment."""
    with tx(settings) as conn:
        conn.execute("UPDATE experiments SET validation_state=?,validation_report=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (state, json.dumps(report), experiment_id))


# ---- cells and runs of an experiment ----------------------------------------------------------------------------------------------------------
def experiment_neurons(settings: Settings, experiment_id: int) -> list[dict]:
    """The cells of an experiment [{id, pipeline_neuron_id}], ordered by cell number."""
    with connect(settings) as conn:
        return rows(conn, "SELECT id,pipeline_neuron_id FROM neurons WHERE experiment_id=? ORDER BY CAST(pipeline_neuron_id AS INTEGER)", (experiment_id,))


def pipeline_neuron_ids(settings: Settings, experiment_id: int) -> list[str]:
    """The pipeline cell numbers (text) of an experiment, in database order."""
    with connect(settings) as conn:
        return [r["pipeline_neuron_id"] for r in conn.execute("SELECT pipeline_neuron_id FROM neurons WHERE experiment_id=?", (experiment_id,))]


def latest_completed_run(settings: Settings, experiment_id: int, step: str) -> dict | None:
    """The newest completed real (not test) run row of a step of an experiment, or None."""
    with connect(settings) as conn:
        row = conn.execute(
            """
            SELECT * FROM analysis_runs
            WHERE experiment_id=? AND step=? AND is_test=0 AND status='completed'
            ORDER BY id DESC LIMIT 1
            """,
            (experiment_id, step),
        ).fetchone()
    return dict(row) if row is not None else None


def experiment_page_data(settings: Settings, experiment_id: int, analysis_steps) -> dict | None:
    """Everything the analysis page reads from the database, or None when the experiment is unknown:
    {exp, neurons, runs (newest first), latest_real_runs {step: newest real run row of each of analysis_steps}, confirmed_count, notes}."""
    with connect(settings) as conn:
        exp = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if exp is None:
            return None
        neurons = rows(conn, "SELECT id,pipeline_neuron_id FROM neurons WHERE experiment_id=? ORDER BY CAST(pipeline_neuron_id AS INTEGER)", (experiment_id,))
        runs = rows(conn, "SELECT * FROM analysis_runs WHERE experiment_id=? ORDER BY created_at DESC, id ASC", (experiment_id,))
        latest = {}
        for step in analysis_steps:
            row = conn.execute(
                """
                SELECT *
                FROM analysis_runs
                WHERE experiment_id=? AND step=? AND is_test=0
                ORDER BY id DESC
                LIMIT 1
                """,
                (experiment_id, step),
            ).fetchone()
            if row is not None:
                latest[step] = dict(row)
        confirmed_count = conn.execute("SELECT COUNT(*) AS n FROM confirmed_neurons WHERE experiment_id=?", (experiment_id,)).fetchone()["n"]
        notes = rows(
            conn,
            """
            SELECT id,note_text,created_at,updated_at
            FROM experiment_notes
            WHERE experiment_id=?
            ORDER BY created_at DESC, id DESC
            """,
            (experiment_id,),
        )
    return {"exp": dict(exp), "neurons": neurons, "runs": runs, "latest_real_runs": latest, "confirmed_count": confirmed_count, "notes": notes}


# ---- notes ------------------------------------------------------------------------------------------------------------------------------------
def add_note(settings: Settings, experiment_id: int, note_text: str) -> bool:
    """Add a note to an experiment. False when the experiment is unknown."""
    with tx(settings) as conn:
        if conn.execute("SELECT 1 FROM experiments WHERE id=?", (experiment_id,)).fetchone() is None:
            return False
        conn.execute("INSERT INTO experiment_notes(experiment_id,note_text) VALUES(?,?)", (experiment_id, note_text))
    return True


def delete_note(settings: Settings, experiment_id: int, note_id: int) -> bool:
    """Delete a note of an experiment. False when there is no such note."""
    with tx(settings) as conn:
        deleted = conn.execute("DELETE FROM experiment_notes WHERE id=? AND experiment_id=?", (note_id, experiment_id)).rowcount
    return bool(deleted)


# ---- the registry: add and scan ---------------------------------------------------------------------------------------------------------------
def is_removed_from_registry(settings: Settings, source_path: str, canonical_id: str) -> bool:
    """True when the experiment was removed from the registry (and so must not be added again)."""
    with connect(settings) as conn:
        return conn.execute("SELECT 1 FROM removed_experiments WHERE source_path=? OR canonical_id=? LIMIT 1", (source_path, canonical_id)).fetchone() is not None


def _registered_id(conn, source_path: str, canonical_id: str) -> int | None:
    row = conn.execute("SELECT id FROM experiments WHERE source_path=? OR canonical_id=?", (source_path, canonical_id)).fetchone()
    return int(row["id"]) if row is not None else None


def registered_experiment_id(settings: Settings, source_path: str, canonical_id: str) -> int | None:
    """The id of the registered experiment with this folder or canonical id, or None."""
    with connect(settings) as conn:
        return _registered_id(conn, source_path, canonical_id)


def _insert_neurons(conn, experiment_id: int, neurons: list[dict]) -> None:
    for n in neurons:
        conn.execute(
            "INSERT INTO neurons(experiment_id,pipeline_neuron_id,roi_source,roi_summary) VALUES(?,?,?,?)",
            (experiment_id, n["pipeline_neuron_id"], n["roi_source"], json.dumps(n["roi_summary"])),
        )


def insert_experiment(settings: Settings, fields: dict, report: dict, workflow_stage: str, preprocessing_review_state: str, suite2p_review_state: str, neurons: list[dict]) -> int:
    """Register an experiment with its cells and return its id. `fields` are canonical_id, source_path, cage, mouse_name, recording_date, behavior, fov,
    z_plane, zoom. When another call registered the same experiment in the meantime, the id of that one is returned."""
    with tx(settings) as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO experiments(
                  canonical_id,source_path,cage,mouse_name,recording_date,behavior,fov,z_plane,zoom,
                  validation_state,validation_report,workflow_stage,preprocessing_review_state,suite2p_review_state
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    fields["canonical_id"], fields["source_path"], fields["cage"], fields["mouse_name"], fields["recording_date"], fields["behavior"],
                    fields["fov"], fields["z_plane"], fields["zoom"],
                    report["state"], json.dumps(report), workflow_stage, preprocessing_review_state, suite2p_review_state,
                ),
            )
        except sqlite3.IntegrityError:
            existing = _registered_id(conn, fields["source_path"], fields["canonical_id"])
            if existing is None:
                raise
            return existing
        experiment_id = int(cur.lastrowid)
        _insert_neurons(conn, experiment_id, neurons)
    return experiment_id


def start_scan(settings: Settings, parent_path: str, started_at: str) -> int:
    """Write the scan-history row of a scan that starts now; returns its id."""
    with tx(settings) as conn:
        return int(conn.execute("INSERT INTO scan_history(parent_path,started_at) VALUES(?,?)", (parent_path, started_at)).lastrowid)


def registry_snapshot(settings: Settings) -> tuple[set[str], set[str], set[str]]:
    """(folders of the registered experiments, folders of the removed ones, canonical ids of the removed ones)."""
    with connect(settings) as conn:
        existing = {r["source_path"] for r in conn.execute("SELECT source_path FROM experiments")}
        removed_paths = {r["source_path"] for r in conn.execute("SELECT source_path FROM removed_experiments")}
        removed_ids = {r["canonical_id"] for r in conn.execute("SELECT canonical_id FROM removed_experiments")}
    return existing, removed_paths, removed_ids


def finish_scan(settings: Settings, scan_id: int, completed_at: str, summary: dict) -> None:
    """Close the scan-history row of a scan and remember when the last scan finished."""
    with tx(settings) as conn:
        conn.execute("UPDATE scan_history SET completed_at=?,summary_json=? WHERE id=?", (completed_at, json.dumps(summary), scan_id))
        conn.execute("INSERT OR REPLACE INTO app_settings(key,value) VALUES('last_scan_time',?)", (completed_at,))


# ---- the registry: confirm, remove, reset -----------------------------------------------------------------------------------------------------
def confirm_preprocessing(settings: Settings, experiment_id: int) -> None:
    with tx(settings) as conn:
        conn.execute(
            """
            UPDATE experiments
            SET preprocessing_review_state=?,
                workflow_stage=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (REVIEW_CONFIRMED, STAGE_AWAITING_SUITE2P, experiment_id),
        )


def confirm_suite2p(settings: Settings, experiment_id: int, neurons: list[dict], report: dict, cancel_steps) -> None:
    """The Suite2p cells were confirmed: cancel the runs of `cancel_steps` (they used the old cells), replace the cells, mark the experiment analysis-ready."""
    with tx(settings) as conn:
        conn.execute(
            f"""
            UPDATE analysis_runs
            SET status='cancelled',
                ended_at=CURRENT_TIMESTAMP,
                review_note='Cancelled because Suite2p cells were refreshed.'
            WHERE experiment_id=?
              AND step IN ({_in_list(cancel_steps)})
              AND status IN ({_in_list(RUN_CANCELLABLE_STATUSES)})
            """,
            (experiment_id, *cancel_steps, *RUN_CANCELLABLE_STATUSES),
        )
        conn.execute("DELETE FROM neurons WHERE experiment_id=?", (experiment_id,))
        _insert_neurons(conn, experiment_id, neurons)
        conn.execute(
            """
            UPDATE experiments
            SET validation_state='valid',
                validation_report=?,
                suite2p_review_state=?,
                workflow_stage=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (json.dumps(report), REVIEW_CONFIRMED, STAGE_ANALYSIS_READY, experiment_id),
        )


def remove_experiment(settings: Settings, experiment_id: int, note: str) -> tuple[dict, list[dict]]:
    """Take an experiment out of the registry and block it from being added again. Returns (its experiments row, its run rows).
    Raises ValueError when the experiment is unknown."""
    with tx(settings) as conn:
        exp = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if exp is None:
            raise ValueError("Experiment not found.")
        run_rows = rows(conn, "SELECT * FROM analysis_runs WHERE experiment_id=?", (experiment_id,))
        conn.execute("INSERT OR REPLACE INTO removed_experiments(canonical_id,source_path,note) VALUES(?,?,?)", (exp["canonical_id"], exp["source_path"], note))
        conn.execute("DELETE FROM experiments WHERE id=?", (experiment_id,))
    return dict(exp), run_rows


def reset_experiment(settings: Settings, experiment_id: int, report: dict, workflow_stage: str, preprocessing_review_state: str) -> tuple[dict, list[dict]]:
    """Delete the runs, cells, confirmed cells and metadata overrides of an experiment and put it back to `workflow_stage` with a fresh validation report.
    Returns ({table: number of rows deleted}, the deleted run rows). Raises ValueError when the experiment is unknown."""
    with tx(settings) as conn:
        if conn.execute("SELECT 1 FROM experiments WHERE id=?", (experiment_id,)).fetchone() is None:
            raise ValueError("Experiment not found.")
        run_rows = rows(conn, "SELECT * FROM analysis_runs WHERE experiment_id=?", (experiment_id,))
        counts = {
            "runs": len(run_rows),
            "neurons": conn.execute("SELECT COUNT(*) AS n FROM neurons WHERE experiment_id=?", (experiment_id,)).fetchone()["n"],
            "confirmed_neurons": conn.execute("SELECT COUNT(*) AS n FROM confirmed_neurons WHERE experiment_id=?", (experiment_id,)).fetchone()["n"],
            "metadata_overrides": conn.execute("SELECT COUNT(*) AS n FROM metadata_overrides WHERE experiment_id=?", (experiment_id,)).fetchone()["n"],
        }
        conn.execute("DELETE FROM confirmed_neurons WHERE experiment_id=?", (experiment_id,))
        conn.execute("DELETE FROM metadata_overrides WHERE experiment_id=?", (experiment_id,))
        conn.execute("DELETE FROM analysis_runs WHERE experiment_id=?", (experiment_id,))
        conn.execute("DELETE FROM neurons WHERE experiment_id=?", (experiment_id,))
        conn.execute(
            """
            UPDATE experiments
            SET validation_state=?,
                validation_report=?,
                workflow_stage=?,
                preprocessing_review_state=?,
                suite2p_review_state=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (report["state"], json.dumps(report), workflow_stage, preprocessing_review_state, REVIEW_UNREVIEWED, experiment_id),
        )
    return counts, run_rows


# ---- cell look-ups ----------------------------------------------------------------------------------------------------------------------------
def neuron_with_experiment(settings: Settings, neuron_id: int) -> dict | None:
    """One cell with its experiment's folder: {id, pipeline_neuron_id, experiment_id, source_path, canonical_id}; None when the cell is unknown."""
    with connect(settings) as conn:
        row = conn.execute(
            """
            SELECT n.id, n.pipeline_neuron_id, e.id AS experiment_id, e.source_path, e.canonical_id
            FROM neurons n JOIN experiments e ON e.id=n.experiment_id
            WHERE n.id=?
            """,
            (neuron_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def neuron_ids_by_cell(settings: Settings, experiment_id: int) -> dict[int, int]:
    """{pipeline cell number: database id of the cell} of an experiment."""
    with connect(settings) as conn:
        return {int(r["pipeline_neuron_id"]): int(r["id"]) for r in conn.execute("SELECT id, pipeline_neuron_id FROM neurons WHERE experiment_id=?", (int(experiment_id),))}
