"""Database access for analysis runs (the analysis_runs and artifacts tables). The only place the run code (app/services/runs.py) touches SQL."""
from __future__ import annotations

import json

from app.db.core import connect, tx
from app.settings import Settings

# columns update_run may set (the names are put into the SQL text, so only these are accepted)
RUN_COLUMNS = frozenset({
    "status", "started_at", "ended_at", "exit_code", "stdout_path", "stderr_path", "log_path", "output_dir", "command_json", "output_manifest_json",
    "review_note",
})


def next_run_id(conn) -> int:
    conn.execute("INSERT OR IGNORE INTO run_id_sequence(id,next_id) VALUES(1,1)")
    conn.execute(
        """
        UPDATE run_id_sequence
        SET next_id = MAX(next_id, (SELECT COALESCE(MAX(id), 0) + 1 FROM analysis_runs))
        WHERE id=1
        """
    )
    row = conn.execute(
        """
        UPDATE run_id_sequence
        SET next_id = next_id + 1
        WHERE id=1
        RETURNING next_id - 1 AS run_id
        """
    ).fetchone()
    return int(row["run_id"])


def insert_run(settings: Settings, experiment_id: int, step: str, params: dict, backend: str, neuron_id: int | None, is_test: bool) -> int:
    with tx(settings) as conn:
        run_id = next_run_id(conn)
        conn.execute(
            "INSERT INTO analysis_runs(id,experiment_id,neuron_id,step,backend,status,is_test,parameters_json) VALUES(?,?,?,?,?,?,?,?)",
            (run_id, experiment_id, neuron_id, step, backend, "pending", int(is_test), json.dumps(params)),
        )
        return run_id


def run_folder_row(settings: Settings, run_id: int):
    """What run_dir needs to find a run's folder (None when the run is unknown or the database cannot be read)."""
    with tx(settings) as conn:
        return conn.execute(
            """
            SELECT r.id,r.step,r.neuron_id,r.output_dir,r.stdout_path,e.canonical_id,n.pipeline_neuron_id
            FROM analysis_runs r
            JOIN experiments e ON e.id=r.experiment_id
            LEFT JOIN neurons n ON n.id=r.neuron_id
            WHERE r.id=?
            """,
            (run_id,),
        ).fetchone()


def run_for_execution(settings: Settings, run_id: int):
    """The run row with its experiment's source folder and the cell's pipeline number."""
    with tx(settings) as conn:
        return conn.execute(
            """
            SELECT r.*,e.source_path,e.canonical_id,n.pipeline_neuron_id
            FROM analysis_runs r
            JOIN experiments e ON e.id=r.experiment_id
            LEFT JOIN neurons n ON n.id=r.neuron_id
            WHERE r.id=?
            """,
            (run_id,),
        ).fetchone()


def update_run(settings: Settings, run_id: int, fields: dict) -> None:
    unknown = set(fields) - RUN_COLUMNS
    if unknown:
        raise ValueError(f"Unknown analysis_runs columns: {sorted(unknown)}")
    assignments = ",".join(f"{name}=?" for name in fields)
    with tx(settings) as conn:
        conn.execute(f"UPDATE analysis_runs SET {assignments} WHERE id=?", (*fields.values(), run_id))


def _replace_artifacts(conn, run_id: int, manifest: list[dict]) -> None:
    """Make the artifacts table of a run match its manifest."""
    conn.execute("DELETE FROM artifacts WHERE run_id=?", (run_id,))
    for item in manifest:
        conn.execute("INSERT INTO artifacts(run_id,artifact_type,relative_path,size_bytes) VALUES(?,?,?,?)", (run_id, item["type"], item["path"], item["size"]))


def finish_run(settings: Settings, run_id: int, fields: dict, manifest: list[dict], experiment_id: int, mark_preprocessing_running: bool) -> None:
    """Final state of a run: its columns, its artifacts and, for a submitted preprocessing job, the experiment's stage - in one transaction."""
    unknown = set(fields) - RUN_COLUMNS
    if unknown:
        raise ValueError(f"Unknown analysis_runs columns: {sorted(unknown)}")
    assignments = ",".join(f"{name}=?" for name in fields)
    with tx(settings) as conn:
        conn.execute(f"UPDATE analysis_runs SET {assignments} WHERE id=?", (*fields.values(), run_id))
        _replace_artifacts(conn, run_id, manifest)
        if mark_preprocessing_running:
            conn.execute("UPDATE experiments SET workflow_stage='preprocessing_running',updated_at=CURRENT_TIMESTAMP WHERE id=?", (experiment_id,))


def pending_run_ids(settings: Settings) -> list[int]:
    with tx(settings) as conn:
        return [int(row["id"]) for row in conn.execute("SELECT id FROM analysis_runs WHERE status='pending' ORDER BY id")]


def latest_failed_runs(settings: Settings, step: str | None) -> list[dict]:
    """The newest run of each experiment and step (real runs only) when it failed and no run of that step is pending, queued or running."""
    params: list[object] = []
    step_filter = ""
    if step is not None:
        step_filter = "AND r.step=?"
        params.append(step)
    with tx(settings) as conn:
        return [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT r.*
                FROM analysis_runs r
                JOIN (
                    SELECT experiment_id,step,MAX(id) AS run_id
                    FROM analysis_runs
                    WHERE is_test=0
                    GROUP BY experiment_id,step
                ) latest ON latest.run_id=r.id
                WHERE r.status='failed'
                  AND NOT EXISTS (
                    SELECT 1
                    FROM analysis_runs active
                    WHERE active.experiment_id=r.experiment_id
                      AND active.step=r.step
                      AND active.is_test=0
                      AND active.status IN ('pending','queued','running')
                )
                {step_filter}
                ORDER BY r.experiment_id,r.step
                """,
                params,
            )
        ]


def latest_completed_run_row(settings: Settings, experiment_id: int, step: str):
    """The newest completed real run of a step: id, is_test, output_dir, stdout_path (None when there is none)."""
    with tx(settings) as conn:
        return conn.execute(
            """
            SELECT id,is_test,output_dir,stdout_path
            FROM analysis_runs
            WHERE experiment_id=? AND step=? AND is_test=0 AND status='completed'
            ORDER BY id DESC LIMIT 1
            """,
            (experiment_id, step),
        ).fetchone()


def create_pending_run_if_ready(settings: Settings, experiment_id: int, step: str, backend: str, params: dict, required_steps: set[str]) -> int | None:
    """Create a pending real run of `step` when all `required_steps` have a completed real run and no run of `step` exists yet. Returns its id or None."""
    with tx(settings) as conn:
        completed = {r["step"] for r in conn.execute("SELECT step FROM analysis_runs WHERE experiment_id=? AND is_test=0 AND status='completed'", (experiment_id,))}
        if not required_steps.issubset(completed):
            return None
        existing = conn.execute(
            "SELECT id FROM analysis_runs WHERE experiment_id=? AND is_test=0 AND step=? AND status IN ('pending','queued','running','completed') LIMIT 1",
            (experiment_id, step),
        ).fetchone()
        if existing is not None:
            return None
        run_id = next_run_id(conn)
        conn.execute(
            "INSERT INTO analysis_runs(id,experiment_id,step,backend,status,is_test,parameters_json) VALUES(?,?,?,?,?,0,?)",
            (run_id, experiment_id, step, backend, "pending", json.dumps(params)),
        )
        return run_id


def latest_step_statuses(settings: Settings) -> dict[int, dict[str, str]]:
    """{experiment id: {step: status of that step's newest real (not test) run}}. The rule that turns this into an experiment status is runs.pipeline_status."""
    result: dict[int, dict[str, str]] = {}
    with tx(settings) as conn:
        for r in conn.execute(
            """
            SELECT r.experiment_id, r.step, r.status
            FROM analysis_runs r
            JOIN (SELECT experiment_id,step,MAX(id) AS run_id FROM analysis_runs WHERE is_test=0 GROUP BY experiment_id,step) latest ON latest.run_id=r.id
            """
        ):
            result.setdefault(int(r["experiment_id"]), {})[r["step"]] = r["status"]
    return result


# ---- what the Slurm monitor (app/services/slurm_monitor.py) reads and writes -------------------------------------------------------------

def active_cluster_runs(settings: Settings, step: str) -> list[dict]:
    """The cluster runs of a step that are queued or running: id, status, output_dir, stderr_path, experiment_id (oldest first)."""
    with connect(settings) as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT id,status,output_dir,stderr_path,experiment_id FROM analysis_runs WHERE step=? AND backend='cluster' AND status IN ('queued','running') ORDER BY id",
                (step,),
            )
        ]


def set_run_status(settings: Settings, run_id: int, status: str) -> None:
    with tx(settings) as conn:
        conn.execute("UPDATE analysis_runs SET status=? WHERE id=?", (status, run_id))


def finish_cluster_run(settings: Settings, run_id: int, status: str, ended_at: str, exit_code: int | None, manifest: list[dict], note: str) -> None:
    """A cluster run that the monitor saw end: status, end time, exit code, note, and its artifacts."""
    with tx(settings) as conn:
        conn.execute(
            "UPDATE analysis_runs SET status=?,ended_at=?,exit_code=?,output_manifest_json=?,review_note=? WHERE id=?",
            (status, ended_at, exit_code, json.dumps(manifest), note, run_id),
        )
        _replace_artifacts(conn, run_id, manifest)


def fail_run_with_note(settings: Settings, run_id: int, ended_at: str, note: str) -> None:
    with tx(settings) as conn:
        conn.execute("UPDATE analysis_runs SET status='failed',ended_at=?,exit_code=1,review_note=? WHERE id=?", (ended_at, note, run_id))


def preprocessing_step_rows(settings: Settings, run_id: int) -> list[dict]:
    """The steps of a preprocessing run in order: step_name, status, slurm_job_id, slurm_log_path, submitted_at, ended_at, message."""
    with connect(settings) as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT step_name,status,slurm_job_id,slurm_log_path,submitted_at,ended_at,message FROM preprocessing_steps WHERE run_id=? ORDER BY step_order",
                (run_id,),
            )
        ]


def mark_preprocessing_step_active(settings: Settings, run_id: int, step_name: str, status: str, started_at: str | None) -> None:
    """The step the monitor saw queued or running: the step and its run get that status; the step's start time is kept once set."""
    with tx(settings) as conn:
        conn.execute("UPDATE analysis_runs SET status=? WHERE id=?", (status, run_id))
        conn.execute(
            "UPDATE preprocessing_steps SET status=?,started_at=COALESCE(started_at,?) WHERE run_id=? AND step_name=?",
            (status, started_at, run_id, step_name),
        )


def fail_preprocessing_step(settings: Settings, run_id: int, experiment_id: int, step_name: str, ended_at: str, note: str, manifest: list[dict]) -> None:
    """A preprocessing step failed: the step, the run and the experiment's stage."""
    with tx(settings) as conn:
        conn.execute("UPDATE preprocessing_steps SET status='failed',ended_at=?,exit_code=1,message=? WHERE run_id=? AND step_name=?", (ended_at, note, run_id, step_name))
        conn.execute(
            "UPDATE analysis_runs SET status='failed',ended_at=?,exit_code=1,output_manifest_json=?,review_note=? WHERE id=?",
            (ended_at, json.dumps(manifest), note, run_id),
        )
        conn.execute("UPDATE experiments SET workflow_stage='preprocessing_failed',updated_at=CURRENT_TIMESTAMP WHERE id=?", (experiment_id,))


def complete_preprocessing_step(settings: Settings, run_id: int, step_name: str, ended_at: str) -> None:
    with tx(settings) as conn:
        conn.execute("UPDATE preprocessing_steps SET status='completed',ended_at=?,exit_code=0 WHERE run_id=? AND step_name=?", (ended_at, run_id, step_name))


def claim_preprocessing_step(settings: Settings, run_id: int, step_name: str) -> bool:
    """Mark a pending step that has no Slurm job yet as 'submitting'; True when this call got it (so only one submitter proceeds)."""
    with tx(settings) as conn:
        updated = conn.execute(
            "UPDATE preprocessing_steps SET status='submitting' WHERE run_id=? AND step_name=? AND status='pending' AND COALESCE(slurm_job_id, '')=''",
            (run_id, step_name),
        ).rowcount
    return updated == 1


def complete_preprocessing_run(settings: Settings, run_id: int, experiment_id: int, ended_at: str, manifest: list[dict]) -> None:
    """The last preprocessing step finished: the run is completed and the experiment waits for the preprocessing review."""
    with tx(settings) as conn:
        conn.execute("UPDATE analysis_runs SET status='completed',ended_at=?,exit_code=0,output_manifest_json=? WHERE id=?", (ended_at, json.dumps(manifest), run_id))
        _replace_artifacts(conn, run_id, manifest)
        conn.execute(
            "UPDATE experiments SET workflow_stage='preprocessing_review',preprocessing_review_state='unreviewed',updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (experiment_id,),
        )


# ---- run pages ----------------------------------------------------------------------------------------------------------------------------

def run_with_experiment(settings: Settings, run_id: int) -> dict | None:
    """One run with its experiment's canonical_id and source_path (None when unknown)."""
    with connect(settings) as conn:
        row = conn.execute("SELECT r.*,e.canonical_id,e.source_path FROM analysis_runs r JOIN experiments e ON e.id=r.experiment_id WHERE r.id=?", (run_id,)).fetchone()
    return dict(row) if row is not None else None


def run_row(settings: Settings, run_id: int) -> dict | None:
    with connect(settings) as conn:
        row = conn.execute("SELECT * FROM analysis_runs WHERE id=?", (run_id,)).fetchone()
    return dict(row) if row is not None else None
