"""Database access of the preprocessing page, the preprocessing service and the preprocessing adapter. Plain arguments in, plain data out."""
from __future__ import annotations

from app.db.core import connect, rows, tx
from app.settings import Settings

ACTIVE_RUN_STATUSES = ("pending", "queued", "running")


def experiment_row(settings: Settings, experiment_id: int) -> dict | None:
    with connect(settings) as conn:
        row = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
    return dict(row) if row is not None else None


def experiment_source_path(settings: Settings, experiment_id: int) -> str | None:
    """source_path of the experiment, or None when there is no such experiment."""
    with connect(settings) as conn:
        row = conn.execute("SELECT source_path FROM experiments WHERE id=?", (experiment_id,)).fetchone()
    return row["source_path"] if row is not None else None


def preprocessing_page_data(settings: Settings, experiment_id: int) -> dict | None:
    """Everything the preprocessing page reads from the database (None: no such experiment): the experiment, its runs (newest first),
    the preprocessing steps of all its runs, and its notes."""
    with connect(settings) as conn:
        exp = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if exp is None:
            return None
        runs = rows(conn, "SELECT * FROM analysis_runs WHERE experiment_id=? ORDER BY created_at DESC, id ASC", (experiment_id,))
        steps = rows(
            conn,
            """
            SELECT ps.*
            FROM preprocessing_steps ps
            JOIN analysis_runs r ON r.id=ps.run_id
            WHERE r.experiment_id=?
            ORDER BY ps.run_id DESC, ps.step_order ASC
            """,
            (experiment_id,),
        )
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
    return {"experiment": dict(exp), "runs": runs, "steps": steps, "notes": notes}


def latest_preprocessing_run(settings: Settings, experiment_id: int) -> dict | None:
    """The newest non-test preprocessing run of the experiment (all columns), or None."""
    with connect(settings) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM analysis_runs
            WHERE experiment_id=? AND step='preprocessing' AND is_test=0
            ORDER BY id DESC LIMIT 1
            """,
            (experiment_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def run_is_active(settings: Settings, run_id: int) -> bool:
    marks = ",".join("?" for _ in ACTIVE_RUN_STATUSES)
    with connect(settings) as conn:
        row = conn.execute(f"SELECT 1 FROM analysis_runs WHERE id=? AND status IN ({marks})", (run_id, *ACTIVE_RUN_STATUSES)).fetchone()
    return row is not None


def step_row(settings: Settings, run_id: int, step_name: str) -> dict | None:
    with connect(settings) as conn:
        row = conn.execute("SELECT * FROM preprocessing_steps WHERE run_id=? AND step_name=?", (run_id, step_name)).fetchone()
    return dict(row) if row is not None else None


def run_with_experiment(settings: Settings, run_id: int) -> dict | None:
    """A run row plus source_path and canonical_id of its experiment, or None."""
    with connect(settings) as conn:
        row = conn.execute(
            """
            SELECT r.*,e.source_path,e.canonical_id
            FROM analysis_runs r
            JOIN experiments e ON e.id=r.experiment_id
            WHERE r.id=?
            """,
            (run_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def replace_preprocessing_steps(settings: Settings, run_id: int, step_names: list[str]) -> None:
    """Make the run's preprocessing_steps rows exactly these steps, in this order, all pending."""
    with tx(settings) as conn:
        conn.execute("DELETE FROM preprocessing_steps WHERE run_id=?", (run_id,))
        for index, step in enumerate(step_names):
            conn.execute(
                "INSERT INTO preprocessing_steps(run_id,step_name,step_order,status) VALUES(?,?,?,?)",
                (run_id, step, index, "pending"),
            )


def mark_step_queued(settings: Settings, run_id: int, step_name: str, job_id: str, log_path: str, submitted_at: str, message: str) -> None:
    with tx(settings) as conn:
        conn.execute(
            """
            UPDATE preprocessing_steps
            SET status='queued',slurm_job_id=?,slurm_log_path=?,submitted_at=?,message=?
            WHERE run_id=? AND step_name=?
            """,
            (job_id, log_path, submitted_at, message, run_id, step_name),
        )


def reset_for_rerun(settings: Settings, experiment_id: int, run_id: int, parameters_json: str, step_statuses: list[tuple[str, str]]) -> None:
    """Put the run back to 'queued' (new parameters, no end/exit/review note), the experiment back to 'preprocessing_running' /
    'unreviewed', and set each (step name, status) of step_statuses with its Slurm fields cleared."""
    with tx(settings) as conn:
        conn.execute(
            """
            UPDATE analysis_runs
            SET status='queued',
                ended_at=NULL,
                exit_code=NULL,
                review_note='',
                parameters_json=?
            WHERE id=?
            """,
            (parameters_json, run_id),
        )
        conn.execute(
            """
            UPDATE experiments
            SET workflow_stage='preprocessing_running',
                preprocessing_review_state='unreviewed',
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (experiment_id,),
        )
        for step_name, status in step_statuses:
            conn.execute(
                """
                UPDATE preprocessing_steps
                SET status=?,
                    slurm_job_id='',
                    slurm_log_path='',
                    submitted_at=NULL,
                    started_at=NULL,
                    ended_at=NULL,
                    exit_code=NULL,
                    message=''
                WHERE run_id=? AND step_name=?
                """,
                (status, run_id, step_name),
            )


def mark_rerun_failed(settings: Settings, run_id: int, step_name: str, message: str, ended_at: str) -> None:
    """The resubmission of a step failed: the step and the run are 'failed' with the message (cut to 1000 characters)."""
    with tx(settings) as conn:
        conn.execute(
            "UPDATE preprocessing_steps SET status='failed',message=?,ended_at=? WHERE run_id=? AND step_name=?",
            (message[:1000], ended_at, run_id, step_name),
        )
        conn.execute(
            "UPDATE analysis_runs SET status='failed',ended_at=?,exit_code=1,review_note=? WHERE id=?",
            (ended_at, message[:1000], run_id),
        )
