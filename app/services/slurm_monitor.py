from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import threading
import time

from app.db.core import connect, tx
from app.services.cluster_ssh import run_ssh_command
from app.services.preprocessing import load_state, save_state, submit_preprocessing_step
from app.services.runs import output_manifest, replace_artifacts
from app.settings import Settings


FAILED_STATES = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")
RUNNING_STATES = ("RUNNING", "COMPLETING", "CONFIGURING", "RESIZING", "STAGE_OUT")
QUEUED_STATES = ("PENDING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "SUSPENDED")


def _ssh_command(settings: Settings, command: str) -> tuple[str, str, int]:
    cluster = dict(settings.raw.get("cluster", {}))
    timeout = int(cluster.get("monitor_command_timeout_seconds") or 8)
    return run_ssh_command(cluster, command, command_timeout_seconds=timeout)


def _job_id(output_dir: str | None) -> str | None:
    if not output_dir:
        return None
    path = Path(output_dir) / "cluster_job_id.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text or None


def _tail(path: Path, limit: int = 2000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-limit:].strip()


def _local_log_status(path: Path) -> tuple[str | None, int | None, str]:
    text = _tail(path, 12000)
    if not text:
        return None, None, ""
    success_markers = (
        "Everything worked well. The Script finished to run.",
        "Done. Summary saved to ",
    )
    if any(marker in text for marker in success_markers):
        return "completed", 0, "completed from local Slurm log"
    failure_markers = ("Traceback (most recent call last)", "slurmstepd: error", "OUT_OF_MEMORY", "Killed")
    if any(marker in text for marker in failure_markers):
        return "failed", 1, text[-1000:]
    return "running", None, "running from local Slurm log"


def _status_from_slurm_states(states: list[str]) -> tuple[str | None, int | None, str]:
    normalized = [state.split()[0].split("+")[0].strip().upper() for state in states if state.strip()]
    note = ", ".join(states)
    if any(state.startswith(FAILED_STATES) for state in normalized):
        return "failed", 1, note
    if any(state.startswith("COMPLETED") for state in normalized):
        return "completed", 0, note
    if any(state in RUNNING_STATES for state in normalized):
        return "running", None, note
    if any(state in QUEUED_STATES for state in normalized):
        return "queued", None, note
    return None, None, note


def _slurm_status(settings: Settings, job_id: str) -> tuple[str | None, int | None, str]:
    cluster = settings.raw.get("cluster", {})
    squeue = str(cluster.get("squeue_command") or "/usr/bin/squeue")
    sacct = str(cluster.get("sacct_command") or "/usr/bin/sacct")
    sq_out, _, sq_code = _ssh_command(settings, f"{squeue} -h -j {job_id} -o '%T'")
    if sq_code == 0 and sq_out.strip():
        states = [line.strip() for line in sq_out.splitlines() if line.strip()]
        return _status_from_slurm_states(states)
    acct_out, acct_err, acct_code = _ssh_command(settings, f"{sacct} -n -j {job_id} --format=State,ExitCode -P")
    if acct_code != 0:
        return None, None, (acct_err or acct_out).strip()
    states = [line.split("|", 1)[0].strip() for line in acct_out.splitlines() if line.strip()]
    if not states:
        return None, None, ""
    return _status_from_slurm_states(states)


def _cluster_log_failure(output_dir: str | None, stderr_path: str | None = None) -> tuple[str | None, int | None, str]:
    paths: list[Path] = []
    if output_dir:
        out = Path(output_dir)
        paths.extend([out / "cluster_stderr.log", out / "cluster_stdout.log"])
    if stderr_path:
        paths.append(Path(stderr_path))
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        status, exit_code, note = _local_log_status(path)
        if status in {"completed", "failed"}:
            return status, exit_code, note
    return None, None, ""


def _sync_preprocessing_state(settings: Settings, run_id: int, out: Path, state: dict) -> dict:
    """Keep the optional JSON status file aligned with SQLite."""
    with connect(settings) as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT step_name,status,slurm_job_id,slurm_log_path,submitted_at,ended_at,message
                FROM preprocessing_steps
                WHERE run_id=?
                ORDER BY step_order
                """,
                (run_id,),
            )
        ]
    if not rows:
        return state
    labels = {step.get("name"): step.get("label", step.get("name")) for step in state.get("steps", [])}
    state["steps"] = [
        {
            "name": row["step_name"],
            "label": labels.get(row["step_name"], row["step_name"].replace("_", " ").title()),
            "status": row["status"],
            "job_id": row["slurm_job_id"] or "",
            "log_path": row["slurm_log_path"] or "",
            "local_log_path": row["slurm_log_path"] or "",
            "submitted_at": row["submitted_at"] or "",
            "ended_at": row["ended_at"] or "",
            "message": row["message"] or "",
        }
        for row in rows
    ]
    state["current_index"] = next(
        (index for index, row in enumerate(rows) if row["status"] != "completed"),
        len(rows),
    )
    save_state(out, state)
    return state


def _copy_remote_log(settings: Settings, remote_log_path: str, local_path: Path) -> None:
    if not remote_log_path:
        return
    source = Path(remote_log_path)
    if source.exists():
        text = _tail(source, 12000)
        if text:
            local_path.write_text(text, encoding="utf-8", errors="replace")
        return
    cluster_log = remote_log_path.replace("\\", "/")
    out, err, code = _ssh_command(settings, f"tail -n 4000 {cluster_log!r}")
    text = out if code == 0 else (err or out)
    if text:
        local_path.write_text(text, encoding="utf-8", errors="replace")


def poll_slurm_once(settings: Settings) -> int:
    with connect(settings) as conn:
        runs = [dict(row) for row in conn.execute(
            """
            SELECT id,status,output_dir,stderr_path
            FROM analysis_runs
            WHERE step='sta' AND backend='cluster' AND status IN ('queued','running')
            ORDER BY id
            """
        )]
    changed = 0
    for run in runs:
        job_id = _job_id(run.get("output_dir"))
        if not job_id:
            continue
        status, exit_code, note_text = _cluster_log_failure(run.get("output_dir"), run.get("stderr_path"))
        if not status:
            try:
                status, exit_code, note_text = _slurm_status(settings, job_id)
            except Exception as exc:
                print(f"[slurm monitor warning] run {run['id']} status lookup failed: {type(exc).__name__}: {exc}")
                continue
        if status in {"queued", "running"}:
            if status != run["status"]:
                with tx(settings) as conn:
                    conn.execute("UPDATE analysis_runs SET status=? WHERE id=?", (status, run["id"]))
                changed += 1
            continue
        manifest = output_manifest(Path(run["output_dir"]))
        note = ""
        if status == "failed":
            stderr_path = Path(run["output_dir"]) / "cluster_stderr.log" if run.get("output_dir") else Path(run.get("stderr_path") or "")
            note = _tail(stderr_path) or note_text
        if status:
            with tx(settings) as conn:
                conn.execute(
                    """
                    UPDATE analysis_runs
                    SET status=?,ended_at=?,exit_code=?,output_manifest_json=?,review_note=?
                    WHERE id=?
                    """,
                    (
                        status,
                        datetime.now().isoformat(timespec="seconds"),
                        exit_code,
                        json.dumps(manifest),
                        note[-1000:],
                        run["id"],
                    ),
                )
                replace_artifacts(conn, run["id"], manifest)
            changed += 1
    changed += poll_preprocessing_once(settings)
    return changed


def poll_preprocessing_once(settings: Settings) -> int:
    with connect(settings) as conn:
        runs = [
            dict(row)
            for row in conn.execute(
                """
                SELECT id,status,output_dir,experiment_id
                FROM analysis_runs
                WHERE step='preprocessing' AND backend='cluster' AND status IN ('queued','running')
                ORDER BY id
                """
            )
        ]
    changed = 0
    for run in runs:
        out = Path(run["output_dir"] or "")
        state = load_state(out)
        if not state:
            continue
        state = _sync_preprocessing_state(settings, int(run["id"]), out, state)
        index = int(state.get("current_index", 0))
        steps = state.get("steps", [])
        if index >= len(steps):
            continue
        step = steps[index]
        job_id = str(step.get("job_id") or "")
        if not job_id:
            continue
        local_log = out / f"preprocessing_{step['name']}.log"
        _copy_remote_log(settings, step.get("log_path", ""), local_log)
        status, exit_code, note = _local_log_status(Path(step.get("local_log_path") or local_log))
        if not status:
            status, exit_code, slurm_note = _slurm_status(settings, job_id)
            note = slurm_note or note
        if status in {"queued", "running"}:
            step["status"] = status
            save_state(out, state)
            with tx(settings) as conn:
                conn.execute("UPDATE analysis_runs SET status=? WHERE id=?", (status, run["id"]))
                conn.execute(
                    "UPDATE preprocessing_steps SET status=?,started_at=COALESCE(started_at,?) WHERE run_id=? AND step_name=?",
                    (
                        status,
                        datetime.now().isoformat(timespec="seconds") if status == "running" else None,
                        run["id"],
                        step["name"],
                    ),
                )
            changed += 1
            continue
        if status == "failed":
            step["status"] = "failed"
            step["ended_at"] = datetime.now().isoformat(timespec="seconds")
            step["message"] = note
            save_state(out, state)
            manifest = output_manifest(out)
            with tx(settings) as conn:
                conn.execute(
                    """
                    UPDATE preprocessing_steps SET status='failed',ended_at=?,exit_code=1,message=?
                    WHERE run_id=? AND step_name=?
                    """,
                    (step["ended_at"], note[-1000:], run["id"], step["name"]),
                )
                conn.execute(
                    """
                    UPDATE analysis_runs SET status='failed',ended_at=?,exit_code=1,output_manifest_json=?,review_note=?
                    WHERE id=?
                    """,
                    (step["ended_at"], json.dumps(manifest), note[-1000:], run["id"]),
                )
                conn.execute(
                    """
                    UPDATE experiments
                    SET workflow_stage='preprocessing_failed',
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (run["experiment_id"],),
                )
            changed += 1
            continue
        if status == "completed":
            step["status"] = "completed"
            step["ended_at"] = datetime.now().isoformat(timespec="seconds")
            save_state(out, state)
            with tx(settings) as conn:
                conn.execute(
                    """
                    UPDATE preprocessing_steps SET status='completed',ended_at=?,exit_code=0
                    WHERE run_id=? AND step_name=?
                    """,
                    (step["ended_at"], run["id"], step["name"]),
                )
            next_index = index + 1
            if next_index < len(steps):
                next_step = steps[next_index]
                if not _claim_preprocessing_step(settings, int(run["id"]), str(next_step["name"])):
                    changed += 1
                    continue
                next_step["status"] = "submitting"
                save_state(out, state)
                try:
                    submit_preprocessing_step(settings, out, state, next_index)
                    with tx(settings) as conn:
                        conn.execute("UPDATE analysis_runs SET status='queued' WHERE id=?", (run["id"],))
                except Exception as exc:
                    with tx(settings) as conn:
                        conn.execute(
                            "UPDATE analysis_runs SET status='failed',ended_at=?,exit_code=1,review_note=? WHERE id=?",
                            (datetime.now().isoformat(timespec="seconds"), str(exc)[:1000], run["id"]),
                        )
                changed += 1
                continue
            manifest = output_manifest(out)
            finished = datetime.now().isoformat(timespec="seconds")
            with tx(settings) as conn:
                conn.execute(
                    """
                    UPDATE analysis_runs
                    SET status='completed',ended_at=?,exit_code=0,output_manifest_json=?
                    WHERE id=?
                    """,
                    (finished, json.dumps(manifest), run["id"]),
                )
                replace_artifacts(conn, run["id"], manifest)
                conn.execute(
                    """
                    UPDATE experiments
                    SET workflow_stage='preprocessing_review',
                        preprocessing_review_state='unreviewed',
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (run["experiment_id"],),
                )
            changed += 1
    return changed


def _claim_preprocessing_step(settings: Settings, run_id: int, step_name: str) -> bool:
    with tx(settings) as conn:
        updated = conn.execute(
            """
            UPDATE preprocessing_steps
            SET status='submitting'
            WHERE run_id=?
              AND step_name=?
              AND status='pending'
              AND COALESCE(slurm_job_id, '')=''
            """,
            (run_id, step_name),
        ).rowcount
    return updated == 1


def start_slurm_monitor(settings: Settings) -> threading.Thread | None:
    if not settings.raw.get("cluster", {}).get("enabled", False):
        return None
    interval = max(15, int(settings.raw.get("cluster", {}).get("poll_interval_seconds", 30)))

    def loop() -> None:
        while True:
            try:
                poll_slurm_once(settings)
            except Exception as exc:
                print(f"[slurm monitor warning] {type(exc).__name__}: {exc}")
            time.sleep(interval)

    thread = threading.Thread(target=loop, daemon=True, name="astro-slurm-monitor")
    thread.start()
    return thread
