from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3
import threading
import time

from app.cluster import client
from app.db import run_repository
from app.services.preprocessing import load_state, save_state, submit_preprocessing_step
from app.services.runs import output_manifest
from app.settings import Settings
from utils.paths_utils import PathKind, path_for


FAILED_STATES = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")
RUNNING_STATES = ("RUNNING", "COMPLETING", "CONFIGURING", "RESIZING", "STAGE_OUT")
QUEUED_STATES = ("PENDING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "SUSPENDED")
CLUSTER_ERRORS = client.CLUSTER_ERRORS
MONITOR_TIMEOUT_SECONDS = 8


def _cluster(settings: Settings) -> dict:
    return settings.raw.get("cluster", {})


def _monitor_timeout(settings: Settings) -> int:
    return int(_cluster(settings).get("monitor_command_timeout_seconds") or MONITOR_TIMEOUT_SECONDS)


def _job_id(output_dir: str | None) -> str | None:
    if not output_dir:
        return None
    path = path_for(PathKind.CLUSTER_JOB_ID, output_dir)
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
    states, note = client.job_state(_cluster(settings), job_id, _monitor_timeout(settings))
    if not states:
        return None, None, note
    return _status_from_slurm_states(states)


def _cluster_log_failure(output_dir: str | None, stderr_path: str | None = None) -> tuple[str | None, int | None, str]:
    paths: list[Path] = []
    if output_dir:
        out = Path(output_dir)
        paths.extend([path_for(PathKind.CLUSTER_STDERR_LOG, out), path_for(PathKind.CLUSTER_STDOUT_LOG, out)])
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
    rows = run_repository.preprocessing_step_rows(settings, run_id)
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
    text = client.tail_remote_file(_cluster(settings), remote_log_path, timeout=_monitor_timeout(settings))
    if text:
        local_path.write_text(text, encoding="utf-8", errors="replace")


def poll_slurm_once(settings: Settings) -> int:
    runs = run_repository.active_cluster_runs(settings, "sta")
    changed = 0
    for run in runs:
        job_id = _job_id(run.get("output_dir"))
        if not job_id:
            continue
        status, exit_code, note_text = _cluster_log_failure(run.get("output_dir"), run.get("stderr_path"))
        if not status:
            try:
                status, exit_code, note_text = _slurm_status(settings, job_id)
            except CLUSTER_ERRORS as exc:
                print(f"[slurm monitor warning] run {run['id']} status lookup failed: {type(exc).__name__}: {exc}")
                continue
        if status in {"queued", "running"}:
            if status != run["status"]:
                run_repository.set_run_status(settings, run["id"], status)
                changed += 1
            continue
        manifest = output_manifest(Path(run["output_dir"]))
        note = ""
        if status == "failed":
            stderr_path = path_for(PathKind.CLUSTER_STDERR_LOG, run["output_dir"]) if run.get("output_dir") else Path(run.get("stderr_path") or "")
            note = _tail(stderr_path) or note_text
        if status:
            run_repository.finish_cluster_run(settings, run["id"], status, datetime.now().isoformat(timespec="seconds"), exit_code, manifest, note[-1000:])
            changed += 1
    changed += poll_preprocessing_once(settings)
    return changed


def poll_preprocessing_once(settings: Settings) -> int:
    runs = run_repository.active_cluster_runs(settings, "preprocessing")
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
        local_log = path_for(PathKind.PREPROCESSING_STEP_LOG, out, step=step["name"])
        _copy_remote_log(settings, step.get("log_path", ""), local_log)
        status, exit_code, note = _local_log_status(Path(step.get("local_log_path") or local_log))
        if not status:
            status, exit_code, slurm_note = _slurm_status(settings, job_id)
            note = slurm_note or note
        if status in {"queued", "running"}:
            step["status"] = status
            save_state(out, state)
            run_repository.mark_preprocessing_step_active(
                settings, run["id"], step["name"], status, datetime.now().isoformat(timespec="seconds") if status == "running" else None
            )
            changed += 1
            continue
        if status == "failed":
            step["status"] = "failed"
            step["ended_at"] = datetime.now().isoformat(timespec="seconds")
            step["message"] = note
            save_state(out, state)
            manifest = output_manifest(out)
            run_repository.fail_preprocessing_step(settings, run["id"], run["experiment_id"], step["name"], step["ended_at"], note[-1000:], manifest)
            changed += 1
            continue
        if status == "completed":
            step["status"] = "completed"
            step["ended_at"] = datetime.now().isoformat(timespec="seconds")
            save_state(out, state)
            run_repository.complete_preprocessing_step(settings, run["id"], step["name"], step["ended_at"])
            next_index = index + 1
            if next_index < len(steps):
                next_step = steps[next_index]
                if not run_repository.claim_preprocessing_step(settings, int(run["id"]), str(next_step["name"])):
                    changed += 1
                    continue
                next_step["status"] = "submitting"
                save_state(out, state)
                try:
                    submit_preprocessing_step(settings, out, state, next_index)
                    run_repository.set_run_status(settings, run["id"], "queued")
                except CLUSTER_ERRORS as exc:
                    run_repository.fail_run_with_note(settings, run["id"], datetime.now().isoformat(timespec="seconds"), str(exc)[:1000])
                changed += 1
                continue
            manifest = output_manifest(out)
            finished = datetime.now().isoformat(timespec="seconds")
            run_repository.complete_preprocessing_run(settings, run["id"], run["experiment_id"], finished, manifest)
            changed += 1
    return changed


def start_slurm_monitor(settings: Settings) -> threading.Thread | None:
    if not settings.raw.get("cluster", {}).get("enabled", False):
        return None
    interval = max(15, int(settings.raw.get("cluster", {}).get("poll_interval_seconds", 30)))

    def loop() -> None:
        while True:
            try:
                poll_slurm_once(settings)
            except (sqlite3.Error, *CLUSTER_ERRORS) as exc:
                print(f"[slurm monitor warning] {type(exc).__name__}: {exc}")
            time.sleep(interval)

    thread = threading.Thread(target=loop, daemon=True, name="astro-slurm-monitor")
    thread.start()
    return thread
