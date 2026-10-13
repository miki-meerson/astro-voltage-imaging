"""Analysis runs: create, launch and execute one step of the pipeline for an experiment (locally, or by submitting a cluster job)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from app.db import run_repository
from app.settings import Settings
from utils.paths_utils import PathKind, path_for

STEPS = ("preprocessing", "spike", "behavior", "anatomy", "sta")
ADAPTER_MODULES = {
    "preprocessing": "preprocessing_adapter",
    "spike": "spike_adapter",
    "behavior": "behavior_adapter",
    "anatomy": "anatomy_adapter",
    "sta": "sta_adapter",
}
CLUSTER_ENV_VARS = {
    "ssh_host": "ASTRO_CLUSTER_HOST",
    "ssh_port": "ASTRO_CLUSTER_PORT",
    "ssh_username": "ASTRO_CLUSTER_USER",
    "ssh_password": "ASTRO_CLUSTER_PASSWORD",
    "ssh_key_filename": "ASTRO_CLUSTER_KEY",
}
STEPS_THAT_FEED_STA = {"spike", "behavior", "anatomy"}
_EXECUTOR_LOCK = threading.Lock()
_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_SIZE = 0


def pipeline_status(latest: dict[str, str], steps: tuple[str, ...] | None = None) -> dict:
    """The status of an experiment from the status of the newest real run of each step ({step: status}).

    'completed' only when EVERY step of the pipeline (STEPS, the one list of steps) has a newest run that completed; a failed rerun after an
    earlier success counts as failed. Otherwise the first of running / pending (pending or queued) / failed (steps in alphabetical order) wins;
    'partial' = some steps completed and none is active or failed; 'none' = no step completed."""
    steps = STEPS if steps is None else steps
    latest = {s: st for s, st in latest.items() if s in steps}  # rows of steps that are no longer in the pipeline are ignored
    done = [s for s in steps if latest.get(s) == "completed"]
    status, step = "none", ""
    for name, wanted in (("running", ("running",)), ("pending", ("pending", "queued")), ("failed", ("failed",))):
        hits = sorted(s for s, st in latest.items() if st in wanted)
        if hits:
            status, step = name, hits[0]
            break
    else:
        if len(done) == len(steps):
            status = "completed"
        elif done:
            status = "partial"
    return {"job_status": status, "job_step": step, "completed_steps": len(done), "total_steps": len(steps)}


def _safe_name(value: str) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")[:120] or "experiment"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def create_run(settings: Settings, experiment_id: int, step: str, params: dict, backend: str = "local", neuron_id: int | None = None, is_test: bool = False) -> int:
    if step not in STEPS:
        raise ValueError(f"Unknown step: {step}")
    return run_repository.insert_run(settings, experiment_id, step, params, backend, neuron_id, is_test)


def _run_folder_fields(row) -> dict:
    """The name fields of a run's folder (file-name safe); a run without a cell is 'all'."""
    cell_id = row["pipeline_neuron_id"] if row["pipeline_neuron_id"] is not None else "all"
    return {"canonical_id": _safe_name(row["canonical_id"]), "cell": _safe_name(cell_id), "step": _safe_name(row["step"]), "run_id": row["id"]}


def run_dir(settings: Settings, run_id: int, is_test: bool = False) -> Path:
    root = settings.test_output_root if is_test else settings.output_root
    row = run_repository.run_folder_row(settings, run_id)
    if row is None:
        raise ValueError(f"Run {run_id} is not in the database.")
    formatted = path_for(PathKind.RUN_DIR, root, **_run_folder_fields(row))
    candidates = []
    if row["output_dir"]:
        candidates.append(Path(row["output_dir"]))
    if row["stdout_path"]:
        candidates.append(Path(row["stdout_path"]).parent)
    candidates.append(formatted)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if row["output_dir"]:
        return Path(row["output_dir"])
    return formatted


def output_manifest(out: Path) -> list[dict]:
    """Every file under a run folder as {path (relative to the outputs root), type, size}; [] when the folder is missing."""
    if not out.exists():
        return []
    return [{"path": str(p.relative_to(out.parent)), "type": p.suffix.lower().lstrip(".") or "file", "size": p.stat().st_size} for p in out.rglob("*") if p.is_file()]


def _allocate_run_dir(settings: Settings, run) -> Path:
    root = settings.test_output_root if run["is_test"] else settings.output_root
    if run["output_dir"]:
        return Path(run["output_dir"])
    fields = _run_folder_fields(run)
    candidate = path_for(PathKind.RUN_DIR, root, **fields)
    attempt = 2
    while candidate.exists():
        candidate = path_for(PathKind.RUN_DIR_ATTEMPT, root, **fields, attempt=attempt)
        attempt += 1
    return candidate


def resume_pending_runs(settings: Settings) -> int:
    run_ids = run_repository.pending_run_ids(settings)
    for run_id in run_ids:
        launch_run(settings, run_id)
    return len(run_ids)


def rerun_failed_steps(settings: Settings, step: str | None = None) -> list[int]:
    if step is not None and step not in STEPS:
        raise ValueError(f"Unknown step: {step}")
    created: list[int] = []
    for run in run_repository.latest_failed_runs(settings, step):
        try:
            run_params = json.loads(run["parameters_json"] or "{}")
        except json.JSONDecodeError:
            run_params = {}
        backend = "cluster" if run["step"] == "sta" else (run["backend"] or "local")
        run_id = create_run(settings, int(run["experiment_id"]), str(run["step"]), run_params, backend=backend, neuron_id=run["neuron_id"], is_test=False)
        launch_run(settings, run_id)
        created.append(run_id)
    return created


def launch_run(settings: Settings, run_id: int) -> None:
    executor = _executor_for(settings)
    executor.submit(_execute_run, settings, run_id)


def _executor_for(settings: Settings) -> ThreadPoolExecutor:
    global _EXECUTOR, _EXECUTOR_SIZE
    local_workers = int(settings.raw.get("local", {}).get("max_workers", 1))
    cluster_workers = int(settings.raw.get("cluster", {}).get("max_workers", 1))
    size = max(1, local_workers + cluster_workers)
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None or _EXECUTOR_SIZE != size:
            _EXECUTOR = ThreadPoolExecutor(max_workers=size, thread_name_prefix="astro-run")
            _EXECUTOR_SIZE = size
        return _EXECUTOR


# ---- one run -------------------------------------------------------------------------------------------------------------------------------

def _execute_run(settings: Settings, run_id: int) -> None:
    """Run one step: build its command, run it, record the outcome. A cluster run only submits the job here; the cluster monitor finishes it."""
    run = run_repository.run_for_execution(settings, run_id)
    if run["backend"] == "cluster" and not settings.raw.get("cluster", {}).get("enabled", False):
        run_repository.update_run(settings, run_id, {"status": "failed", "ended_at": _now(), "command_json": json.dumps({"error": "Cluster backend disabled."})})
        return
    out = _allocate_run_dir(settings, run)
    out.mkdir(parents=True, exist_ok=False)
    stdout, stderr = path_for(PathKind.RUN_STDOUT_LOG, out), path_for(PathKind.RUN_STDERR_LOG, out)
    where = {"stdout_path": str(stdout), "stderr_path": str(stderr), "log_path": str(stdout), "output_dir": str(out)}
    warning = _missing_input_warning(settings, run)
    if warning:
        stderr.write_text(f"{warning}\n", encoding="utf-8")
        now = _now()
        run_repository.update_run(settings, run_id, {"status": "failed", "started_at": now, "ended_at": now, "exit_code": 1, **where, "command_json": json.dumps({"error": warning})})
        return
    cmd = _step_command(settings, run, out)
    run_repository.update_run(settings, run_id, {"status": "running", "started_at": _now(), **where, "command_json": json.dumps(cmd)})
    code = _run_process(settings, cmd, stdout, stderr)
    _record_outcome(settings, run, out, cmd, code, where)
    if code == 0 and run["step"] in STEPS_THAT_FEED_STA and not run["is_test"]:
        _launch_sta_if_ready(settings, int(run["experiment_id"]))


def _missing_input_warning(settings: Settings, run) -> str | None:
    """Text saying what the step needs and does not have yet, or None when it can start."""
    if run["step"] == "sta" and run["backend"] == "cluster" and _latest_completed_output(settings, int(run["experiment_id"]), "spike") is None:
        return "STA requires a completed spike run for the same experiment."
    return None


def _adapter_command(run, out: Path, params_json: str, extra_args: tuple[str, ...] = ()) -> list[str]:
    adapter = ADAPTER_MODULES[run["step"]]
    return [sys.executable, "-u", "-m", f"app.pipelines.{adapter}", "--experiment", run["source_path"], "--output", str(out), "--params-json", params_json, *extra_args]


def _preprocessing_command(settings: Settings, run, out: Path) -> list[str]:
    return _adapter_command(run, out, run["parameters_json"], ("--run-id", str(run["id"])))


def _sta_command(settings: Settings, run, out: Path) -> list[str]:
    """A cluster STA run also gets the runs it reads from and the cluster settings; a local one is a plain adapter call."""
    if run["backend"] != "cluster":
        return _adapter_command(run, out, run["parameters_json"])
    experiment_id = int(run["experiment_id"])
    spike_run = _latest_completed_output(settings, experiment_id, "spike")
    params =json.loads(run["parameters_json"] or "{}")
    params["spike_run"] = str(spike_run)
    for step, key in (("behavior", "behavior_run"), ("anatomy", "anatomy_run")):
        found = _latest_completed_output(settings, experiment_id, step)
        if found is not None:
            params[key] = str(found)
    cluster = settings.raw.get("cluster", {})
    params["cluster_submit_command"] = cluster.get("submit_command", "sbatch")
    params["cluster_script"] = cluster.get("sta_script", path_for(PathKind.STA_JOB_SCRIPT, "").as_posix())
    params["cluster_submission_method"] = cluster.get("submission_method", "paramiko")
    params["cluster_paramiko_timeout_seconds"] = cluster.get("paramiko_timeout_seconds", 600)
    params["cluster_paramiko_attempts"] = cluster.get("paramiko_attempts", 5)
    params["cluster_path_map"] = cluster.get("path_map", {})
    return _adapter_command(run, out, json.dumps(params))


# step -> function(settings, run, out) giving the command line; a step that is not listed is a plain adapter call
_COMMAND_BUILDERS = {"preprocessing": _preprocessing_command, "sta": _sta_command}


def _step_command(settings: Settings, run, out: Path) -> list[str]:
    builder = _COMMAND_BUILDERS.get(run["step"])
    return builder(settings, run, out) if builder else _adapter_command(run, out, run["parameters_json"])


def _process_env(settings: Settings) -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(settings.project_root) + os.pathsep + env.get("PYTHONPATH", "")
    cluster = settings.raw.get("cluster", {})
    for key, env_key in CLUSTER_ENV_VARS.items():
        value = cluster.get(key)
        if value not in (None, ""):
            env[env_key] = str(value)
    return env


def _run_process(settings: Settings, cmd: list[str], stdout: Path, stderr: Path) -> int:
    """Run the command with its output in the two log files; exit code 1 when it could not be started (the reason goes to the stderr log)."""
    try:
        with stdout.open("w", encoding="utf-8") as so, stderr.open("w", encoding="utf-8") as se:
            return subprocess.run(cmd, cwd=settings.project_root, env=_process_env(settings), text=True, stdout=so, stderr=se).returncode
    except (OSError, ValueError) as exc:  # not startable (missing interpreter, bad arguments)
        stderr.write_text(str(exc), encoding="utf-8")
        return 1


def _record_outcome(settings: Settings, run, out: Path, cmd: list[str], code: int, where: dict) -> None:
    """A cluster run whose adapter submitted a job stays 'queued' (the cluster monitor completes it); everything else ends now."""
    submitted = run["backend"] == "cluster" and code == 0 and path_for(PathKind.CLUSTER_JOB_ID, out).exists()
    manifest = output_manifest(out)
    fields = {
        "status": "queued" if submitted else ("completed" if code == 0 else "failed"),
        "ended_at": None if submitted else _now(),
        "exit_code": None if submitted else code,
        "stdout_path": where["stdout_path"], "stderr_path": where["stderr_path"], "log_path": where["log_path"],
        "command_json": json.dumps(cmd), "output_manifest_json": json.dumps(manifest),
    }
    run_repository.finish_run(settings, int(run["id"]), fields, manifest, int(run["experiment_id"]), mark_preprocessing_running=submitted and run["step"] == "preprocessing")


def _launch_sta_if_ready(settings: Settings, experiment_id: int) -> None:
    params = _sta_params(settings)
    required = {"spike", "behavior"}
    if "anatomy" in params.get("result_types", []):
        required.add("anatomy")
    sta_run_id = run_repository.create_pending_run_if_ready(settings, experiment_id, "sta", "cluster", params, required)
    if sta_run_id is not None:
        launch_run(settings, sta_run_id)


def _latest_completed_output(settings: Settings, experiment_id: int, step: str) -> Path | None:
    row = run_repository.latest_completed_run_row(settings, experiment_id, step)
    if row is None:
        return None
    return run_dir(settings, int(row["id"]), bool(row["is_test"]))


def _sta_params(settings: Settings) -> dict:
    params = dict(settings.raw.get("defaults", {}).get("sta", {}))
    params.setdefault("anatomy_spatial_bin", 1)
    return params
