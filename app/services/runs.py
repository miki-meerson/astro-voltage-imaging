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
from app.settings import PROJECT_ROOT, Settings
from utils import globals_utils as g
from utils.hypotheses import Hypothesis, catalogue_file, hypothesis_name, is_sta_step, load_hypotheses, sta_step
from utils.paths_utils import PathKind, path_for

BASE_STEPS = (g.PREPROCESSING_STEP, g.SPIKE_STEP, g.BEHAVIOR_STEP, g.ANATOMY_STEP, g.DECOMPOSITION_STEP)
BASE_ADAPTER_MODULES = {
    g.PREPROCESSING_STEP: "preprocessing_adapter",
    g.SPIKE_STEP: "spike_adapter",
    g.BEHAVIOR_STEP: "behavior_adapter",
    g.ANATOMY_STEP: "anatomy_adapter",
    g.DECOMPOSITION_STEP: "decomposition_adapter",
}
STA_ADAPTER_MODULE = "sta_adapter"  # the adapter of every hypothesis step


def build_steps(hypotheses: tuple[Hypothesis, ...]) -> tuple[str, ...]:
    """The steps of the pipeline: the base steps, then one STA step (sta:<name>) for each hypothesis, in catalogue order."""
    return (*BASE_STEPS, *(sta_step(h.name) for h in hypotheses))


def build_adapter_modules(steps: tuple[str, ...]) -> dict[str, str]:
    """{step: adapter module name} for the given steps."""
    return {step: STA_ADAPTER_MODULE if is_sta_step(step) else BASE_ADAPTER_MODULES[step] for step in steps}


HYPOTHESES = load_hypotheses(catalogue_file(PROJECT_ROOT))  # the catalogue is read once, when the app starts
STEPS = build_steps(HYPOTHESES)
ADAPTER_MODULES = build_adapter_modules(STEPS)
CLUSTER_ENV_VARS = {
    "ssh_host": "ASTRO_CLUSTER_HOST",
    "ssh_port": "ASTRO_CLUSTER_PORT",
    "ssh_username": "ASTRO_CLUSTER_USER",
    "ssh_password": "ASTRO_CLUSTER_PASSWORD",
    "ssh_key_filename": "ASTRO_CLUSTER_KEY",
}
STEPS_THAT_FEED_STA = {g.SPIKE_STEP, g.BEHAVIOR_STEP, g.ANATOMY_STEP}
ACTIVE_STATUSES = ("waiting", "pending", "queued", "running")  # a run in one of these is not finished and not failed
_EXECUTOR_LOCK = threading.Lock()
_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_SIZE = 0


def sta_steps() -> tuple[str, ...]:
    """The STA step of every hypothesis, in catalogue order."""
    return tuple(step for step in STEPS if is_sta_step(step))


def default_sta_step() -> str:
    """The STA step whose results every page shows until the pages can choose a hypothesis: the first one of the catalogue."""
    steps = sta_steps()
    if not steps:
        raise ValueError("The pipeline has no STA step: the hypothesis catalogue is empty.")
    return steps[0]


def hypothesis_for(step: str) -> Hypothesis:
    """The catalogue entry of an STA step."""
    name = hypothesis_name(step)
    for hypothesis in HYPOTHESES:
        if hypothesis.name == name:
            return hypothesis
    raise ValueError(f"Step {step} is not in the hypothesis catalogue ({catalogue_file(PROJECT_ROOT).name}).")


BEHAVIORAL_RESULT = "behavioral"  # the result type that needs the behavior split
STA_RESULT_TYPES = [  # the result types a run can be asked for (name, label on the page), in the order they are run
    (g.ALL_GROUP, "All"),
    (g.ANATOMY_GROUP, "Anatomy"),
    (BEHAVIORAL_RESULT, "Behavioral"),
]


def feeding_steps(result_types) -> set[str]:
    """The steps whose outputs an STA run of these result types reads: spike always, behavior only when the behavioral result type is marked (without it the
    run just reads the behavior run if one exists), and anatomy when that result type is marked."""
    return {g.SPIKE_STEP} | ({g.BEHAVIOR_STEP} if BEHAVIORAL_RESULT in result_types else set()) | ({g.ANATOMY_STEP} if g.ANATOMY_GROUP in result_types else set())


def pipeline_status(latest: dict[str, str], steps: tuple[str, ...] | None = None) -> dict:
    """The status of an experiment from the status of the newest real run of each step ({step: status}).

    'completed' only when EVERY step of the pipeline (STEPS, the one list of steps: the base steps and one STA step per hypothesis) has a newest
    run that completed; a failed rerun after an earlier success counts as failed. Otherwise the first of running / pending (waiting, pending or
    queued) / failed (steps in alphabetical order) wins; 'partial' = some steps completed and none is active or failed; 'none' = no step completed."""
    steps = STEPS if steps is None else steps
    latest = {s: st for s, st in latest.items() if s in steps}  # rows of steps that are no longer in the pipeline are ignored
    done = [s for s in steps if latest.get(s) == "completed"]
    status, step = "none", ""
    for name, wanted in (("running", ("running",)), ("pending", ("waiting", "pending", "queued")), ("failed", ("failed",))):
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


def create_run(settings: Settings, experiment_id: int, step: str, params: dict, backend: str = "local", neuron_id: int | None = None, is_test: bool = False,
               status: str = "pending") -> int:
    if step not in STEPS:
        raise ValueError(f"Unknown step: {step}")
    return run_repository.insert_run(settings, experiment_id, step, params, backend, neuron_id, is_test, status)


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
    """After a restart: launch every pending run, and start the waiting runs whose feeding steps have completed in the meantime."""
    run_ids = run_repository.pending_run_ids(settings)
    for run_id in run_ids:
        launch_run(settings, run_id)
    for experiment_id in sorted({int(run["experiment_id"]) for run in run_repository.waiting_runs(settings)}):
        run_ids += _launch_sta_if_ready(settings, experiment_id)
    return len(run_ids)


def rerun_failed_steps(settings: Settings, step: str | None = None) -> list[int]:
    if step is not None and step not in STEPS:
        raise ValueError(f"Unknown step: {step}")
    created: list[int] = []
    for run in run_repository.latest_failed_runs(settings, step):
        backend = "cluster" if is_sta_step(run["step"]) else (run["backend"] or "local")
        run_id = create_run(settings, int(run["experiment_id"]), str(run["step"]), json.loads(run["parameters_json"]),backend=backend, neuron_id=run["neuron_id"], is_test=False)
        launch_run(settings, run_id)
        created.append(run_id)
    return created


def launch_run(settings: Settings, run_id: int) -> None:
    global _EXECUTOR, _EXECUTOR_SIZE
    local_workers = int(settings.raw.get("local", {}).get("max_workers", 1))
    cluster_workers = int(settings.cluster.get("max_workers", 1))
    size = max(1, local_workers + cluster_workers)
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None or _EXECUTOR_SIZE != size:
            _EXECUTOR = ThreadPoolExecutor(max_workers=size, thread_name_prefix="astro-run")
            _EXECUTOR_SIZE = size
        executor = _EXECUTOR
    executor.submit(_execute_run, settings, run_id)


# ---- one run -------------------------------------------------------------------------------------------------------------------------------

def _execute_run(settings: Settings, run_id: int) -> None:
    """Run one step: build its command, run it, record the outcome. A cluster run only submits the job here; the cluster monitor finishes it."""
    run = run_repository.run_for_execution(settings, run_id)
    if run["backend"] == "cluster" and not settings.cluster.get("enabled", False):
        run_repository.update_run(settings, run_id, {"status": "failed", "ended_at": _now(), "command_json": json.dumps({"error": "Cluster backend disabled."})})
        return
    out = _allocate_run_dir(settings, run)
    out.mkdir(parents=True, exist_ok=False)
    stdout, stderr = path_for(PathKind.RUN_STDOUT_LOG, out), path_for(PathKind.RUN_STDERR_LOG, out)
    where = {"stdout_path": str(stdout), "stderr_path": str(stderr), "log_path": str(stdout), "output_dir": str(out)}
    if is_sta_step(run["step"]) and run["backend"] == "cluster" and _latest_completed_output(settings, int(run["experiment_id"]), g.SPIKE_STEP) is None:
        warning = "STA requires a completed spike run for the same experiment."
        stderr.write_text(f"{warning}\n", encoding="utf-8")
        now = _now()
        run_repository.update_run(settings, run_id, {"status": "failed", "started_at": now, "ended_at": now, "exit_code": 1, **where, "command_json": json.dumps({"error": warning})})
        return
    builder = _sta_command if is_sta_step(run["step"]) else _COMMAND_BUILDERS.get(run["step"])
    cmd = builder(settings, run, out) if builder else _adapter_command(run, out, run["parameters_json"])
    run_repository.update_run(settings, run_id, {"status": "running", "started_at": _now(), **where, "command_json": json.dumps(cmd)})
    code = _run_process(settings, cmd, stdout, stderr)
    _record_outcome(settings, run, out, cmd, code, where)
    if code == 0 and run["step"] in STEPS_THAT_FEED_STA and not run["is_test"]:
        _launch_sta_if_ready(settings, int(run["experiment_id"]))


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
    spike_run = _latest_completed_output(settings, experiment_id, g.SPIKE_STEP)
    params = json.loads(run["parameters_json"])
    params["spike_run"] = str(spike_run)
    for step, key in ((g.BEHAVIOR_STEP, "behavior_run"), (g.ANATOMY_STEP, "anatomy_run")):
        found = _latest_completed_output(settings, experiment_id, step)
        if found is not None:
            params[key] = str(found)
    # the cluster settings once, without the credentials: those reach the adapter through the environment (_process_env)
    params["cluster"] = {key: value for key, value in settings.cluster.items() if key not in CLUSTER_ENV_VARS}
    return _adapter_command(run, out, json.dumps(params))


def _decomposition_command(settings: Settings, run, out: Path) -> list[str]:
    """The adapter call of the decomposition step, with the folder of the newest completed STA run of its hypothesis (empty when there is none: the adapter says what to run first) and the
    cluster settings (without the credentials: those reach the adapter through the environment)."""
    sta_run = _latest_completed_output(settings, int(run["experiment_id"]), sta_step(g.DECOMPOSITION_HYPOTHESIS))
    params = {**json.loads(run["parameters_json"]), "sta_run": "" if sta_run is None else str(sta_run), "cluster": {key: value for key, value in settings.cluster.items() if key not in CLUSTER_ENV_VARS}}
    return _adapter_command(run, out, json.dumps(params))


# step -> function(settings, run, out) giving the command line; a step that is not listed is a plain adapter call
_COMMAND_BUILDERS = {g.PREPROCESSING_STEP: _preprocessing_command, g.DECOMPOSITION_STEP: _decomposition_command}


def _process_env(settings: Settings) -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(settings.project_root) + os.pathsep + env.get("PYTHONPATH", "")
    cluster = settings.cluster
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
    run_repository.finish_run(settings, int(run["id"]), fields, manifest, int(run["experiment_id"]), mark_preprocessing_running=submitted and run["step"] == g.PREPROCESSING_STEP)


def _launch_sta_if_ready(settings: Settings, experiment_id: int) -> list[int]:
    """Start the WAITING hypothesis runs of an experiment whose feeding steps have all completed; returns the ids it launched.

    Called when a feeding step completes. It creates no run (start_analysis does), and it leaves a waiting run alone while another run of
    the same hypothesis step is pending, queued or running."""
    completed = run_repository.steps_with_status(settings, experiment_id, ("completed",))
    busy = run_repository.steps_with_status(settings, experiment_id, ("pending", "queued", "running"))
    launched: list[int] = []
    for run in run_repository.waiting_runs(settings, experiment_id):
        run_id = int(run["id"])
        if run["step"] in busy or not feeding_steps(list(json.loads(run["parameters_json"]).get("result_types", []))) <= completed:
            continue
        if run_repository.release_waiting_run(settings, run_id):
            launch_run(settings, run_id)
            busy.add(run["step"])
            launched.append(run_id)
    return launched


def _latest_completed_output(settings: Settings, experiment_id: int, step: str) -> Path | None:
    row = run_repository.latest_run(settings, experiment_id, step)
    return None if row is None else run_dir(settings, int(row["id"]), bool(row["is_test"]))
