from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from app.cluster import client
from app.db import preprocessing_repository as repo
from app.db import run_repository
from app.settings import Settings
from utils import globals_utils as g
from utils.file_utils import read_json, write_json
from utils.paths_utils import PathKind, path_for


PREPROCESSING_STEPS = ("split_2ch", "motion_correction", "photobleaching_correction")
STEP_LABELS = {
    "split_2ch": "Split 2 Channels",
    "motion_correction": "Motion Correction",
    "photobleaching_correction": "Photobleaching Correction",
}
RUNNER_SCRIPTS = {
    "split_2ch": "run_split_2ch.sh",
    "motion_correction": "run_motion_correction.sh",
    "photobleaching_correction": "run_photobleaching_correction.sh",
}


def default_preprocessing_params(settings: Settings, source_path: str | Path) -> dict:
    defaults = dict(settings.defaults.get(g.PREPROCESSING_STEP, {}))
    defaults.setdefault("steps", list(PREPROCESSING_STEPS))
    defaults.setdefault("pw_rigid", False)
    defaults.setdefault("gSig_filt", 3)
    defaults.setdefault("max_shifts_x", 50)
    defaults.setdefault("max_shifts_y", 50)
    defaults.setdefault("strides_x", 48)
    defaults.setdefault("strides_y", 48)
    defaults.setdefault("overlaps_x", 24)
    defaults.setdefault("overlaps_y", 24)
    defaults.setdefault("max_deviation_rigid", 3)
    defaults.setdefault("trimmed", 0)
    defaults["raw_video_path"] = str(path_for(PathKind.RAW_MOVIE, source_path))
    defaults["home_dir"] = str(Path(source_path))
    return defaults


def write_params_file(settings: Settings, run: dict, out: Path, params: dict) -> Path:
    now = datetime.now().strftime("%d-%m-%Y___%H-%M-%S")
    raw_video = Path(params.get("raw_video_path") or path_for(PathKind.RAW_MOVIE, run["source_path"]))
    home = Path(run["source_path"])
    cluster = settings.cluster
    payload = {
        **params,
        g.PARAM_GUI_TIME: f"app_run_{run['id']}__{now}",
        "raw_video_path": str(raw_video),
        g.PARAM_RAW_VIDEO_PATH_LINUX: client.to_cluster_path(raw_video, cluster),
        "home_dir": str(home),
        "home_dir_linux": client.to_cluster_path(home, cluster),
    }
    path_for(PathKind.PREPROCESSING_PARAMS_DIR, out).mkdir(parents=True, exist_ok=True)
    path = path_for(PathKind.PREPROCESSING_PARAMS, out, run_id=run["id"])
    write_json(path, payload)
    return path


def save_state(out: Path, state: dict) -> None:
    write_json(path_for(PathKind.PREPROCESSING_STATE, out), state)


def initialize_preprocessing_run(settings: Settings, run: dict, out: Path) -> dict:
    params = {**default_preprocessing_params(settings, run["source_path"]), **json.loads(run["parameters_json"])}
    steps = [step for step in params.get("steps", PREPROCESSING_STEPS) if step in PREPROCESSING_STEPS]
    if not steps:
        raise ValueError("No preprocessing steps were selected.")
    params_path = write_params_file(settings, run, out, params)
    state = {
        "run_id": int(run["id"]),
        "params_path": str(params_path),
        "steps": [
            {
                "name": step,
                "label": STEP_LABELS[step],
                "status": "pending",
                "job_id": "",
                "log_path": "",
                "submitted_at": "",
                "ended_at": "",
                "message": "",
            }
            for step in steps
        ],
        "current_index": 0,
    }
    save_state(out, state)
    repo.replace_preprocessing_steps(settings, int(run["id"]), steps)
    return submit_preprocessing_step(settings, out, state, 0)


def submit_preprocessing_step(settings: Settings, out: Path, state: dict, index: int) -> dict:
    step = state["steps"][index]
    script = path_for(PathKind.PREPROCESSING_RUNNER, settings.project_root, script=RUNNER_SCRIPTS[step["name"]])
    if not script.exists():
        raise FileNotFoundError(f"Missing preprocessing runner script: {script}")
    cluster = settings.cluster
    log_path = out / f"preprocessing_{step['name']}_%j.log"
    job_id = client.submit_job(
        cluster, script, [Path(state["params_path"])], path_for(PathKind.PREPROCESSING_SOURCE, settings.project_root), log_path
    )
    real_log_path = str(log_path).replace("%j", job_id)
    message = f"Submitted batch job {job_id}"
    step.update(
        {
            "status": "queued",
            "job_id": job_id,
            "log_path": client.to_cluster_path(real_log_path, cluster),
            "local_log_path": real_log_path,
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "message": message,
        }
    )
    state["current_index"] = index
    save_state(out, state)
    path_for(PathKind.CLUSTER_JOB_ID, out).write_text(job_id + "\n", encoding="utf-8")
    repo.mark_step_queued(settings, state["run_id"], step["name"], job_id, real_log_path, step["submitted_at"], message)
    return state


def _apply_step_rerun_params(state: dict, extra_params: dict | None) -> dict:
    params_path = Path(state["params_path"])
    params = read_json(params_path, "The preprocessing params file")
    for key, value in (extra_params or {}).items():
        if key == "trimmed":
            params[key] = max(0, int(value))
        else:
            params[key] = value
    write_json(params_path, params)
    return params


def rerun_preprocessing_step(settings: Settings, experiment_id: int, step_name: str, extra_params: dict | None = None) -> dict:
    if step_name not in PREPROCESSING_STEPS:
        raise ValueError(f"Unknown preprocessing step: {step_name}")
    run = run_repository.latest_run(settings, experiment_id, g.PREPROCESSING_STEP, status=None)
    if run is None:
        raise ValueError("No preprocessing run exists for this experiment.")
    if repo.run_is_active(settings, run["id"]):
        raise ValueError("A preprocessing run is already active.")
    step_row = repo.step_row(settings, run["id"], step_name)
    if step_row is None:
        raise ValueError(f"Preprocessing step was not found: {step_name}")
    rerunnable_statuses = {"blocked", "cancelled", "completed", "failed"}
    if step_row["status"] not in rerunnable_statuses:
        allowed = ", ".join(sorted(rerunnable_statuses))
        raise ValueError(f"This step can be rerun only from: {allowed}. Current status is {step_row['status']}.")

    out = Path(run["output_dir"] or "")
    state = read_json(path_for(PathKind.PREPROCESSING_STATE, out), "The preprocessing state file", default={})
    if not state:
        raise ValueError("Preprocessing state file is missing.")
    steps = state.get("steps", [])
    index = next((i for i, step in enumerate(steps) if step.get("name") == step_name), None)
    if index is None:
        raise ValueError(f"Step is missing from preprocessing state: {step_name}")
    if extra_params:
        _apply_step_rerun_params(state, extra_params)

    for i in range(index, len(steps)):
        steps[i].update(
            {
                "status": "submitting" if i == index else "pending",
                "job_id": "",
                "log_path": "",
                "local_log_path": "",
                "submitted_at": "",
                "ended_at": "",
                "message": "",
            }
        )
    state["current_index"] = index
    save_state(out, state)
    repo.reset_for_rerun(
        settings,
        experiment_id,
        run["id"],
        Path(state["params_path"]).read_text(encoding="utf-8"),
        [(step["name"], "submitting" if i == index else "pending") for i, step in enumerate(steps) if i >= index],
    )
    try:
        return submit_preprocessing_step(settings, out, state, index)
    except Exception as exc:
        steps[index]["status"] = "failed"
        steps[index]["message"] = str(exc)
        save_state(out, state)
        repo.mark_rerun_failed(settings, run["id"], step_name, str(exc), datetime.now().isoformat(timespec="seconds"))
        raise


def latest_preprocessing_params(settings: Settings, experiment_id: int) -> dict:
    """Parameters of the newest preprocessing run: the params file its state file points to, else the parameters saved with the run."""
    run = run_repository.latest_run(settings, experiment_id, g.PREPROCESSING_STEP, status=None)
    if run is None:
        return {}
    state = read_json(path_for(PathKind.PREPROCESSING_STATE, Path(run["output_dir"] or "")), "The preprocessing state file", default={})
    params_path = Path(state.get("params_path") or "")
    if state.get("params_path") and params_path.exists():
        return read_json(params_path, "The preprocessing params file")
    return json.loads(run["parameters_json"])
