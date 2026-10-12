from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shlex

from app.db.core import connect, tx
from app.services.cluster_ssh import run_ssh_command
from app.settings import Settings


PREPROCESSING_STEPS = ("split_2ch", "motion_correction", "photobleaching_correction")
STEP_LABELS = {
    "split_2ch": "Split 2 Channels",
    "motion_correction": "Motion Correction",
    "photobleaching_correction": "Photobleaching Correction",
}
PREPROCESSING_SOURCE_DIR = Path("preprocessing") / "source" / "pipeline_steps"
RUNNER_DIR = PREPROCESSING_SOURCE_DIR / "cluster_runners_scripts"
RUNNER_SCRIPTS = {
    "split_2ch": "run_split_2ch.sh",
    "motion_correction": "run_motion_correction.sh",
    "photobleaching_correction": "run_photobleaching_correction.sh",
}


def cluster_path(path: Path | str, settings_or_params) -> str:
    value = str(path)
    raw = settings_or_params.raw if hasattr(settings_or_params, "raw") else settings_or_params
    mapping = (raw.get("cluster", {}) if "cluster" in raw else raw).get("path_map", {})
    to_root = str(mapping.get("to_cluster") or "").rstrip("/")
    from_roots = [
        str(mapping.get("from_windows") or "").rstrip("\\/"),
        str(mapping.get("from_windows_unc") or "").rstrip("\\/"),
    ]
    for from_root in [root for root in from_roots if root]:
        if to_root and value.lower().startswith(from_root.lower()):
            suffix = value[len(from_root):].lstrip("\\/")
            return to_root + ("/" + suffix.replace("\\", "/") if suffix else "")
    return value.replace("\\", "/")


def raw_movie_path(experiment_path: str | Path) -> Path:
    home = Path(experiment_path)
    candidates = [
        home / "Image_scan_1_region_0_0.tif",
        home / "Image_scan_1_region_0_0.raw",
        home / "raw_video.tif",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raw_files = sorted(home.glob("*.raw")) + sorted(home.glob("*.tif")) + sorted(home.glob("*.tiff"))
    if raw_files:
        return raw_files[0]
    raise FileNotFoundError(f"No raw movie file found in {home}")


def default_preprocessing_params(settings: Settings, source_path: str | Path) -> dict:
    defaults = dict(settings.raw.get("defaults", {}).get("preprocessing", {}))
    defaults.setdefault("is_2ch", True)
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
    defaults["raw_video_path"] = str(raw_movie_path(source_path))
    defaults["home_dir"] = str(Path(source_path))
    return defaults


def write_params_file(settings: Settings, run: dict, out: Path, params: dict) -> Path:
    now = datetime.now().strftime("%d-%m-%Y___%H-%M-%S")
    raw_video = Path(params.get("raw_video_path") or raw_movie_path(run["source_path"]))
    home = Path(run["source_path"])
    payload = {
        **params,
        "gui_time": f"app_run_{run['id']}__{now}",
        "raw_video_path": str(raw_video),
        "raw_video_path_linux": cluster_path(raw_video, settings),
        "home_dir": str(home),
        "home_dir_linux": cluster_path(home, settings),
        "is_2ch": bool(params.get("is_2ch", True)),
    }
    params_dir = out / "pipeline_params"
    params_dir.mkdir(parents=True, exist_ok=True)
    path = params_dir / f"run_{run['id']}_preprocessing_params.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def state_path(out: Path) -> Path:
    return out / "preprocessing_state.json"


def load_state(out: Path) -> dict:
    path = state_path(out)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(out: Path, state: dict) -> None:
    state_path(out).write_text(json.dumps(state, indent=2), encoding="utf-8")


def initialize_preprocessing_run(settings: Settings, run: dict, out: Path) -> dict:
    params = json.loads(run["parameters_json"] or "{}")
    params = {**default_preprocessing_params(settings, run["source_path"]), **params}
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
    with tx(settings) as conn:
        conn.execute("DELETE FROM preprocessing_steps WHERE run_id=?", (run["id"],))
        for index, step in enumerate(steps):
            conn.execute(
                "INSERT INTO preprocessing_steps(run_id,step_name,step_order,status) VALUES(?,?,?,?)",
                (run["id"], step, index, "pending"),
            )
    return submit_preprocessing_step(settings, out, state, 0)


def submit_preprocessing_step(settings: Settings, out: Path, state: dict, index: int) -> dict:
    step = state["steps"][index]
    script = settings.project_root / RUNNER_DIR / RUNNER_SCRIPTS[step["name"]]
    if not script.exists():
        raise FileNotFoundError(f"Missing preprocessing runner script: {script}")
    cluster = settings.raw.get("cluster", {})
    submit = str(cluster.get("submit_command") or "/usr/bin/sbatch")
    params_path = Path(state["params_path"])
    log_path = out / f"preprocessing_{step['name']}_%j.log"
    cmd = " ".join(
        [
            shlex.quote(submit),
            f"--chdir={shlex.quote(cluster_path(settings.project_root / PREPROCESSING_SOURCE_DIR, settings))}",
            f"--output={shlex.quote(cluster_path(log_path, settings))}",
            shlex.quote(cluster_path(script, settings)),
            shlex.quote(cluster_path(params_path, settings)),
        ]
    )
    stdout, stderr, code = run_ssh_command(cluster, cmd)
    if code != 0:
        raise RuntimeError(f"Failed to submit {step['name']}: {(stderr or stdout).strip()}")
    job_id = next((token for token in stdout.replace(".", " ").split() if token.isdigit()), "")
    if not job_id:
        raise RuntimeError(f"Could not parse Slurm job id from sbatch output: {stdout!r}")
    real_log_path = str(log_path).replace("%j", job_id)
    remote_log_path = cluster_path(Path(real_log_path), settings)
    step.update(
        {
            "status": "queued",
            "job_id": job_id,
            "log_path": remote_log_path,
            "local_log_path": real_log_path,
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "message": stdout.strip(),
        }
    )
    state["current_index"] = index
    save_state(out, state)
    (out / "cluster_job_id.txt").write_text(job_id + "\n", encoding="utf-8")
    with tx(settings) as conn:
        conn.execute(
            """
            UPDATE preprocessing_steps
            SET status='queued',slurm_job_id=?,slurm_log_path=?,submitted_at=?,message=?
            WHERE run_id=? AND step_name=?
            """,
            (job_id, real_log_path, step["submitted_at"], stdout.strip(), state["run_id"], step["name"]),
        )
    return state


def _load_params_file(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f"Preprocessing params file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_params_file(path: Path, params: dict) -> None:
    path.write_text(json.dumps(params, indent=2), encoding="utf-8")


def _apply_step_rerun_params(state: dict, extra_params: dict | None) -> dict:
    params_path = Path(state["params_path"])
    params = _load_params_file(params_path)
    for key, value in (extra_params or {}).items():
        if key == "trimmed":
            params[key] = max(0, int(value))
        else:
            params[key] = value
    _save_params_file(params_path, params)
    return params


def rerun_preprocessing_step(settings: Settings, experiment_id: int, step_name: str, extra_params: dict | None = None) -> dict:
    if step_name not in PREPROCESSING_STEPS:
        raise ValueError(f"Unknown preprocessing step: {step_name}")
    with connect(settings) as conn:
        run = conn.execute(
            """
            SELECT r.*
            FROM analysis_runs r
            WHERE r.experiment_id=?
              AND r.step='preprocessing'
              AND r.is_test=0
            ORDER BY r.id DESC
            LIMIT 1
            """,
            (experiment_id,),
        ).fetchone()
        if run is None:
            raise ValueError("No preprocessing run exists for this experiment.")
        run = dict(run)
        active = conn.execute(
            """
            SELECT 1
            FROM analysis_runs
            WHERE id=? AND status IN ('pending','queued','running')
            """,
            (run["id"],),
        ).fetchone()
        if active is not None:
            raise ValueError("A preprocessing run is already active.")
        step_row = conn.execute(
            """
            SELECT *
            FROM preprocessing_steps
            WHERE run_id=? AND step_name=?
            """,
            (run["id"], step_name),
        ).fetchone()
        if step_row is None:
            raise ValueError(f"Preprocessing step was not found: {step_name}")
        rerunnable_statuses = {"blocked", "cancelled", "completed", "failed"}
        if step_row["status"] not in rerunnable_statuses:
            allowed = ", ".join(sorted(rerunnable_statuses))
            raise ValueError(f"This step can be rerun only from: {allowed}. Current status is {step_row['status']}.")

    out = Path(run["output_dir"] or "")
    state = load_state(out)
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
            (Path(state["params_path"]).read_text(encoding="utf-8"), run["id"]),
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
        for i, step in enumerate(steps):
            if i < index:
                continue
            status = "submitting" if i == index else "pending"
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
                (status, run["id"], step["name"]),
            )
    try:
        return submit_preprocessing_step(settings, out, state, index)
    except Exception as exc:
        steps[index]["status"] = "failed"
        steps[index]["message"] = str(exc)
        save_state(out, state)
        with tx(settings) as conn:
            conn.execute(
                """
                UPDATE preprocessing_steps
                SET status='failed',message=?,ended_at=?
                WHERE run_id=? AND step_name=?
                """,
                (str(exc)[:1000], datetime.now().isoformat(timespec="seconds"), run["id"], step_name),
            )
            conn.execute(
                "UPDATE analysis_runs SET status='failed',ended_at=?,exit_code=1,review_note=? WHERE id=?",
                (datetime.now().isoformat(timespec="seconds"), str(exc)[:1000], run["id"]),
            )
        raise
