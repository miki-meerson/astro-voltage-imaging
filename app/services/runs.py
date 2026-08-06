from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import json, os, shlex, subprocess, sys, threading

from app.db.core import tx
from app.services.cluster_ssh import load_cluster_credentials, run_ssh_command
from app.settings import Settings

STEPS = ("spike", "behavior", "anatomy", "sta")
_EXECUTOR_LOCK = threading.Lock()
_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_SIZE = 0


def _safe_name(value: str) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")[:120] or "experiment"


def create_run(settings: Settings, experiment_id: int, step: str, params: dict, backend: str = "local", neuron_id: int | None = None, is_test: bool = False) -> int:
    if step not in STEPS:
        raise ValueError(f"Unknown step: {step}")
    with tx(settings) as conn:
        run_id = _next_run_id(conn)
        conn.execute(
            "INSERT INTO analysis_runs(id,experiment_id,neuron_id,step,backend,status,is_test,parameters_json) VALUES(?,?,?,?,?,?,?,?)",
            (run_id, experiment_id, neuron_id, step, backend, "pending", int(is_test), json.dumps(params)),
        )
        return run_id


def _next_run_id(conn) -> int:
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


def run_dir(settings: Settings, run_id: int, is_test: bool = False) -> Path:
    root = settings.test_output_root if is_test else settings.output_root
    legacy = root / f"run_{run_id}"
    try:
        with tx(settings) as conn:
            row = conn.execute(
                """
                SELECT r.id,r.step,r.neuron_id,r.output_dir,r.stdout_path,e.canonical_id,n.pipeline_neuron_id
                FROM analysis_runs r
                JOIN experiments e ON e.id=r.experiment_id
                LEFT JOIN neurons n ON n.id=r.neuron_id
                WHERE r.id=?
                """,
                (run_id,),
            ).fetchone()
    except Exception:
        row = None
    if row is None:
        return legacy
    cell_id = row["pipeline_neuron_id"] if row["pipeline_neuron_id"] is not None else "all"
    dirname = f"{_safe_name(row['canonical_id'])}__cell_{_safe_name(cell_id)}__{_safe_name(row['step'])}__run_{run_id}"
    formatted = root / dirname
    candidates = []
    if row["output_dir"]:
        candidates.append(Path(row["output_dir"]))
    if row["stdout_path"]:
        candidates.append(Path(row["stdout_path"]).parent)
    candidates.extend([formatted, legacy])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if row["output_dir"]:
        return Path(row["output_dir"])
    return formatted


def _allocate_run_dir(settings: Settings, run) -> Path:
    root = settings.test_output_root if run["is_test"] else settings.output_root
    if run["output_dir"]:
        return Path(run["output_dir"])
    cell_id = run["pipeline_neuron_id"] if run["pipeline_neuron_id"] is not None else "all"
    base = root / f"{_safe_name(run['canonical_id'])}__cell_{_safe_name(cell_id)}__{_safe_name(run['step'])}__run_{run['id']}"
    candidate = base
    attempt = 2
    while candidate.exists():
        candidate = root / f"{base.name}__attempt_{attempt}"
        attempt += 1
    return candidate


def prune_to_latest_runs(settings: Settings) -> None:
    return None


def resume_pending_runs(settings: Settings) -> int:
    with tx(settings) as conn:
        run_ids = [
            int(row["id"])
            for row in conn.execute(
                "SELECT id FROM analysis_runs WHERE status='pending' ORDER BY id"
            )
        ]
    for run_id in run_ids:
        launch_run(settings, run_id)
    return len(run_ids)


def rerun_failed_steps(settings: Settings, step: str | None = None) -> list[int]:
    if step is not None and step not in STEPS:
        raise ValueError(f"Unknown step: {step}")
    params: list[object] = []
    step_filter = ""
    if step is not None:
        step_filter = "AND r.step=?"
        params.append(step)
    with tx(settings) as conn:
        failed = [
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
    created: list[int] = []
    for run in failed:
        try:
            run_params = json.loads(run["parameters_json"] or "{}")
        except json.JSONDecodeError:
            run_params = {}
        backend = "cluster" if run["step"] == "sta" else (run["backend"] or "local")
        run_id = create_run(
            settings,
            int(run["experiment_id"]),
            str(run["step"]),
            run_params,
            backend=backend,
            neuron_id=run["neuron_id"],
            is_test=False,
        )
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


def _execute_run(settings: Settings, run_id: int) -> None:
    with tx(settings) as conn:
        run = conn.execute(
            """
            SELECT r.*,e.source_path,e.canonical_id,n.pipeline_neuron_id
            FROM analysis_runs r
            JOIN experiments e ON e.id=r.experiment_id
            LEFT JOIN neurons n ON n.id=r.neuron_id
            WHERE r.id=?
            """,
            (run_id,),
        ).fetchone()
        if run["backend"] == "cluster" and not settings.raw.get("cluster", {}).get("enabled", False):
            conn.execute("UPDATE analysis_runs SET status='failed',ended_at=?,command_json=? WHERE id=?", (datetime.now().isoformat(timespec="seconds"), json.dumps({"error": "Cluster backend disabled."}), run_id))
            return
    out = _allocate_run_dir(settings, run); out.mkdir(parents=True, exist_ok=False)
    stdout, stderr = out / "stdout.log", out / "stderr.log"
    adapter = {"spike": "spike_adapter", "behavior": "behavior_adapter", "anatomy": "anatomy_adapter", "sta": "sta_adapter"}[run["step"]]
    cmd = [sys.executable, "-u", "-m", f"app.pipelines.{adapter}", "--experiment", run["source_path"], "--output", str(out), "--params-json", run["parameters_json"]]
    if run["backend"] == "cluster" and run["step"] == "sta":
        raw_params = json.loads(run["parameters_json"] or "{}")
        spike_run = _latest_completed_output(settings, int(run["experiment_id"]), "spike")
        behavior_run = _latest_completed_output(settings, int(run["experiment_id"]), "behavior")
        anatomy_run = _latest_completed_output(settings, int(run["experiment_id"]), "anatomy")
        if spike_run is None:
            message = "STA requires a completed spike run for the same experiment."
            stderr.write_text(message + "\n", encoding="utf-8")
            with tx(settings) as conn:
                conn.execute(
                    "UPDATE analysis_runs SET status='failed',started_at=?,ended_at=?,exit_code=1,stdout_path=?,stderr_path=?,log_path=?,output_dir=?,command_json=? WHERE id=?",
                    (
                        datetime.now().isoformat(timespec="seconds"),
                        datetime.now().isoformat(timespec="seconds"),
                        str(stdout),
                        str(stderr),
                        str(stdout),
                        str(out),
                        json.dumps({"error": message}),
                        run_id,
                    ),
                )
            return
        raw_params["spike_run"] = str(spike_run)
        if behavior_run is not None:
            raw_params["behavior_run"] = str(behavior_run)
        if anatomy_run is not None:
            raw_params["anatomy_run"] = str(anatomy_run)
        raw_params["cluster_submit_command"] = settings.raw.get("cluster", {}).get("submit_command", "sbatch")
        raw_params["cluster_script"] = settings.raw.get("cluster", {}).get("sta_script", "cluster-code/run_sta_maps.sh")
        raw_params["cluster_submission_method"] = settings.raw.get("cluster", {}).get("submission_method", "paramiko")
        raw_params["cluster_paramiko_timeout_seconds"] = settings.raw.get("cluster", {}).get("paramiko_timeout_seconds", 600)
        raw_params["cluster_paramiko_attempts"] = settings.raw.get("cluster", {}).get("paramiko_attempts", 5)
        raw_params["cluster_path_map"] = settings.raw.get("cluster", {}).get("path_map", {})
        raw_params["cluster_poll_interval_seconds"] = settings.raw.get("cluster", {}).get("poll_interval_seconds", 30)
        raw_params["cluster_squeue_command"] = settings.raw.get("cluster", {}).get("squeue_command", "squeue")
        raw_params["cluster_sacct_command"] = settings.raw.get("cluster", {}).get("sacct_command", "sacct")
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "app.pipelines.sta_adapter",
            "--experiment",
            run["source_path"],
            "--output",
            str(out),
            "--params-json",
            json.dumps(raw_params),
            "--submit-cluster",
        ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(settings.project_root) + os.pathsep + env.get("PYTHONPATH", "")
    cluster_settings = settings.raw.get("cluster", {})
    env_map = {
        "ssh_host": "ASTRO_CLUSTER_HOST",
        "ssh_port": "ASTRO_CLUSTER_PORT",
        "ssh_username": "ASTRO_CLUSTER_USER",
        "ssh_password": "ASTRO_CLUSTER_PASSWORD",
        "ssh_key_filename": "ASTRO_CLUSTER_KEY",
    }
    for key, env_key in env_map.items():
        value = cluster_settings.get(key)
        if value not in (None, ""):
            env[env_key] = str(value)
    with tx(settings) as conn:
        conn.execute(
            "UPDATE analysis_runs SET status='running',started_at=?,stdout_path=?,stderr_path=?,log_path=?,output_dir=?,command_json=? WHERE id=?",
            (datetime.now().isoformat(timespec="seconds"), str(stdout), str(stderr), str(stdout), str(out), json.dumps(cmd), run_id),
        )
    code = 1
    try:
        with stdout.open("w", encoding="utf-8") as so, stderr.open("w", encoding="utf-8") as se:
            code = subprocess.run(cmd, cwd=settings.project_root, env=env, text=True, stdout=so, stderr=se).returncode
    except Exception as exc:
        stderr.write_text(str(exc), encoding="utf-8")
    manifest = [{"path": str(p.relative_to(out.parent)), "type": p.suffix.lower().lstrip(".") or "file", "size": p.stat().st_size} for p in out.rglob("*") if p.is_file()]
    cluster_submitted = run["backend"] == "cluster" and run["step"] == "sta" and code == 0 and (out / "cluster_job_id.txt").exists()
    final_status = "queued" if cluster_submitted else ("completed" if code == 0 else "failed")
    final_exit_code = None if cluster_submitted else code
    ended_at = None if cluster_submitted else datetime.now().isoformat(timespec="seconds")
    with tx(settings) as conn:
        conn.execute("UPDATE analysis_runs SET status=?,ended_at=?,exit_code=?,stdout_path=?,stderr_path=?,log_path=?,command_json=?,output_manifest_json=?,code_provenance=? WHERE id=?",
                     (final_status, ended_at, final_exit_code, str(stdout), str(stderr), str(stdout), json.dumps(cmd), json.dumps(manifest), "docs/provenance.md", run_id))
        for item in manifest:
            conn.execute("INSERT INTO artifacts(run_id,artifact_type,relative_path,size_bytes) VALUES(?,?,?,?)", (run_id, item["type"], item["path"], item["size"]))
    if code == 0 and run["step"] in {"spike", "behavior", "anatomy"} and not run["is_test"]:
        _launch_sta_if_ready(settings, int(run["experiment_id"]))


def _launch_sta_if_ready(settings: Settings, experiment_id: int) -> None:
    with tx(settings) as conn:
        completed = {
            r["step"]
            for r in conn.execute(
                "SELECT step FROM analysis_runs WHERE experiment_id=? AND is_test=0 AND status='completed'",
                (experiment_id,),
            )
        }
        exp = conn.execute("SELECT source_path FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        params = _sta_params_for_experiment(settings, exp["source_path"] if exp else "")
        required = {"spike", "behavior"}
        if "anatomy" in params.get("result_types", []):
            required.add("anatomy")
        if not required.issubset(completed):
            return
        existing_sta = conn.execute(
            "SELECT id FROM analysis_runs WHERE experiment_id=? AND is_test=0 AND step='sta' AND status IN ('pending','queued','running','completed') LIMIT 1",
            (experiment_id,),
        ).fetchone()
        if existing_sta is not None:
            return
        sta_run_id = _next_run_id(conn)
        existing_sta = conn.execute(
            "SELECT id FROM analysis_runs WHERE experiment_id=? AND is_test=0 AND step='sta' AND status IN ('pending','queued','running','completed') LIMIT 1",
            (experiment_id,),
        ).fetchone()
        if existing_sta is not None:
            return
        conn.execute(
            "INSERT INTO analysis_runs(id,experiment_id,step,backend,status,is_test,parameters_json) VALUES(?,?,?,?,?,0,?)",
            (sta_run_id, experiment_id, "sta", "cluster", "pending", json.dumps(params)),
        )
    launch_run(settings, sta_run_id)


def _latest_completed_output(settings: Settings, experiment_id: int, step: str) -> Path | None:
    with tx(settings) as conn:
        row = conn.execute(
            """
            SELECT id,is_test,output_dir,stdout_path
            FROM analysis_runs
            WHERE experiment_id=? AND step=? AND is_test=0 AND status='completed'
            ORDER BY id DESC LIMIT 1
            """,
            (experiment_id, step),
        ).fetchone()
    if row is None:
        return None
    return run_dir(settings, int(row["id"]), bool(row["is_test"]))


def _sta_params_for_experiment(settings: Settings, source_path: str | Path) -> dict:
    return dict(settings.raw.get("defaults", {}).get("sta", {}))


def _cluster_ready_error(settings: Settings) -> str | None:
    cluster = settings.raw.get("cluster", {})
    submit = str(cluster.get("submit_command") or "sbatch")
    method = str(cluster.get("submission_method") or "paramiko").strip().lower()
    if method != "paramiko":
        return "Cluster submission blocked: STA cluster submission supports only Paramiko remote sbatch."
    try:
        import paramiko  # noqa: F401
    except ImportError as exc:
        return f"Cluster submission blocked: paramiko is not available in this environment. {exc}"
    try:
        load_cluster_credentials(cluster)
        out, err, code = run_ssh_command(cluster, f"command -v {shlex.quote(submit)}")
        if code != 0:
            return f"Cluster submission blocked: submit command is not reachable through Paramiko: {submit}. {(err or out).strip()}"
    except Exception as exc:
        return f"Cluster submission blocked: Paramiko could not connect to the cluster. {type(exc).__name__}: {exc}"
    return None
