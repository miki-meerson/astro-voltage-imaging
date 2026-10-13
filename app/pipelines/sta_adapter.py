from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from app.cluster import client
from app.settings import PROJECT_ROOT
from utils import globals_utils as g
from utils.paths_utils import PathKind, cell_dirs, path_for, sta_dir_of_run
from utils.suite2p_utils import load_suite2p_context

DEFAULT_RESULT_TYPES = ["all", "simple", "complex", "behavioral"]


def _cluster(params: dict) -> dict:
    """The cluster settings the app passed in the params (without credentials: those come from the ASTRO_CLUSTER_* environment)."""
    return params.get("cluster") or {}


def _optional_path(params: dict, key: str) -> Path | None:
    return Path(params[key]) if params.get(key) else None


def _result_types(params: dict) -> list[str]:
    result_types = params.get("result_types", DEFAULT_RESULT_TYPES)
    if isinstance(result_types, str):
        return [result_types]
    return [str(item) for item in result_types]


def _spike_cell_ids(spike_run: Path) -> list[int]:
    return sorted(cell_dirs(spike_run))


def _validate_spike_suite2p_alignment(source_home: Path, spike_run: Path) -> None:
    suite_path = path_for(PathKind.SUITE2P_PLANE0, source_home)
    suite2p = load_suite2p_context(suite_path)
    n_cells = int(suite2p["n_cells"])
    spike_cells = _spike_cell_ids(spike_run)
    invalid = [cell_id for cell_id in spike_cells if cell_id < 0 or cell_id >= n_cells]
    if invalid:
        raise ValueError(
            "STA requires spike labels that match the current Suite2p accepted-cell list. "
            f"Current Suite2p has {n_cells} cells (valid ids 0..{max(0, n_cells - 1)}), "
            f"but spike run {spike_run.name} contains invalid cell folder(s): {invalid[:12]}. "
            "Refresh Suite2p Cells, rerun Spike Detection, and then rerun STA Maps."
        )


def _check_inputs_and_make_output_dir(source_home: Path, out: Path, params: dict) -> tuple[Path, Path, Path | None]:
    """Check that the input runs the STA reads exist and create the run's output folder. Returns (output home, spike run, behavior run)."""
    sta_home = path_for(PathKind.STA_ALGORITHM_HOME, out)
    sta_home.mkdir(parents=True, exist_ok=True)
    path_for(PathKind.RUN_STA_ANALYSIS, sta_home).mkdir(parents=True, exist_ok=True)

    spike_run = _optional_path(params, "spike_run")
    if spike_run is None or not spike_run.exists():
        raise FileNotFoundError("STA requires spike_run for the same experiment. Run spike analysis first.")
    _validate_spike_suite2p_alignment(source_home, spike_run)

    behavior_run = _optional_path(params, "behavior_run")
    if "behavioral" in _result_types(params):
        if behavior_run is None or not path_for(PathKind.BEHAVIOR_SEGMENTS, behavior_run).exists():
            raise FileNotFoundError("STA behavioral result_type requires a project-owned behavioral run output.")

    if "anatomy" in _result_types(params):
        anatomy_run = _optional_path(params, "anatomy_run")
        crop_path = path_for(PathKind.ANATOMY_CROP_PARAMS, anatomy_run) if anatomy_run else None
        if crop_path is None or not crop_path.exists():
            raise FileNotFoundError(
                "STA anatomy result_type was requested, but no completed app Anatomy Crop output was supplied. "
                "Run the Anatomy Crop step first, or rerun STA Maps without the Anatomy checkbox."
            )

    return sta_home, spike_run, behavior_run


def _write_config(sta_home: Path, out: Path, params: dict, source_home: Path, spike_run: Path, behavior_run: Path | None) -> Path:
    cfg = dict(params)
    cfg["home_dir"] = client.to_cluster_path(sta_home, _cluster(params))
    cfg["source_home_dir"] = client.to_cluster_path(source_home, _cluster(params))
    cfg["spike_run_dir"] = client.to_cluster_path(spike_run, _cluster(params))
    if behavior_run is not None:
        cfg["behavior_segments_path"] = client.to_cluster_path(path_for(PathKind.BEHAVIOR_SEGMENTS, behavior_run), _cluster(params))
    anatomy_run = _optional_path(params, "anatomy_run")
    if anatomy_run is not None:
        cfg["anatomy_crop_params_path"] = client.to_cluster_path(path_for(PathKind.ANATOMY_CROP_PARAMS, anatomy_run), _cluster(params))
    cfg["run_name"] = out.name
    cfg.setdefault("cell_ids", "all")
    cfg.setdefault("result_types", DEFAULT_RESULT_TYPES)
    cfg.setdefault("n_permutations", 200)
    cfg.setdefault("spatial_bin", 1)
    cfg.setdefault("anatomy_spatial_bin", 1)
    cfg.setdefault("voxel_alpha", 0.01)
    cfg.setdefault("cluster_alpha", 0.1)
    cfg.setdefault("rng_seed", 0)
    cfg.setdefault("analysis_half_window", 10)
    cfg.setdefault("n_workers", 1)
    cfg["summary_csv"] = path_for(PathKind.STA_SUMMARY_CSV, path_for(PathKind.RUN_STA_ANALYSIS, ".")).as_posix()
    cfg_path = path_for(PathKind.STA_JOB_PARAMS_EFFECTIVE, out)
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg_path


def _write_metrics(out: Path, sta_home: Path, config_path: Path, spike_run: Path) -> None:
    sta_dir = path_for(PathKind.RUN_STA_ANALYSIS, sta_home)
    npz_files = sorted(p for cell_dir in cell_dirs(sta_dir).values() for p in cell_dir.glob(g.RESULT_GLOB))
    rows = []
    for p in npz_files:
        rows.append({"path": str(p.relative_to(out)), "size_bytes": p.stat().st_size})
    metrics = {
        "algorithm": path_for(PathKind.STA_ALGORITHM_SCRIPT, "").as_posix(),
        "effective_config": str(config_path.relative_to(out)),
        "source_spike_run": str(spike_run),
        "sta_npz_count": len(npz_files),
        "sta_outputs": rows,
    }
    path_for(PathKind.RUN_METRICS_JSON, out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _submit_group_test(out: Path, source_home: Path, params: dict, sta_job_id: str) -> str | None:
    """Submit the anatomy group test (real vs shifted-spike results of all cells) as a Slurm job that starts when the STA job finished OK.

    Part of the pipeline: it needs the finished anatomy results, so it is chained with a dependency and needs no waiting here. A problem
    only prints a warning: the STA job is already submitted and does not depend on it. Returns the job id, or None."""
    if not params.get("run_group_test_after_sta", True):
        return None
    if "anatomy" not in _result_types(params):
        print("Group test not submitted: this run has no 'anatomy' result type.")
        return None
    script = path_for(PathKind.GROUP_TEST_JOB_SCRIPT, PROJECT_ROOT)
    if not script.exists():
        print(f"WARNING: group test not submitted: missing {script}", file=sys.stderr)
        return None
    sta_dir = sta_dir_of_run(out)
    log = path_for(PathKind.GROUP_TEST_STDOUT_LOG, out)
    cluster = _cluster(params)
    print("Submitting the anatomy group test (starts after the STA job):")
    try:
        job_id = client.submit_job(
            cluster, script, [sta_dir, source_home], path_for(PathKind.CLUSTER_CODE, PROJECT_ROOT), log,
            dependency=sta_job_id, kill_on_invalid_dep=True, error_log=log,
        )
    except client.CLUSTER_ERRORS as exc:
        print(f"WARNING: group test not submitted: {str(exc)[:300]}", file=sys.stderr)
        return None
    path_for(PathKind.GROUP_TEST_JOB_ID, out).write_text(job_id + "\n", encoding="utf-8")
    print(f"Submitted group test job {job_id} (after STA job {sta_job_id}); it writes {g.GROUP_TEST} in the STA folder.")
    return job_id


def _sta_script(cluster: dict) -> Path:
    script = Path(cluster.get("sta_script") or path_for(PathKind.STA_JOB_SCRIPT, PROJECT_ROOT))
    if not script.is_absolute():
        script = PROJECT_ROOT / script
    if not script.exists():
        raise FileNotFoundError(f"Missing cluster STA script: {script}")
    return script


def _run_cluster_job(config_path: Path, out: Path, params: dict, source_home: Path | None = None) -> int:
    cluster = _cluster(params)
    client.check_ready(cluster)
    cluster_script = _sta_script(cluster)
    cluster_stdout = path_for(PathKind.CLUSTER_STDOUT_LOG, out)
    cluster_stderr = path_for(PathKind.CLUSTER_STDERR_LOG, out)
    print("Submitting STA Maps cluster job:")
    print(f"Cluster stdout: {client.to_cluster_path(cluster_stdout, cluster)}")
    print(f"Cluster stderr: {client.to_cluster_path(cluster_stderr, cluster)}")
    try:
        job_id = client.submit_job(cluster, cluster_script, [config_path], path_for(PathKind.CLUSTER_CODE, PROJECT_ROOT), cluster_stdout, error_log=cluster_stderr)
    except RuntimeError as exc:
        print(f"Cluster submission failed: {exc}", file=sys.stderr)
        return 1
    path_for(PathKind.CLUSTER_JOB_ID, out).write_text(job_id + "\n", encoding="utf-8")
    print(f"Submitted Slurm job {job_id}")
    print("Slurm job submitted; the web app will monitor completion.")
    if source_home is not None:
        _submit_group_test(out, source_home, params, job_id)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--params-json", default="{}")
    args = ap.parse_args()

    source_home = Path(args.experiment)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    params = json.loads(args.params_json or "{}")

    sta_home, spike_run, behavior_run = _check_inputs_and_make_output_dir(source_home, out, params)
    config_path = _write_config(sta_home, out, params, source_home, spike_run, behavior_run)
    return_code = _run_cluster_job(config_path, out, params, source_home)
    if return_code != 0:
        raise SystemExit(return_code)
    _write_metrics(out, sta_home, config_path, spike_run)


if __name__ == "__main__":
    main()
