from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time

import paramiko

from app.services.cluster_ssh import load_cluster_credentials, run_ssh_command
from app.settings import PROJECT_ROOT
from utils import globals_utils as g
from utils.paths_utils import PathKind, cell_dirs, path_for, sta_dir_of_run
from utils.suite2p_utils import load_suite2p_context

DEFAULT_RESULT_TYPES = ["all", "simple", "complex", "behavioral"]
DATA_SHARE_NAME = "Adam-Lab-Shared"  # the folder of the lab share that the cluster mounts at path_map["to_cluster"]


def _cluster_path(path: Path | str, params: dict) -> str:
    value = str(path)
    mapping = params.get("cluster_path_map") or {}
    to_root = str(mapping.get("to_cluster") or "").rstrip("/")
    from_roots = [
        str(mapping.get("from_windows") or "").rstrip("\\/"),
        str(mapping.get("from_windows_unc") or "").rstrip("\\/"),
    ]
    for from_root in [root for root in from_roots if root]:
        if to_root and value.lower().startswith(from_root.lower()):
            suffix = value[len(from_root):].lstrip("\\/")
            return to_root + ("/" + suffix.replace("\\", "/") if suffix else "")
    normalized = value.replace("\\", "/")
    marker = DATA_SHARE_NAME
    if to_root and marker in normalized:
        suffix = normalized.split(marker, 1)[1].lstrip("/")
        return to_root + ("/" + suffix if suffix else "")
    return normalized


def _command_parts(command: str | list[str]) -> list[str]:
    if isinstance(command, list):
        return [str(part) for part in command]
    return shlex.split(str(command))


def _submission_method(params: dict) -> str:
    return str(params.get("cluster_submission_method") or "paramiko").strip().lower()


def _submit_command_available(command: list[str], params: dict) -> None:
    if not command:
        raise ValueError("cluster_submit_command is empty.")
    if _submission_method(params) != "paramiko":
        raise ValueError("Cluster STA submission supports only Paramiko remote sbatch.")
    load_cluster_credentials(_ssh_settings(params))


def _ssh_settings(params: dict) -> dict:
    return {
        "ssh_host": params.get("cluster_ssh_host"),
        "ssh_port": params.get("cluster_ssh_port"),
        "ssh_username": params.get("cluster_ssh_username"),
        "ssh_password": params.get("cluster_ssh_password"),
        "ssh_key_filename": params.get("cluster_ssh_key_filename"),
        "paramiko_timeout_seconds": params.get("cluster_paramiko_timeout_seconds"),
    }


def _run_paramiko_command(command: str, params: dict) -> tuple[str, str, int]:
    attempts = int(params.get("cluster_paramiko_attempts") or 5)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return run_ssh_command(_ssh_settings(params), command)
        except (OSError, paramiko.SSHException, EOFError) as exc:  # connection problems; anything else is a bug and propagates
            last_exc = exc
            if attempt == attempts:
                raise
            delay = min(30, 2 * attempt)
            print(
                f"Paramiko command attempt {attempt}/{attempts} failed: {type(exc).__name__}: {exc}. "
                f"Retrying in {delay}s.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"Paramiko command was not run: cluster_paramiko_attempts is {attempts}, it must be at least 1.")


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


def _prepare_cluster_mirror(source_home: Path, out: Path, params: dict) -> tuple[Path, Path, Path | None]:
    mirror_home = path_for(PathKind.STA_ALGORITHM_HOME, out)
    mirror_home.mkdir(parents=True, exist_ok=True)
    path_for(PathKind.RUN_STA_ANALYSIS, mirror_home).mkdir(parents=True, exist_ok=True)

    spike_run = Path(params.get("spike_run") or "") if params.get("spike_run") else None
    if spike_run is None or not spike_run.exists():
        raise FileNotFoundError("STA requires spike_run for the same experiment. Run spike analysis first.")
    _validate_spike_suite2p_alignment(source_home, spike_run)

    behavior_run = Path(params.get("behavior_run") or "") if params.get("behavior_run") else None
    if "behavioral" in _result_types(params):
        if behavior_run is None or not path_for(PathKind.BEHAVIOR_SEGMENTS, behavior_run).exists():
            raise FileNotFoundError("STA behavioral result_type requires a project-owned behavioral run output.")

    if "anatomy" in _result_types(params):
        anatomy_run = Path(params.get("anatomy_run") or "") if params.get("anatomy_run") else None
        crop_path = path_for(PathKind.ANATOMY_CROP_PARAMS, anatomy_run) if anatomy_run else None
        if crop_path is None or not crop_path.exists():
            raise FileNotFoundError(
                "STA anatomy result_type was requested, but no completed app Anatomy Crop output was supplied. "
                "Run the Anatomy Crop step first, or rerun STA Maps without the Anatomy checkbox."
            )

    return mirror_home, spike_run, behavior_run


def _sh_single(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _cluster_preflight(source_home: Path, mirror_home: Path, spike_run: Path, behavior_run: Path | None, params: dict) -> str:
    source = _sh_single(_cluster_path(source_home, params))
    mirror = _sh_single(_cluster_path(mirror_home, params))
    spike = _sh_single(_cluster_path(spike_run, params))
    behavior = _sh_single(_cluster_path(behavior_run, params)) if behavior_run else "''"
    anatomy_run = Path(params.get("anatomy_run") or "") if params.get("anatomy_run") else None
    anatomy = _sh_single(_cluster_path(anatomy_run, params)) if anatomy_run else "''"

    def at(kind: PathKind, root: str, **fields) -> str:
        return path_for(kind, root, **fields).as_posix()  # a shell path under $SOURCE_HOME, $MIRROR_HOME or $ANATOMY_RUN

    mirror_sta = at(PathKind.RUN_STA_ANALYSIS, "$MIRROR_HOME")
    return f"""
set -euo pipefail
SOURCE_HOME={source}
MIRROR_HOME={mirror}
SPIKE_RUN={spike}
BEHAVIOR_RUN={behavior}
ANATOMY_RUN={anatomy}

echo "Preparing cluster STA mirror at $MIRROR_HOME"
mkdir -p "{at(PathKind.PB_CORRECTED_DIR, "$MIRROR_HOME")}"
mkdir -p "{at(PathKind.SUITE2P_PLANE0, "$MIRROR_HOME")}"
mkdir -p "{at(PathKind.QC_DIR, "$MIRROR_HOME")}"
mkdir -p "{mirror_sta}"

ln -sfn "{at(PathKind.EXPERIMENT_XML, "$SOURCE_HOME")}" "{at(PathKind.EXPERIMENT_XML, "$MIRROR_HOME")}"
ln -sfn "{at(PathKind.RAW_MOVIE, "$SOURCE_HOME")}" "{at(PathKind.RAW_MOVIE, "$MIRROR_HOME")}"
for behavior_csv in "$SOURCE_HOME"/{g.IMAGING_CSV_GLOB}; do
  if [ -e "$behavior_csv" ]; then ln -sfn "$behavior_csv" "$MIRROR_HOME/$(basename "$behavior_csv")"; fi
done
ln -sfn "{at(PathKind.ASTRO_PB_MOVIE, "$SOURCE_HOME")}" "{at(PathKind.ASTRO_PB_MOVIE, "$MIRROR_HOME")}"
if [ -e "{at(PathKind.NEURON_PB_MOVIE, "$SOURCE_HOME")}" ]; then
  ln -sfn "{at(PathKind.NEURON_PB_MOVIE, "$SOURCE_HOME")}" "{at(PathKind.NEURON_PB_MOVIE, "$MIRROR_HOME")}"
fi
if [ -e "{at(PathKind.BAD_SEGMENTS, "$SOURCE_HOME")}" ]; then
  ln -sfn "{at(PathKind.BAD_SEGMENTS, "$SOURCE_HOME")}" "{at(PathKind.BAD_SEGMENTS, "$MIRROR_HOME")}"
fi
rm -rf "{at(PathKind.SUITE2P_PLANE0, "$MIRROR_HOME")}"
ln -sfn "{at(PathKind.SUITE2P_PLANE0, "$SOURCE_HOME")}" "{at(PathKind.SUITE2P_PLANE0, "$MIRROR_HOME")}"

rm -rf "{at(PathKind.SPIKE_RUN_LINK, "$MIRROR_HOME")}"
ln -sfn "$SPIKE_RUN" "{at(PathKind.SPIKE_RUN_LINK, "$MIRROR_HOME")}"
if [ -n "$BEHAVIOR_RUN" ] && [ -e "{at(PathKind.BEHAVIOR_SEGMENTS, "$BEHAVIOR_RUN")}" ]; then
  ln -sfn "{at(PathKind.BEHAVIOR_SEGMENTS, "$BEHAVIOR_RUN")}" "{at(PathKind.STA_BEHAVIOR_SEGMENTS, mirror_sta)}"
fi
if [ -n "$ANATOMY_RUN" ] && [ -e "{at(PathKind.ANATOMY_CROP_PARAMS, "$ANATOMY_RUN")}" ]; then
  ln -sfn "{at(PathKind.ANATOMY_CROP_PARAMS, "$ANATOMY_RUN")}" "{at(PathKind.ANATOMY_CROP_PARAMS, "$MIRROR_HOME")}"
fi
"""


def _write_config(mirror_home: Path, out: Path, params: dict, source_home: Path) -> Path:
    cfg = dict(params)
    cfg["home_dir"] = _cluster_path(mirror_home, params)
    cfg["source_home_dir"] = _cluster_path(source_home, params)
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


def _write_metrics(out: Path, mirror_home: Path, config_path: Path, spike_run: Path) -> None:
    sta_dir = path_for(PathKind.RUN_STA_ANALYSIS, mirror_home)
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
    cmd = _command_parts(params.get("cluster_submit_command") or "sbatch") + [
        f"--dependency=afterok:{sta_job_id}",
        "--kill-on-invalid-dep=yes",
        f"--chdir={_cluster_path(path_for(PathKind.CLUSTER_CODE, PROJECT_ROOT), params)}",
        f"--output={_cluster_path(log, params)}",
        f"--error={_cluster_path(log, params)}",
        _cluster_path(script, params),
        _cluster_path(sta_dir, params),
        _cluster_path(source_home, params),
    ]
    print("Submitting the anatomy group test (starts after the STA job):")
    print(" ".join(cmd))
    try:
        stdout, stderr, code = _run_paramiko_command(shlex.join(cmd), params)
    except (OSError, ValueError, RuntimeError, paramiko.SSHException) as exc:
        print(f"WARNING: group test not submitted: {exc}", file=sys.stderr)
        return None
    job_id = _parse_sbatch_job_id(stdout) if code == 0 else None
    if not job_id:
        print(f"WARNING: group test not submitted (sbatch exit {code}): {(stderr or stdout).strip()[:300]}", file=sys.stderr)
        return None
    path_for(PathKind.GROUP_TEST_JOB_ID, out).write_text(job_id + "\n", encoding="utf-8")
    print(f"Submitted group test job {job_id} (after STA job {sta_job_id}); it writes {g.GROUP_TEST} in the STA folder.")
    return job_id


def _run_cluster_job(config_path: Path, out: Path, params: dict, preflight: str = "", source_home: Path | None = None) -> int:
    submit_command = _command_parts(params.get("cluster_submit_command") or "sbatch")
    _submit_command_available(submit_command, params)
    cluster_script = Path(params.get("cluster_script") or path_for(PathKind.STA_JOB_SCRIPT, PROJECT_ROOT))
    if not cluster_script.is_absolute():
        cluster_script = PROJECT_ROOT / cluster_script
    if not cluster_script.exists():
        raise FileNotFoundError(f"Missing cluster STA script: {cluster_script}")
    cluster_stdout = path_for(PathKind.CLUSTER_STDOUT_LOG, out)
    cluster_stderr = path_for(PathKind.CLUSTER_STDERR_LOG, out)
    if preflight.strip():
        print("Preparing cluster STA mirror through remote shell commands.")
        preflight_out, preflight_err, preflight_code = _run_paramiko_command(preflight, params)
        if preflight_out:
            print(preflight_out.strip())
        if preflight_err:
            print(preflight_err.strip(), file=sys.stderr)
        if preflight_code != 0:
            print(f"Cluster mirror preparation exited with code {preflight_code}", file=sys.stderr)
            return int(preflight_code)
    cmd = submit_command + [
        f"--chdir={_cluster_path(path_for(PathKind.CLUSTER_CODE, PROJECT_ROOT), params)}",
        f"--output={_cluster_path(cluster_stdout, params)}",
        f"--error={_cluster_path(cluster_stderr, params)}",
        _cluster_path(cluster_script, params),
        _cluster_path(config_path, params),
    ]
    print("Submitting STA Maps cluster job:")
    print(" ".join(cmd))
    print(f"Cluster stdout: {_cluster_path(cluster_stdout, params)}")
    print(f"Cluster stderr: {_cluster_path(cluster_stderr, params)}")
    stdout, stderr, code = _run_paramiko_command(shlex.join(cmd), params)
    if stdout:
        print(stdout.strip())
    if stderr:
        print(stderr.strip(), file=sys.stderr)
    if code != 0:
        print(f"Cluster submit command exited with code {code}")
        return int(code)
    job_id = _parse_sbatch_job_id(stdout)
    if not job_id:
        print("Could not parse a Slurm job id from sbatch output; treating submission as failed.", file=sys.stderr)
        return 1
    path_for(PathKind.CLUSTER_JOB_ID, out).write_text(job_id + "\n", encoding="utf-8")
    print(f"Submitted Slurm job {job_id}")
    print("Slurm job submitted; the web app will monitor completion.")
    if source_home is not None:
        _submit_group_test(out, source_home, params, job_id)
    return 0


def _parse_sbatch_job_id(text: str) -> str | None:
    for token in text.replace(".", " ").split():
        if token.isdigit():
            return token
    return None


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

    mirror_home, spike_run, behavior_run = _prepare_cluster_mirror(source_home, out, params)
    config_path = _write_config(mirror_home, out, params, source_home)
    preflight = _cluster_preflight(source_home, mirror_home, spike_run, behavior_run, params)
    return_code = _run_cluster_job(config_path, out, params, preflight, source_home)
    if return_code != 0:
        raise SystemExit(return_code)
    _write_metrics(out, mirror_home, config_path, spike_run)


if __name__ == "__main__":
    main()
