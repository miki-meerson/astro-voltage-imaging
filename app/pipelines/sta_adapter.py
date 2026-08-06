from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import time

from app.services.cluster_ssh import load_cluster_credentials, run_ssh_command


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STA_SCRIPT = PROJECT_ROOT / "cluster-code" / "sta_maps_sparse.py"
BEHAVIOR_SEGMENTS = "behavioral_segements.npz"


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
    return value.replace("\\", "/")


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
    try:
        import paramiko  # noqa: F401
    except ImportError as exc:
        raise FileNotFoundError("paramiko is required for cluster submission but is not installed.") from exc
    _load_cluster_credentials(params)


def _run_slurm_command(command: list[str], params: dict, cwd: Path | None = None) -> subprocess.CompletedProcess:
    stdout, stderr, code = _run_paramiko_command(shlex.join(command), params)
    return subprocess.CompletedProcess(command, code, stdout, stderr)


def _load_cluster_credentials(params: dict):
    return load_cluster_credentials(
        {
            "ssh_host": params.get("cluster_ssh_host"),
            "ssh_port": params.get("cluster_ssh_port"),
            "ssh_username": params.get("cluster_ssh_username"),
            "ssh_password": params.get("cluster_ssh_password"),
            "ssh_key_filename": params.get("cluster_ssh_key_filename"),
            "paramiko_timeout_seconds": params.get("cluster_paramiko_timeout_seconds"),
        }
    )


def _run_paramiko_command(command: str, params: dict) -> tuple[str, str, int]:
    attempts = int(params.get("cluster_paramiko_attempts") or 5)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return run_ssh_command(
                {
                    "ssh_host": params.get("cluster_ssh_host"),
                    "ssh_port": params.get("cluster_ssh_port"),
                    "ssh_username": params.get("cluster_ssh_username"),
                    "ssh_password": params.get("cluster_ssh_password"),
                    "ssh_key_filename": params.get("cluster_ssh_key_filename"),
                    "paramiko_timeout_seconds": params.get("cluster_paramiko_timeout_seconds"),
                },
                command,
            )
        except Exception as exc:
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
    raise RuntimeError(f"Paramiko command failed: {last_exc}")


def _link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _link_or_copy_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst, dirs_exist_ok=True)


def _copy_spike_labels(spike_run: Path, mirror_home: Path) -> None:
    dest_root = mirror_home / "automated-spike-analysis"
    dest_root.mkdir(parents=True, exist_ok=True)
    for src_cell in sorted(spike_run.glob("cell_*")):
        if not src_cell.is_dir():
            continue
        dst_cell = dest_root / src_cell.name
        dst_cell.mkdir(exist_ok=True)
        for name in (
            "simple_complex_spike_labels.npz",
            "simple_complex_peaks_for_analysis.npz",
            "peak_detection_params.npz",
        ):
            src = src_cell / name
            if src.exists():
                shutil.copy2(src, dst_cell / name)


def _result_types(params: dict) -> list[str]:
    result_types = params.get("result_types", ["all", "simple", "complex", "behavioral"])
    if isinstance(result_types, str):
        return [result_types]
    return [str(item) for item in result_types]


def _prepare_mirror(source_home: Path, out: Path, params: dict) -> tuple[Path, Path]:
    mirror_home = out / "sta_algorithm_home"
    if mirror_home.exists():
        attempt = 2
        while (out / f"sta_algorithm_home_attempt_{attempt}").exists():
            attempt += 1
        mirror_home = out / f"sta_algorithm_home_attempt_{attempt}"
    mirror_home.mkdir(parents=True)

    _link_or_copy_file(source_home / "Experiment.xml", mirror_home / "Experiment.xml")
    _link_or_copy_file(
        source_home / "pipeline_results" / "pb_corrected" / "astro_photobleaching_corrected.tif",
        mirror_home / "pipeline_results" / "pb_corrected" / "astro_photobleaching_corrected.tif",
    )
    suite_src = source_home / "pipeline_results" / "for_suite2p" / "suite2p" / "plane0"
    _link_or_copy_dir(suite_src, mirror_home / "pipeline_results" / "for_suite2p" / "suite2p" / "plane0")

    spike_run = Path(params.get("spike_run") or "") if params.get("spike_run") else None
    if spike_run is None or not spike_run.exists():
        raise FileNotFoundError("STA requires spike_run for the same experiment. Run spike analysis first.")
    _copy_spike_labels(spike_run, mirror_home)

    behavior_run = Path(params.get("behavior_run") or "") if params.get("behavior_run") else None
    result_types = _result_types(params)
    anatomy_run = Path(params.get("anatomy_run") or "") if params.get("anatomy_run") else None
    src_crop = anatomy_run / "sta-analysis" / "anatomy_crop_params.npz" if anatomy_run else None
    if "behavioral" in result_types:
        if behavior_run is None or not (behavior_run / BEHAVIOR_SEGMENTS).exists():
            raise FileNotFoundError("STA behavioral result_type requires a project-owned behavioral run output.")
        sta_dir = mirror_home / "sta-analysis"
        sta_dir.mkdir(exist_ok=True)
        shutil.copy2(behavior_run / BEHAVIOR_SEGMENTS, sta_dir / BEHAVIOR_SEGMENTS)

    if "anatomy" in result_types:
        if src_crop is None or not src_crop.exists():
            raise FileNotFoundError(
                "STA anatomy result_type was requested, but no completed app Anatomy Crop output was supplied. "
                "Run the Anatomy Crop step first, or rerun STA Maps without the Anatomy checkbox."
            )
        sta_dir = mirror_home / "sta-analysis"
        sta_dir.mkdir(exist_ok=True)
        shutil.copy2(src_crop, sta_dir / "anatomy_crop_params.npz")

    return mirror_home, spike_run


def _prepare_cluster_mirror(source_home: Path, out: Path, params: dict) -> tuple[Path, Path, Path | None]:
    mirror_home = out / "sta_algorithm_home"
    mirror_home.mkdir(parents=True, exist_ok=True)
    (mirror_home / "sta-analysis").mkdir(parents=True, exist_ok=True)

    spike_run = Path(params.get("spike_run") or "") if params.get("spike_run") else None
    if spike_run is None or not spike_run.exists():
        raise FileNotFoundError("STA requires spike_run for the same experiment. Run spike analysis first.")

    behavior_run = Path(params.get("behavior_run") or "") if params.get("behavior_run") else None
    if "behavioral" in _result_types(params):
        if behavior_run is None or not (behavior_run / BEHAVIOR_SEGMENTS).exists():
            raise FileNotFoundError("STA behavioral result_type requires a project-owned behavioral run output.")

    if "anatomy" in _result_types(params):
        anatomy_run = Path(params.get("anatomy_run") or "") if params.get("anatomy_run") else None
        crop_path = anatomy_run / "sta-analysis" / "anatomy_crop_params.npz" if anatomy_run else None
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
    behavioral_segments = _sh_single(BEHAVIOR_SEGMENTS)
    return f"""
set -euo pipefail
SOURCE_HOME={source}
MIRROR_HOME={mirror}
SPIKE_RUN={spike}
BEHAVIOR_RUN={behavior}
ANATOMY_RUN={anatomy}
BEHAVIOR_SEGMENTS={behavioral_segments}

echo "Preparing cluster STA mirror at $MIRROR_HOME"
mkdir -p "$MIRROR_HOME/pipeline_results/pb_corrected"
mkdir -p "$MIRROR_HOME/pipeline_results/for_suite2p/suite2p"
mkdir -p "$MIRROR_HOME/pipeline_results/qc"
mkdir -p "$MIRROR_HOME/sta-analysis"

ln -sfn "$SOURCE_HOME/Experiment.xml" "$MIRROR_HOME/Experiment.xml"
ln -sfn "$SOURCE_HOME/Image_scan_1_region_0_0.tif" "$MIRROR_HOME/Image_scan_1_region_0_0.tif"
for behavior_csv in "$SOURCE_HOME"/imaging_*.csv; do
  if [ -e "$behavior_csv" ]; then ln -sfn "$behavior_csv" "$MIRROR_HOME/$(basename "$behavior_csv")"; fi
done
ln -sfn "$SOURCE_HOME/pipeline_results/pb_corrected/astro_photobleaching_corrected.tif" "$MIRROR_HOME/pipeline_results/pb_corrected/astro_photobleaching_corrected.tif"
if [ -e "$SOURCE_HOME/pipeline_results/pb_corrected/neuron_photobleaching_corrected.tif" ]; then
  ln -sfn "$SOURCE_HOME/pipeline_results/pb_corrected/neuron_photobleaching_corrected.tif" "$MIRROR_HOME/pipeline_results/pb_corrected/neuron_photobleaching_corrected.tif"
fi
if [ -e "$SOURCE_HOME/pipeline_results/qc/bad_segments.json" ]; then
  ln -sfn "$SOURCE_HOME/pipeline_results/qc/bad_segments.json" "$MIRROR_HOME/pipeline_results/qc/bad_segments.json"
fi
rm -rf "$MIRROR_HOME/pipeline_results/for_suite2p/suite2p/plane0"
ln -sfn "$SOURCE_HOME/pipeline_results/for_suite2p/suite2p/plane0" "$MIRROR_HOME/pipeline_results/for_suite2p/suite2p/plane0"

rm -rf "$MIRROR_HOME/automated-spike-analysis"
ln -sfn "$SPIKE_RUN" "$MIRROR_HOME/automated-spike-analysis"
if [ -n "$BEHAVIOR_RUN" ] && [ -e "$BEHAVIOR_RUN/$BEHAVIOR_SEGMENTS" ]; then
  ln -sfn "$BEHAVIOR_RUN/$BEHAVIOR_SEGMENTS" "$MIRROR_HOME/sta-analysis/$BEHAVIOR_SEGMENTS"
fi
if [ -n "$ANATOMY_RUN" ] && [ -e "$ANATOMY_RUN/sta-analysis/anatomy_crop_params.npz" ]; then
  ln -sfn "$ANATOMY_RUN/sta-analysis/anatomy_crop_params.npz" "$MIRROR_HOME/sta-analysis/anatomy_crop_params.npz"
fi
"""


def _write_config(mirror_home: Path, out: Path, params: dict, cluster_paths: bool = False) -> Path:
    cfg = dict(params)
    cfg["home_dir"] = _cluster_path(mirror_home, params) if cluster_paths else str(mirror_home)
    cfg.setdefault("cell_ids", "all")
    cfg.setdefault("result_types", ["all", "simple", "complex", "behavioral"])
    cfg.setdefault("n_permutations", 200)
    cfg.setdefault("spatial_bin", 8)
    cfg.setdefault("voxel_alpha", 0.01)
    cfg.setdefault("cluster_alpha", 0.1)
    cfg.setdefault("rng_seed", 0)
    cfg.setdefault("analysis_half_window", 10)
    cfg.setdefault("n_workers", 1)
    cfg["summary_csv"] = "sta-analysis/sta_maps_batch_summary.csv"
    cfg_path = out / "sta_maps_job_params.effective.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg_path


def _write_metrics(out: Path, mirror_home: Path, config_path: Path, spike_run: Path) -> None:
    sta_dir = mirror_home / "sta-analysis"
    npz_files = sorted(sta_dir.glob("cell_*/astro_sta_cluster_permutation*.npz"))
    rows = []
    for p in npz_files:
        rows.append({"path": str(p.relative_to(out)), "size_bytes": p.stat().st_size})
    metrics = {
        "algorithm": "cluster-code/sta_maps_sparse.py",
        "effective_config": str(config_path.relative_to(out)),
        "source_spike_run": str(spike_run),
        "sta_npz_count": len(npz_files),
        "sta_outputs": rows,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _run_cluster_job(config_path: Path, out: Path, params: dict, preflight: str = "") -> int:
    submit_command = _command_parts(params.get("cluster_submit_command") or "sbatch")
    _submit_command_available(submit_command, params)
    cluster_script = Path(params.get("cluster_script") or (PROJECT_ROOT / "cluster-code" / "run_sta_maps.sh"))
    if not cluster_script.is_absolute():
        cluster_script = PROJECT_ROOT / cluster_script
    if not cluster_script.exists():
        raise FileNotFoundError(f"Missing cluster STA script: {cluster_script}")
    cluster_stdout = out / "cluster_stdout.log"
    cluster_stderr = out / "cluster_stderr.log"
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
        f"--chdir={_cluster_path(STA_SCRIPT.parent, params)}",
        f"--output={_cluster_path(cluster_stdout, params)}",
        f"--error={_cluster_path(cluster_stderr, params)}",
        _cluster_path(cluster_script, params),
        _cluster_path(config_path, params),
    ]
    print("Submitting STA Maps cluster job:")
    print(" ".join(cmd))
    print(f"Cluster stdout: {_cluster_path(cluster_stdout, params)}")
    print(f"Cluster stderr: {_cluster_path(cluster_stderr, params)}")
    completed = _run_slurm_command(cmd, params, cwd=STA_SCRIPT.parent)
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.stderr:
        print(completed.stderr.strip(), file=sys.stderr)
    if completed.returncode != 0:
        print(f"Cluster submit command exited with code {completed.returncode}")
        return int(completed.returncode)
    job_id = _parse_sbatch_job_id(completed.stdout)
    if not job_id:
        print("Could not parse a Slurm job id from sbatch output; treating submission as failed.", file=sys.stderr)
        return 1
    (out / "cluster_job_id.txt").write_text(job_id + "\n", encoding="utf-8")
    print(f"Submitted Slurm job {job_id}")
    if params.get("cluster_wait_for_completion"):
        return _wait_for_slurm_job(job_id, params)
    print("Slurm job submitted; the web app will monitor completion.")
    return 0


def _parse_sbatch_job_id(text: str) -> str | None:
    for token in text.replace(".", " ").split():
        if token.isdigit():
            return token
    return None


def _wait_for_slurm_job(job_id: str, params: dict) -> int:
    interval = max(5, int(params.get("cluster_poll_interval_seconds") or 30))
    squeue = _command_parts(params.get("cluster_squeue_command") or "squeue")
    sacct = _command_parts(params.get("cluster_sacct_command") or "sacct")
    print(f"Polling Slurm job {job_id} every {interval} seconds.")
    while True:
        sq = _run_slurm_command(squeue + ["-h", "-j", job_id, "-o", "%T"], params)
        if sq.returncode == 0 and sq.stdout.strip():
            state = sq.stdout.strip().splitlines()[0].strip()
            print(f"Slurm job {job_id}: {state}", flush=True)
            time.sleep(interval)
            continue
        acct = _run_slurm_command(sacct + ["-n", "-j", job_id, "--format=State,ExitCode", "-P"], params)
        if acct.returncode != 0:
            if acct.stderr:
                print(acct.stderr.strip(), file=sys.stderr)
            print(f"Job {job_id} is no longer in squeue, but sacct status is unavailable.")
            return 0
        states = [line.split("|", 1)[0].strip() for line in acct.stdout.splitlines() if line.strip()]
        print(f"Slurm accounting for {job_id}: {', '.join(states) or 'no rows'}")
        if any(state.startswith(("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")) for state in states):
            return 1
        if any(state.startswith("COMPLETED") for state in states):
            return 0
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--params-json", default="{}")
    ap.add_argument("--submit-cluster", action="store_true")
    args = ap.parse_args()

    source_home = Path(args.experiment)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    params = json.loads(args.params_json or "{}")

    if not STA_SCRIPT.exists():
        raise FileNotFoundError(f"Missing STA script: {STA_SCRIPT}")

    if args.submit_cluster:
        mirror_home, spike_run, behavior_run = _prepare_cluster_mirror(source_home, out, params)
        config_path = _write_config(mirror_home, out, params, cluster_paths=True)
        preflight = _cluster_preflight(source_home, mirror_home, spike_run, behavior_run, params)
        return_code = _run_cluster_job(config_path, out, params, preflight)
    else:
        mirror_home, spike_run = _prepare_mirror(source_home, out, params)
        config_path = _write_config(mirror_home, out, params)
        cmd = [sys.executable, "-u", str(STA_SCRIPT), str(config_path)]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        print("Running real STA algorithm:")
        print(" ".join(cmd))
        completed = subprocess.run(cmd, cwd=STA_SCRIPT.parent, env=env, text=True)
        return_code = int(completed.returncode)
    if return_code != 0:
        raise SystemExit(return_code)
    _write_metrics(out, mirror_home, config_path, spike_run)


if __name__ == "__main__":
    main()
