from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time

from app.services.cluster_ssh import load_cluster_credentials, run_ssh_command
from utils.suite2p_utils import load_suite2p_context


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
    normalized = value.replace("\\", "/")
    marker = "Adam-Lab-Shared"
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
    try:
        import paramiko  # noqa: F401
    except ImportError as exc:
        raise FileNotFoundError("paramiko is required for cluster submission but is not installed.") from exc
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


def _result_types(params: dict) -> list[str]:
    result_types = params.get("result_types", ["all", "simple", "complex", "behavioral"])
    if isinstance(result_types, str):
        return [result_types]
    return [str(item) for item in result_types]


def _spike_cell_ids(spike_run: Path) -> list[int]:
    cell_ids = []
    for path in spike_run.glob("cell_*"):
        if not path.is_dir():
            continue
        try:
            cell_ids.append(int(path.name.split("_", 1)[1]))
        except ValueError:
            continue
    return sorted(cell_ids)


def _validate_spike_suite2p_alignment(source_home: Path, spike_run: Path) -> None:
    suite_path = source_home / "pipeline_results" / "for_suite2p" / "suite2p" / "plane0"
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
    mirror_home = out / "sta_algorithm_home"
    mirror_home.mkdir(parents=True, exist_ok=True)
    (mirror_home / "sta-analysis").mkdir(parents=True, exist_ok=True)

    spike_run = Path(params.get("spike_run") or "") if params.get("spike_run") else None
    if spike_run is None or not spike_run.exists():
        raise FileNotFoundError("STA requires spike_run for the same experiment. Run spike analysis first.")
    _validate_spike_suite2p_alignment(source_home, spike_run)

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


def _write_config(mirror_home: Path, out: Path, params: dict, source_home: Path) -> Path:
    cfg = dict(params)
    cfg["home_dir"] = _cluster_path(mirror_home, params)
    cfg["source_home_dir"] = _cluster_path(source_home, params)
    cfg.setdefault("cell_ids", "all")
    cfg.setdefault("result_types", ["all", "simple", "complex", "behavioral"])
    cfg.setdefault("n_permutations", 200)
    cfg.setdefault("spatial_bin", 1)
    cfg.setdefault("anatomy_spatial_bin", 1)
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
    (out / "cluster_job_id.txt").write_text(job_id + "\n", encoding="utf-8")
    print(f"Submitted Slurm job {job_id}")
    print("Slurm job submitted; the web app will monitor completion.")
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

    if not STA_SCRIPT.exists():
        raise FileNotFoundError(f"Missing STA script: {STA_SCRIPT}")

    mirror_home, spike_run, behavior_run = _prepare_cluster_mirror(source_home, out, params)
    config_path = _write_config(mirror_home, out, params, source_home)
    preflight = _cluster_preflight(source_home, mirror_home, spike_run, behavior_run, params)
    return_code = _run_cluster_job(config_path, out, params, preflight)
    if return_code != 0:
        raise SystemExit(return_code)
    _write_metrics(out, mirror_home, config_path, spike_run)


if __name__ == "__main__":
    main()
