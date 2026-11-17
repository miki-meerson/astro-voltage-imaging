"""The adapter of a step that is one Slurm job over the saved results of other steps (the decomposition, the SNAPT maps): reads the adapter arguments, submits the sbatch script of the step
with the folders the step reads, and writes the job id; the web app follows the job and completes the run when it ends. Each adapter module calls `main` with its own script and the
function that gives the job's arguments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from app.cluster import client
from app.settings import PROJECT_ROOT
from utils.paths_utils import PathKind, path_for


def main(label: str, script_kind: PathKind, job_args: Callable[[dict, Path, str], list]) -> None:
    """`job_args(params, out, experiment)` gives the arguments of the sbatch script (it raises FileNotFoundError, saying what to run first, when an input run is missing)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--params-json", default="{}")
    args = ap.parse_args()
    out, params = Path(args.output), json.loads(args.params_json or "{}")
    out.mkdir(parents=True, exist_ok=True)
    arguments = job_args(params, out, args.experiment)
    cluster = params.get("cluster") or {}
    client.check_ready(cluster)
    script = path_for(script_kind, PROJECT_ROOT)
    if not script.exists():
        raise FileNotFoundError(f"The cluster script {script} is missing.")
    stdout, stderr = path_for(PathKind.CLUSTER_STDOUT_LOG, out), path_for(PathKind.CLUSTER_STDERR_LOG, out)
    print(f"Submitting the {label} cluster job:")
    try:
        job_id = client.submit_job(cluster, script, arguments, path_for(PathKind.CLUSTER_CODE, PROJECT_ROOT), stdout, error_log=stderr)
    except RuntimeError as exc:
        print(f"Cluster submission failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    path_for(PathKind.CLUSTER_JOB_ID, out).write_text(job_id + "\n", encoding="utf-8")
    print(f"Submitted Slurm job {job_id}; the web app will monitor completion.")
