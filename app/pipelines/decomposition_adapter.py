"""The decomposition step: submits the Slurm job that makes the PCA and ICA of the whole-field STA movies of the newest completed STA run of the fast hypothesis (utils/decomposition_run.py).
The web app follows the job and completes the run when it ends."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.cluster import client
from app.settings import PROJECT_ROOT
from utils.paths_utils import PathKind, path_for, sta_dir_of_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--params-json", default="{}")
    args = ap.parse_args()
    out, params = Path(args.output), json.loads(args.params_json or "{}")
    out.mkdir(parents=True, exist_ok=True)
    sta_run = Path(params["sta_run"]) if params.get("sta_run") else None
    if sta_run is None or not sta_dir_of_run(sta_run).exists():
        raise FileNotFoundError("The decomposition needs a completed STA Maps run of the fast hypothesis (with the 'all' result type): run STA Maps for it first.")
    cluster = params.get("cluster") or {}
    client.check_ready(cluster)
    script = path_for(PathKind.DECOMPOSITION_JOB_SCRIPT, PROJECT_ROOT)
    if not script.exists():
        raise FileNotFoundError(f"The cluster script {script} is missing.")
    stdout, stderr = path_for(PathKind.CLUSTER_STDOUT_LOG, out), path_for(PathKind.CLUSTER_STDERR_LOG, out)
    print("Submitting the decomposition cluster job:")
    try:
        job_id = client.submit_job(cluster, script, [sta_dir_of_run(sta_run), out, args.experiment], path_for(PathKind.CLUSTER_CODE, PROJECT_ROOT), stdout, error_log=stderr)
    except RuntimeError as exc:
        print(f"Cluster submission failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    path_for(PathKind.CLUSTER_JOB_ID, out).write_text(job_id + "\n", encoding="utf-8")
    print(f"Submitted Slurm job {job_id}; the web app will monitor completion.")


if __name__ == "__main__":
    main()
