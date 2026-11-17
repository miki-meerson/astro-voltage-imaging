"""The decomposition step: submits the Slurm job that makes the PCA and ICA of the whole-field STA movies of the newest completed STA run of the fast hypothesis (utils/decomposition_run.py)."""
from __future__ import annotations

from pathlib import Path

from app.pipelines.cluster_step import main
from utils.paths_utils import PathKind, sta_dir_of_run


def job_args(params: dict, out: Path, experiment: str) -> list:
    sta_run = Path(params["sta_run"]) if params.get("sta_run") else None
    if sta_run is None or not sta_dir_of_run(sta_run).exists():
        raise FileNotFoundError("The decomposition needs a completed STA Maps run of the fast hypothesis (with the 'all' result type): run STA Maps for it first.")
    return [sta_dir_of_run(sta_run), out, experiment]


if __name__ == "__main__":
    main("decomposition", PathKind.DECOMPOSITION_JOB_SCRIPT, job_args)
