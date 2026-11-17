"""The SNAPT step: submits the Slurm job that makes the propagation maps (utils/snapt_run.py) from the newest completed decomposition and the STA run it was made from."""
from __future__ import annotations

from pathlib import Path

from app.pipelines.cluster_step import main
from utils.paths_utils import PathKind, sta_dir_of_run


def job_args(params: dict, out: Path, experiment: str) -> list:
    sta_run, decomposition_run = (Path(params[key]) if params.get(key) else None for key in ("sta_run", "decomposition_run"))
    if decomposition_run is None or not decomposition_run.exists():
        raise FileNotFoundError("SNAPT needs a completed PCA / ICA run (the decomposition step): run it first.")
    if sta_run is None or not sta_dir_of_run(sta_run).exists():
        raise FileNotFoundError("SNAPT needs a completed STA Maps run of the fast hypothesis (with the 'all' result type): run STA Maps for it first.")
    return [sta_dir_of_run(sta_run), decomposition_run, out, experiment]


if __name__ == "__main__":
    main("SNAPT", PathKind.SNAPT_JOB_SCRIPT, job_args)
