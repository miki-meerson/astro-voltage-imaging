"""PCA and ICA of the whole-field STA movie of every cell of an STA run that is already done (the decomposition step; see utils/decomposition_run.py).

    python -m tools.sta_decomposition --sta-dir <.../sta-analysis> --output <decomposition run folder> --source-dir <experiment folder>

It opens one result file per cell: run it as a Slurm job (cluster-code/run_decomposition.sh), not from a PC.
"""
from __future__ import annotations

import argparse

from utils.decomposition_run import decompose_sta_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sta-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source-dir", required=True)
    args = ap.parse_args()
    print(f"Done. Summary saved to {decompose_sta_run(args.sta_dir, args.output, args.source_dir, log=lambda m: print(m, flush=True))}", flush=True)


if __name__ == "__main__":
    main()
