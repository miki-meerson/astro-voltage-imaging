"""SNAPT propagation maps of every cell of a finished decomposition (the SNAPT step; see utils/snapt_run.py).

    python -m tools.snapt --sta-dir <.../sta-analysis> --decomposition-dir <decomposition run folder> --output <snapt run folder> --source-dir <experiment folder> [--workers N]

It fits every pixel of every cell: run it as a Slurm job (cluster-code/run_snapt.sh), not from a PC.
"""
from __future__ import annotations

import argparse

from utils.snapt_run import snapt_sta_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sta-dir", required=True)
    ap.add_argument("--decomposition-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    print(f"Done. Summary saved to {snapt_sta_run(args.sta_dir, args.decomposition_dir, args.output, args.source_dir, log=lambda m: print(m, flush=True), workers=args.workers)}", flush=True)


if __name__ == "__main__":
    main()
