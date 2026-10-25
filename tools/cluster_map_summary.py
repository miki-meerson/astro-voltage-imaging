"""Make the cluster map summary of an STA run that is already done (a new STA Maps run makes it by itself; see utils/cluster_map_summary.py).

    python -m tools.cluster_map_summary --sta-dir <.../sta-analysis> --source-dir <experiment folder>

It opens one ~300 MB result file per cell with a significant cluster: run it as a Slurm job (cluster-code/run_cluster_map_summary.sh), not from a PC.
"""
from __future__ import annotations

import argparse

from utils.cluster_map_summary import save_cluster_map_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sta-dir", required=True)
    ap.add_argument("--source-dir", required=True)
    args = ap.parse_args()
    print(f"Saved {save_cluster_map_summary(args.sta_dir, args.source_dir, log=lambda m: print(m, flush=True))}")


if __name__ == "__main__":
    main()
