"""Group test for the anatomy clusters of one STA run (see utils/anatomy_group_test.py for the plain-language description).

    python tools/anatomy_cluster_null.py --sta-dir <.../sta-analysis> --source-dir <experiment folder> [--workers 8] [--draws 3000]

Heavy (it re-runs the cluster search on 200 fakes for every cell): run it as a Slurm job, cluster-code/run_anatomy_group_test.sh.
Writes, inside the STA folder:
  anatomy_group_records.json.gz   per cell: real clusters and the clusters found in each fake (reusable, so new draws need no recompute)
  anatomy_group_test.json         the comparison shown on the experiment page
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]

from utils.anatomy_group_test import cell_records, group_test  # noqa: E402

NPZ = "astro_sta_cluster_permutation_anatomy.npz"


def _one(args):
    cell, npz, roi_center = args
    return cell, cell_records(npz, roi_center)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sta-dir", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--draws", type=int, default=3000)
    ap.add_argument("--reuse", action="store_true", help="only redo the draws from the saved records")
    a = ap.parse_args()
    sta = Path(a.sta_dir)
    rec_path = sta / "anatomy_group_records.json.gz"

    if a.reuse and rec_path.exists():
        with gzip.open(rec_path, "rt", encoding="utf-8") as fh:
            records = {int(k): v for k, v in json.load(fh).items()}
    else:
        s2p = Path(a.source_dir) / "pipeline_results" / "for_suite2p" / "suite2p" / "plane0"
        stat = np.load(s2p / "stat.npy", allow_pickle=True)[np.asarray(np.load(s2p / "iscell.npy", allow_pickle=True)[:, 0], dtype=bool)]
        jobs, skipped = [], {"motion_flagged": 0, "no_motion_diagnostics": 0, "no_anatomy_result": 0}
        for cell_dir in sorted(sta.glob("cell_*")):
            cid = int(cell_dir.name.split("_")[1])
            if not (cell_dir / NPZ).exists() or cid >= len(stat):
                skipped["no_anatomy_result"] += 1
                continue
            m = cell_dir / "motion_sta.npz"
            if not m.exists():
                skipped["no_motion_diagnostics"] += 1
                continue
            with np.load(m, allow_pickle=True) as z:
                if bool(z["flag"]):
                    skipped["motion_flagged"] += 1
                    continue
            roi = stat[cid]
            jobs.append((cid, cell_dir / NPZ, (float(np.mean(roi["ypix"])), float(np.mean(roi["xpix"])))))
        print(f"{len(jobs)} cells go into the test; left out: {skipped}", flush=True)
        records = {}
        with ProcessPoolExecutor(a.workers) as ex:
            for i, (cid, rec) in enumerate(ex.map(_one, jobs), 1):
                records[cid] = rec
                if i % 20 == 0:
                    print(f"  {i}/{len(jobs)} cells", flush=True)
        tmp = rec_path.with_name(rec_path.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(records, fh)
        tmp.replace(rec_path)
        skipped_info = skipped
    result = group_test(records, n_draws=a.draws)
    if not a.reuse:
        result["left_out"] = skipped_info
    out = sta / "anatomy_group_test.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    tmp.replace(out)
    print(json.dumps(result["stats"], indent=1), flush=True)


if __name__ == "__main__":
    main()
