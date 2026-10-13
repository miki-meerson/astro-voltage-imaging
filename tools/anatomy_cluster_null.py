"""Group test for the anatomy clusters of one STA run (see utils/anatomy_group_test.py for the plain-language description).

    python -m tools.anatomy_cluster_null --sta-dir <.../sta-analysis> --source-dir <experiment folder> [--workers 8] [--draws 3000]

Heavy (it re-runs the cluster search on 200 fakes for every cell): run it as a Slurm job, cluster-code/run_anatomy_group_test.sh.
Writes, inside the STA folder:
  the group-test records (GROUP_TEST_RECORDS)  per cell: real clusters and the clusters found in each fake (reusable, so new draws need no recompute)
  the group test (GROUP_TEST)                  the comparison shown on the experiment page
(file names: utils/globals_utils.py)
"""
from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from utils.anatomy_group_test import cell_records, group_test
from utils import globals_utils as g
from utils.motion_diagnostics import cell_motion_status
from utils.paths_utils import PathKind, cell_dirs, path_for
from utils.suite2p_utils import roi_center, suite2p_stat_cells


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
    rec_path = path_for(PathKind.GROUP_TEST_RECORDS, sta)

    if a.reuse and rec_path.exists():
        with gzip.open(rec_path, "rt", encoding="utf-8") as fh:
            records = {int(k): v for k, v in json.load(fh).items()}
    else:
        stat = suite2p_stat_cells(a.source_dir)
        jobs, skipped = [], {"motion_flagged": 0, "no_motion_diagnostics": 0, "no_anatomy_result": 0}
        status = cell_motion_status(pd.read_csv(path_for(PathKind.STA_SUMMARY_CSV, sta)))
        for cid in cell_dirs(sta):
            npz = path_for(PathKind.ANATOMY_RESULT, sta, cell=cid)
            if not npz.exists() or cid >= len(stat):
                skipped["no_anatomy_result"] += 1
                continue
            cell_status = status.get(cid, g.MOTION_NOT_TESTED)
            if cell_status == g.MOTION_NOT_TESTED:
                skipped["no_motion_diagnostics"] += 1
            elif cell_status == g.MOTION_SUSPECTED:
                skipped["motion_flagged"] += 1
            else:
                jobs.append((cid, npz, roi_center(stat[cid])))
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
    out = path_for(PathKind.GROUP_TEST, sta)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    tmp.replace(out)
    print(json.dumps(result["stats"], indent=1), flush=True)


if __name__ == "__main__":
    main()
