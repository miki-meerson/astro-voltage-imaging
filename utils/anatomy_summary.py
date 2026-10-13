"""Experiment-level summary of the significant anatomy-configuration clusters, computed where the data is created and saved as
<sta run>/sta_algorithm_home/sta-analysis/anatomy_summary.json. The experiment page only reads that file (no Flask, no app imports here).

Written by the STA cluster job (cluster-code/sta_maps_sparse.py, at the end of the STA Maps step).

Selection:
  1. the cell has at least one significant cluster in the anatomy configuration (rows spike_group == 'anatomy' with n_significant_clusters > 0
     in sta_maps_batch_summary.csv),
  2. the cell is not motion-suspected (cell_<N>/motion_sta.npz flag is False). Cells flagged are excluded and listed; cells without motion
     diagnostics are excluded and listed too.
For every significant anatomy cluster of the selected cells: the mean trace (dF/F, -50..+50 frames from the spike; the same trace as on the
cell page) and the offset of the cluster centre from the centre of the cell's ROI, in full-resolution pixels. The cluster sign is stored AS SAVED
('positive' / 'negative', the STA convention of the cluster jobs); the app maps it to depolarizing / hyperpolarizing when it reads
(app/services/polarity.py).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils import globals_utils as g
from utils.cluster_result import cluster_basic_fields, cluster_offset_from_roi, load_cluster_result
from utils.motion_diagnostics import cell_motion_status
from utils.output_errors import missing_output_message
from utils.paths_utils import PathKind, path_for, run_dir_of_sta_dir
from utils.suite2p_utils import roi_center, suite2p_stat_cells



def _significant_anatomy_cells(df: pd.DataFrame) -> tuple[bool, list[int]]:
    """(the run has anatomy rows at all, the cells with at least one significant anatomy cluster)."""
    anatomy = df[df["spike_group"].astype(str) == g.ANATOMY_GROUP]
    significant = anatomy[anatomy["n_significant_clusters"].fillna(0) > 0]
    return not anatomy.empty, sorted(int(c) for c in significant["cell_id"].unique())


def _run_id_of_folder(folder: str) -> int | None:
    match = re.search(r"run_(\d+)$", folder)
    return int(match.group(1)) if match else None


def _cluster_record(cell: int, cluster: dict, mask: np.ndarray, meta: dict, roi_yx: tuple[float, float]) -> dict:
    """One significant cluster: statistics, time span, offset from the ROI centre and its mean dF/F trace."""
    spatial_bin = int(meta["spatial_bin"])
    dy, dx = cluster_offset_from_roi(np.asarray(mask, dtype=bool), meta["anatomy_crop_bounds"], spatial_bin, roi_yx)
    t, trace = cluster["wide_trace"]
    return {
        "cell": cell, **cluster_basic_fields(cluster, meta["tested_window"][0]),
        "n_px": int(np.asarray(mask).sum()), "bin": spatial_bin,
        "dx": float(dx), "dy": float(dy), "dist": float(np.hypot(dx, dy)),
        "t": [int(v) for v in t], "trace": [round(float(v), 6) for v in trace],
    }


def _cell_clusters(sta_dir: Path, cell: int, stat_cell) -> list[dict]:
    """The records of the significant anatomy clusters of one cell (only clusters that have a saved trace)."""
    _, _, significant, meta = load_cluster_result(path_for(PathKind.ANATOMY_RESULT, sta_dir, cell=cell))
    roi_yx = roi_center(stat_cell)
    return [_cluster_record(cell, cluster, mask, meta, roi_yx) for cluster, mask in significant if cluster.get("wide_trace") is not None]


def build_anatomy_summary(sta_dir: str | Path, source_dir: str | Path, run_id: int | None = None, summary_csv: str | Path | None = None, log=None) -> dict:
    """The summary dict (see the module docstring). Reads the cells' anatomy results (~1.5 s per cell over the share).

    sta_dir: <run folder>/sta_algorithm_home/sta-analysis; source_dir: the experiment folder (for the Suite2p ROIs); run_id: the app's run id
    (parsed from the run folder name 'run_<id>' when None); summary_csv: the STA summary csv (default: the one in sta_dir).
    A run without anatomy rows gives an empty summary whose "warning" says so. A missing csv, anatomy result or Suite2p file raises.
    """
    log = log or (lambda m: None)
    sta_dir = Path(sta_dir)
    folder = run_dir_of_sta_dir(sta_dir).name
    summary = {
        "run_folder": folder, "run_id": run_id if run_id is not None else _run_id_of_folder(folder),
        "warning": None, "cells": [], "clusters": [],
        "excluded": {"motion_flagged": [], "no_motion_diagnostics": []},
    }
    table = pd.read_csv(Path(summary_csv) if summary_csv else path_for(PathKind.STA_SUMMARY_CSV, sta_dir))
    has_anatomy, cell_ids = _significant_anatomy_cells(table)
    if not has_anatomy:
        summary["warning"] = "this STA run has no anatomy results (run STA Maps with the 'anatomy' result type)"
    status = cell_motion_status(table)
    summary["excluded"] = {
        "motion_flagged": [c for c in cell_ids if status.get(c) == g.MOTION_SUSPECTED],
        "no_motion_diagnostics": [c for c in cell_ids if status.get(c, g.MOTION_NOT_TESTED) == g.MOTION_NOT_TESTED],
    }
    usable = [c for c in cell_ids if status.get(c) == g.MOTION_OK]
    log(f"anatomy clusters summary: {len(cell_ids)} cells with a significant anatomy cluster, {len(usable)} usable")
    stat_cells = suite2p_stat_cells(source_dir) if usable else []
    for cell in usable:
        clusters = _cell_clusters(sta_dir, cell, stat_cells[cell])
        summary["clusters"] += clusters
        if clusters:
            summary["cells"].append(cell)
        log(f"cell {cell}: {len(clusters)} clusters")
    summary["counts"] = {
        "cells_with_sig_anatomy_cluster": len(cell_ids), "motion_flagged": len(summary["excluded"]["motion_flagged"]),
        "no_motion_diagnostics": len(summary["excluded"]["no_motion_diagnostics"]),
        "cells_shown": len(summary["cells"]), "clusters_shown": len(summary["clusters"]),
    }
    return summary


def save_anatomy_summary(sta_dir: str | Path, source_dir: str | Path, run_id: int | None = None, summary_csv: str | Path | None = None, log=None) -> Path:
    """Build the summary and save it as <sta_dir>/anatomy_summary.json (atomic: temp file, then replace). What the STA job calls."""
    path = path_for(PathKind.ANATOMY_SUMMARY, sta_dir)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(build_anatomy_summary(sta_dir, source_dir, run_id, summary_csv, log)), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_anatomy_summary(sta_dir: str | Path, run_label=None) -> dict:
    """The saved summary of this STA run. Raises FileNotFoundError (saying which step makes it) when it is missing, ValueError when it is
    unreadable or made for another run folder. run_label: how the error names the run (default: its folder name)."""
    path = path_for(PathKind.ANATOMY_SUMMARY, sta_dir)
    label = run_label if run_label is not None else run_dir_of_sta_dir(sta_dir).name
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(missing_output_message("The anatomy clusters summary", label)) from None
    except ValueError as exc:
        raise ValueError(f"The anatomy clusters summary of STA run {label} is unreadable ({exc}): {path}") from exc
    if data["run_folder"] != run_dir_of_sta_dir(sta_dir).name:
        raise ValueError(f"The anatomy clusters summary of STA run {label} was made for another STA run folder: {path}. Rerun STA Maps.")
    return data
