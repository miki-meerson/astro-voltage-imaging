"""The cluster map of an experiment for the Hypotheses tab: every cell with a significant cluster, with the clusters in the field of view and their traces, read from the
cluster_map_summary.json of the newest completed run of a hypothesis (written by the STA job, utils/cluster_map_summary.py)."""
from __future__ import annotations

import numpy as np

from app.db import experiment_repository
from app.services import caches, polarity
from app.services.hypothesis_summary import completed_sta_run
from app.services.runs import hypothesis_for
from utils import globals_utils as g
from utils.cluster_map_summary import load_cluster_map_summary
from utils.file_utils import read_json
from utils.movie_utils import read_frame_rate
from utils.paths_utils import PathKind, path_for


def cluster_map(settings, experiment_id: int, step: str) -> dict:
    """The saved summary of the newest completed run of the hypothesis `step` (see utils/cluster_map_summary.py): {run_id, fps, run_folder, result_types, cells: {cell: {roi, motion,
    n_spikes, neuron_trace}}, clusters: [{cell, result_type, sign, p, mass, first_rel, last_rel, n_px, center, ys, xs, t, trace}], dynamics}. Raises FileNotFoundError saying which
    step makes the file when the run has none, and ValueError for a pooled hypothesis (one result for the whole experiment, no cells to map)."""
    if hypothesis_for(step).pool_cells:
        raise ValueError("A pooled hypothesis has one result for the whole experiment: there are no cells to map.")
    run, sta_dir = completed_sta_run(settings, experiment_id, step)
    path = path_for(PathKind.CLUSTER_MAP_SUMMARY, sta_dir)
    key = (str(path), path.stat().st_mtime_ns if path.exists() else None)
    hit = caches.CLUSTER_MAP.get(key)
    if hit is None:
        hit = load_cluster_map_summary(sta_dir, run_label=int(run["id"]))  # a failed read is not cached
        caches.CLUSTER_MAP.put(key, hit)
    return dict(hit, run_id=int(run["id"]), fps=read_frame_rate(experiment_repository.experiment_source_path(settings, int(experiment_id))))  # the times of the clusters are frames


def cluster_offsets(settings, experiment_id: int, step: str) -> dict:
    """The clusters of the newest completed run of the hypothesis `step`, of every result type, for the traces and offsets panel: {available, run_id, fps, tested_window (frames),
    result_types, clusters: [{cell, neuron_id, result_type, sign (the polarity), p, mass, first_rel, last_rel, n_px, dx, dy, dist, t, trace}] of the cells that are not motion-suspected
    (dx, dy, dist: from the centre of the cell's ROI to the centre of the cluster, full-resolution pixels), excluded: {motion_flagged, no_motion_diagnostics: [{cell, result_types}]},
    group_test (the real-vs-fake test of the anatomy clusters, when its job has run)}. A pooled hypothesis gives {available: False, message}."""
    if hypothesis_for(step).pool_cells:
        return {"available": False, "message": "The cluster traces and offsets list the clusters of single cells: a pooled hypothesis has one result for the whole experiment."}
    summary = cluster_map(settings, experiment_id, step)
    _, sta_dir = completed_sta_run(settings, experiment_id, step)
    centers = {int(cell): (float(np.mean(record["roi"]["ys"])), float(np.mean(record["roi"]["xs"]))) for cell, record in summary["cells"].items()}
    neuron_ids = experiment_repository.neuron_ids_by_cell(settings, int(experiment_id))
    clusters, excluded = [], {"motion_flagged": {}, "no_motion_diagnostics": {}}
    for c in summary["clusters"]:
        motion = summary["cells"][str(c["cell"])]["motion"]
        if motion != g.MOTION_OK:
            excluded["motion_flagged" if motion == g.MOTION_SUSPECTED else "no_motion_diagnostics"].setdefault(c["cell"], set()).add(c["result_type"])
            continue
        dy, dx = c["center"][0] - centers[c["cell"]][0], c["center"][1] - centers[c["cell"]][1]
        clusters.append({
            **{key: c[key] for key in ("cell", "result_type", "p", "mass", "first_rel", "last_rel", "n_px", "t", "trace")},
            "sign": polarity.polarity_from_sign(c["sign"]), "neuron_id": neuron_ids.get(int(c["cell"])), "dx": float(dx), "dy": float(dy), "dist": float(np.hypot(dx, dy)),
        })
    return {
        "available": True, "run_id": summary["run_id"], "fps": summary["fps"], "result_types": summary["result_types"], "clusters": clusters,
        "tested_window": read_json(path_for(PathKind.STA_USED_PARAMETERS_JSON, sta_dir), "The STA run parameters")["analysis_window"],
        "excluded": {kind: [{"cell": cell, "result_types": sorted(types)} for cell, types in sorted(cells.items())] for kind, cells in excluded.items()},
        "group_test": read_json(path_for(PathKind.GROUP_TEST, sta_dir), "The real-vs-fake group test", default=None),  # a separate cluster job: shown when it has run
    }
