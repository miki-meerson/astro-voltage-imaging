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
from utils.delay_distance import DISTANCE_MEASURES, delay_distance_stats, distance_measures, peak_time
from utils.file_utils import read_json
from utils.movie_utils import read_frame_rate
from utils.output_errors import missing_output_message
from utils.paths_utils import PathKind, path_for
from utils.roi_groups import part_of_anatomy_group


def cluster_map(settings, experiment_id: int, step: str) -> dict:
    """The saved summary of the newest completed run of the hypothesis `step` (see utils/cluster_map_summary.py): {run_id, fps, excluded_segments, anatomy_radius_px (the half-size of the window the anatomical configuration looks at around a ROI, None without one), run_folder, result_types, cells: {cell: {roi, motion,
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
    used = read_json(path_for(PathKind.STA_USED_PARAMETERS_JSON, sta_dir), "The STA used-parameters file", missing=missing_output_message("The used-parameters file", run["id"]))
    return dict(hit, run_id=int(run["id"]), fps=read_frame_rate(experiment_repository.experiment_source_path(settings, int(experiment_id))),  # the times of the clusters are frames
                excluded_segments=used.get("excluded_segments", []),  # [start, end) frames left out of the analysis
                anatomy_radius_px=(used.get("anatomy_settings") or {}).get("anatomy_fullres_radius_px"))  # half-size of the square the anatomical configuration looks at around a ROI; None when the run has none


def _cluster_origin(record: dict, cluster: dict) -> tuple[float, float]:
    """The (y, x) centre a cluster's offset is measured from: the centre of the part of a merged ROI that an anatomy cluster was tested around (the anatomy of a merged ROI is done
    around each of its parts), else the centre of the ROI."""
    part = part_of_anatomy_group(cluster["result_type"])
    parts = record.get("part_centers", [])
    return tuple(parts[part - 1]) if part and len(parts) >= part else (float(np.mean(record["roi"]["ys"])), float(np.mean(record["roi"]["xs"])))


def cluster_delays(settings, experiment_id: int, step: str, result_types: list[str] | None = None, clear_peaks_only: bool = True) -> dict:
    """The delay between each cluster's response peak and its neuron's calcium peak, with the distances of the cluster from the neuron (utils/delay_distance.py), for the "Delay and
    distance" panel: {available, run_id, fps, points: [{cell, result_type, sign, p, n_px, delay (seconds, positive = the cluster peaks later), cluster_peak, neuron_peak (seconds
    from the spike), clear (neither peak is on the edge of its window), distances: {measure: pixels or None}}], stats: {measure: {n, rho, p, slope, intercept, slope_p} or None},
    excluded: {motion, no_neuron_trace, window_outside_trace: [cell]}}; each point also has what the page draws to show how it was measured: paths
    {measure: [[y, x], ...] or None} (the points the distance is measured between), the cluster's pixels ys, xs, the frames t with the cluster's trace and the neuron's neuron_trace, and
    the cluster's window (frames); rois: {cell: {ys, xs}} of the points' neurons. `result_types` limits the clusters (all of them when None); `clear_peaks_only` leaves out the points whose peak is on the edge.
    The neuron's peak is the highest point of its spike-triggered dF/F (the spikes are aligned to the neuron's own peak, so it is at the spike, give or take a fraction of a frame); a
    cluster's is the extremum of its trace between its first and last significant frame (the part of that window the trace covers), up for a depolarizing cluster and down for a
    hyperpolarizing one. Motion-suspected and untested cells are left out, as in the offsets. A pooled hypothesis gives {available: False, message}."""
    if hypothesis_for(step).pool_cells:
        return {"available": False, "message": "The delay and distance need the clusters of single cells: a pooled hypothesis has one result for the whole experiment."}
    summary = cluster_map(settings, experiment_id, step)
    points, excluded = [], {"motion": [], "no_neuron_trace": [], "window_outside_trace": []}
    for c in summary["clusters"]:
        record = summary["cells"][str(c["cell"])]
        result_type = g.ANATOMY_GROUP if part_of_anatomy_group(c["result_type"]) else c["result_type"]
        if result_types is not None and result_type not in result_types:
            continue
        if record["motion"] != g.MOTION_OK:
            if c["cell"] not in excluded["motion"]:
                excluded["motion"].append(c["cell"])
            continue
        if not record["neuron_trace"]:
            if c["cell"] not in excluded["no_neuron_trace"]:
                excluded["no_neuron_trace"].append(c["cell"])
            continue
        window = (max(c["first_rel"], c["t"][0]), min(c["last_rel"], c["t"][-1]))  # the trace covers only the frames around the spike: the part of the cluster's window inside it
        if window[0] > window[1]:
            if c["cell"] not in excluded["window_outside_trace"]:
                excluded["window_outside_trace"].append(c["cell"])
            continue
        direction = 1 if polarity.polarity_from_sign(c["sign"]) == polarity.DEPOLARIZING else -1
        cluster_peak, cluster_edge = peak_time(c["trace"], c["t"], window, direction)
        neuron_peak, neuron_edge = peak_time(record["neuron_trace"], c["t"], (c["t"][0], c["t"][-1]), 1)
        distances, paths = distance_measures(record["roi"]["ys"], record["roi"]["xs"], c["ys"], c["xs"], _cluster_origin(record, c), single_shape=len(record.get("part_centers", [])) < 2)
        points.append({
            "cell": c["cell"], "result_type": result_type, "sign": polarity.polarity_from_sign(c["sign"]), "p": c["p"], "n_px": c["n_px"], "delay": (cluster_peak - neuron_peak) / summary["fps"],
            "cluster_peak": cluster_peak / summary["fps"], "neuron_peak": neuron_peak / summary["fps"], "clear": not (cluster_edge or neuron_edge),
            "distances": distances, "paths": paths, "ys": c["ys"], "xs": c["xs"], "t": c["t"], "trace": c["trace"], "neuron_trace": record["neuron_trace"], "window": list(window),
        })
    used = [p for p in points if p["clear"]] if clear_peaks_only else points
    stats = {measure: delay_distance_stats([p["distances"][measure] for p in used if p["distances"][measure] is not None], [p["delay"] for p in used if p["distances"][measure] is not None])
             for measure in DISTANCE_MEASURES}
    rois = {str(cell): summary["cells"][str(cell)]["roi"] for cell in dict.fromkeys(p["cell"] for p in used)}  # the pixels of the points' neurons, for the picture of how a distance is measured
    return {"available": True, "run_id": summary["run_id"], "fps": summary["fps"], "points": used, "rois": rois, "n_unclear": len(points) - len(used), "stats": stats, "excluded": excluded}


def cluster_offsets(settings, experiment_id: int, step: str) -> dict:
    """The clusters of the newest completed run of the hypothesis `step`, of every result type, for the traces and offsets panel: {available, run_id, fps, tested_window (frames),
    result_types, clusters: [{cell, neuron_id, result_type, sign (the polarity), p, mass, first_rel, last_rel, n_px, dx, dy, dist, t, trace}] of the cells that are not motion-suspected
    (dx, dy, dist: from the centre of the cell's ROI to the centre of the cluster, full-resolution pixels), excluded: {motion_flagged, no_motion_diagnostics: [{cell, result_types}]},
    group_test (the real-vs-fake test of the anatomy clusters, when its job has run)}. A pooled hypothesis gives {available: False, message}."""
    if hypothesis_for(step).pool_cells:
        return {"available": False, "message": "The cluster traces and offsets list the clusters of single cells: a pooled hypothesis has one result for the whole experiment."}
    summary = cluster_map(settings, experiment_id, step)
    _, sta_dir = completed_sta_run(settings, experiment_id, step)
    neuron_ids = experiment_repository.neuron_ids_by_cell(settings, int(experiment_id))
    clusters, excluded = [], {"motion_flagged": {}, "no_motion_diagnostics": {}}
    for c in summary["clusters"]:
        motion = summary["cells"][str(c["cell"])]["motion"]
        if motion != g.MOTION_OK:
            excluded["motion_flagged" if motion == g.MOTION_SUSPECTED else "no_motion_diagnostics"].setdefault(c["cell"], set()).add(c["result_type"])
            continue
        part = part_of_anatomy_group(c["result_type"])
        origin = _cluster_origin(summary["cells"][str(c["cell"])], c)
        dy, dx = c["center"][0] - origin[0], c["center"][1] - origin[1]
        clusters.append({
            **{key: c[key] for key in ("cell", "p", "mass", "first_rel", "last_rel", "n_px", "t", "trace")}, "result_type": g.ANATOMY_GROUP if part else c["result_type"],
            "sign": polarity.polarity_from_sign(c["sign"]), "neuron_id": neuron_ids.get(int(c["cell"])), "dx": float(dx), "dy": float(dy), "dist": float(np.hypot(dx, dy)),
        })
    return {
        "available": True, "run_id": summary["run_id"], "fps": summary["fps"], "result_types": list(dict.fromkeys(g.ANATOMY_GROUP if part_of_anatomy_group(name) else name for name in summary["result_types"])), "clusters": clusters,
        "tested_window": read_json(path_for(PathKind.STA_USED_PARAMETERS_JSON, sta_dir), "The STA run parameters")["analysis_window"],
        "excluded": {kind: [{"cell": cell, "result_types": sorted(types)} for cell, types in sorted(cells.items())] for kind, cells in excluded.items()},
        "group_test": read_json(path_for(PathKind.GROUP_TEST, sta_dir), "The real-vs-fake group test", default=None),  # a separate cluster job: shown when it has run
    }
