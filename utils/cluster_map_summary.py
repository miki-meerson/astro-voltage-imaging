"""Every significant cluster of an STA run, of every result type, with where it is in the field of view and how the signal moves in it: saved as
<sta run>/sta_algorithm_home/sta-analysis/cluster_map_summary.json. The pages that show the clusters only read that file (no Flask, no app imports here).

Written by the STA cluster job (cluster-code/sta_maps_sparse.py, at the end of the STA Maps step) and by `python -m tools.cluster_map_summary` for a run that is already done.
A cell's result file is ~300 MB and the movie is bigger, so this is cluster work: the pages never open those files.

For each cell that has a significant cluster in a result type, and for each such cluster: the sign (as saved: the app maps it to depolarizing / hyperpolarizing when it
reads), p, mass, first and last frame relative to the spike, the cluster's pixels in the full-resolution field of view (the 2D footprint, as the tested movie is binned and may
be cropped), its centre, and the mean dF/F trace over the cluster's footprint (the same trace as on the cell page). Cells are listed with their motion status: the clusters of
motion-suspected cells are kept and flagged, not dropped, so the page lets the user decide. A cell also has its ROI pixels, its spikes and its own spike-triggered mean dF/F
(neuron movie, over the frames of its clusters' traces), from the spikes of its 'all' result when the run has that result type, else of its first.
A cluster also has, from the movie the STA used, the relative change of the mean of its pixels around each individual spike of the cell (`spike_traces`) and the same for the
pixels right next to it, as many as the cluster has (`control`): the moments the mean trace is the average of, and what the neighbouring pixels do at those moments. Both also
have `raw_trace`: the mean intensity of those pixels in every frame of the recording (the app's polarity: depolarization up), to see the moments in the whole recording.

`dynamics` says which clusters have the same time course: the correlation between the traces of every pair of clusters and the hierarchical clustering (average linkage on
1 - correlation) of them. The correlation looks at the shape of the trace only (not its size); a negative cluster against a positive one has a negative correlation, so
depolarizing and hyperpolarizing dynamics do not mix. The page cuts the tree where the user chooses.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import squareform

from utils import globals_utils as g
from utils.cluster_result import cluster_pixels_fullres, cluster_records, load_cluster_result, read_result_header
from utils.file_utils import read_json, write_json
from utils.motion_diagnostics import cell_motion_status
from utils.movie_utils import read_astro_movie
from utils.output_errors import missing_output_message
from utils.paths_utils import PathKind, path_for, result_file_name, run_dir_of_sta_dir
from utils.sta_stats_utils import f0_for_valid_centers, filter_valid_peaks
from utils.sta_trace_utils import cluster_raw_trace_from_movie, event_triggered_mean, lag_corr_matrix
from utils.suite2p_utils import load_suite2p_context, roi_pixels, suite2p_trace_dff

MIN_CLUSTERS_FOR_DYNAMICS = 2


def _cell_clusters(sta_dir: Path, cell: int, result_type: str) -> list[dict]:
    """The records of the significant clusters of one cell in one result type: the shared cluster records (utils/cluster_result.py) with the pixels in the full-resolution field
    of view, the centre and the crop of the result (full-resolution (y0, y1, x0, x1); the STA of an anatomy result normalises by the mean of that region)."""
    result = load_cluster_result(path_for(PathKind.CELL_DIR, sta_dir, cell=cell) / result_file_name(result_type))
    crop, spatial_bin = result[3]["anatomy_crop_bounds"], result[3]["spatial_bin"]
    records = []
    for record in cluster_records(*result):
        ys, xs = cluster_pixels_fullres(record.pop("mask"), crop, spatial_bin)
        records.append({
            "cell": cell, "result_type": result_type, **record, "t": [int(v) for v in record["t"]], "trace": [round(float(v), 6) for v in record["trace"]],
            "crop": None if crop is None else [int(v) * spatial_bin for v in crop],
            "n_px": int(ys.size), "center": [float(ys.mean()), float(xs.mean())], "ys": [int(v) for v in ys], "xs": [int(v) for v in xs],
        })
    return records


def _cell_record(sta_dir: Path, cell: int, t: list[int], spike_source: str, suite2p: dict, motion: str) -> dict:
    """What the page shows of a cell besides its clusters: ROI pixels, motion status, its spikes (the kept spikes of the result type `spike_source`) with the baseline window of
    the result, and the neuron's own spike-triggered mean dF/F at the frames `t` (empty when no spike has the whole window inside the recording)."""
    head = read_result_header(path_for(PathKind.CELL_DIR, sta_dir, cell=cell) / result_file_name(spike_source))
    neuron = event_triggered_mean(suite2p_trace_dff(cell, suite2p_ctx=suite2p), head["peaks"], np.asarray(t, dtype=int))
    ys, xs = roi_pixels(suite2p["stat_cells"][cell])
    return {
        "roi": {"ys": [int(v) for v in ys], "xs": [int(v) for v in xs]}, "motion": motion, "n_spikes": int(head["peaks"].size),
        "spikes": [int(v) for v in head["peaks"]], "baseline_window": head["baseline_window"],
        "neuron_trace": [] if neuron is None else [round(float(v), 6) for v in neuron],
    }


def _group_spike_traces(movie: np.ndarray, f0: np.ndarray, spikes: np.ndarray, rels: np.ndarray, baseline: list[int], ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """(spikes, frames) relative change of the mean of the pixels (ys, xs) around each spike, the quantity the STA averages: the frame over the spike's F0 plus 1 (the movie is in
    the STA's flipped polarity, see read_astro_movie), minus the spike's own baseline-window mean."""
    y0, x0 = int(ys.min()), int(xs.min())
    local = np.zeros((int(ys.max()) - y0 + 1, int(xs.max()) - x0 + 1), dtype=bool)
    local[ys - y0, xs - x0] = True
    in_baseline = (rels >= baseline[0]) & (rels <= baseline[1])
    rows = []
    for spike, spike_f0 in zip(spikes, f0):
        frames = np.asarray(movie[spike + rels[0]: spike + rels[-1] + 1, y0: y0 + local.shape[0], x0: x0 + local.shape[1]], dtype=np.float32)[:, local]
        change = frames.mean(axis=1) / spike_f0 + 1.0
        rows.append(change - change[in_baseline].mean())
    return np.asarray(rows)


def _add_spike_traces(movie: np.ndarray, cell: dict, cluster: dict) -> None:
    """Adds to `cluster` its `spike_traces` (see _group_spike_traces) and a `control`: the same for the pixels right next to the cluster, as many as the cluster has (the ones
    nearest to it outside it, ties broken by the distance to its centre), so the two can be compared moment by moment."""
    ys, xs = np.asarray(cluster["ys"]), np.asarray(cluster["xs"])
    taken = np.zeros(movie.shape[1:], dtype=bool)
    taken[ys, xs] = True
    distance = distance_transform_edt(~taken)
    distance[taken] = np.inf
    yy, xx = np.indices(taken.shape)
    nearest = np.lexsort((np.hypot(yy - cluster["center"][0], xx - cluster["center"][1]).ravel(), distance.ravel()))[: ys.size]
    control_ys, control_xs = np.unravel_index(nearest, taken.shape)
    rels = np.asarray(cluster["t"], dtype=int)
    spikes = filter_valid_peaks(cell["spikes"], movie.shape[0], -int(rels[0]), int(rels[-1]))
    crop = cluster["crop"]
    f0 = f0_for_valid_centers(movie if crop is None else movie[:, crop[0]:crop[1], crop[2]:crop[3]], spikes, cell["baseline_window"])  # normalised by the mean of the STA's own region
    control = np.zeros_like(taken)
    control[control_ys, control_xs] = True
    cluster["spike_traces"] = np.round(_group_spike_traces(movie, f0, spikes, rels, cell["baseline_window"], ys, xs), 5).tolist()
    cluster["raw_trace"] = np.round(cluster_raw_trace_from_movie(movie, taken), 1).tolist()
    cluster["control"] = {
        "n_px": int(ys.size), "center": [float(control_ys.mean()), float(control_xs.mean())],
        "spike_traces": np.round(_group_spike_traces(movie, f0, spikes, rels, cell["baseline_window"], control_ys, control_xs), 5).tolist(),
        "raw_trace": np.round(cluster_raw_trace_from_movie(movie, control), 1).tolist(),
    }


def trace_dynamics(clusters: list[dict]) -> dict | None:
    """{corr: n x n, linkage: scipy linkage matrix} of the traces of `clusters` (in their order), or None when there are fewer than two or their traces do not cover the
    same frames (the clusters of one STA run always do)."""
    if len(clusters) < MIN_CLUSTERS_FOR_DYNAMICS or len({tuple(c["t"]) for c in clusters}) != 1:
        return None
    corr = lag_corr_matrix(np.asarray([c["trace"] for c in clusters], dtype=float), 0)  # a flat trace has no shape to compare: no correlation
    np.fill_diagonal(corr, 1.0)
    distance = np.clip(1 - corr, 0, 2)
    return {"corr": np.round(corr, 4).tolist(), "linkage": linkage(squareform((distance + distance.T) / 2, checks=False), method="average").tolist()}


def build_cluster_map_summary(sta_dir: str | Path, source_dir: str | Path, summary_csv: str | Path | None = None, log=None) -> dict:
    """The summary dict (see the module text). Reads the result files of the cells that have a significant cluster (~1.5 s per file over the share) and the astro movie.
    A run without any significant cluster gives empty lists. A missing csv, result file, movie or Suite2p file raises."""
    log = log or (lambda m: None)
    sta_dir = Path(sta_dir)
    folder = run_dir_of_sta_dir(sta_dir).name
    folder_match = re.search(r"run_(\d+)$", folder)
    table = pd.read_csv(Path(summary_csv) if summary_csv else path_for(PathKind.STA_SUMMARY_CSV, sta_dir))
    status = cell_motion_status(table)
    significant = table[table["n_significant_clusters"].fillna(0) > 0]
    result_types = sorted(table["spike_group"].astype(str).unique())
    suite2p = load_suite2p_context(path_for(PathKind.SUITE2P_PLANE0, source_dir)) if len(significant) else None
    clusters, cells = [], {}
    for result_type, rows in significant.groupby(significant["spike_group"].astype(str)):
        for cell in sorted(int(c) for c in rows["cell_id"].unique()):
            found = _cell_clusters(sta_dir, cell, result_type)
            clusters += found
            if found and cell not in cells:
                cells[cell] = _cell_record(sta_dir, cell, found[0]["t"], g.ALL_GROUP if g.ALL_GROUP in result_types else result_type, suite2p, status.get(cell, g.MOTION_NOT_TESTED))
            log(f"{result_type} cell {cell}: {len(found)} clusters")
    if clusters:  # the movie the STA used, in its flipped polarity, for the traces around each individual spike
        movie = read_astro_movie(str(path_for(PathKind.ASTRO_PB_MOVIE, source_dir)), as_memmap=True)
        for number, cluster in enumerate(clusters):
            _add_spike_traces(movie, cells[cluster["cell"]], cluster)
            log(f"cluster {number + 1} of {len(clusters)}: cell {cluster['cell']} {cluster['result_type']}")
    return {
        "run_folder": folder, "run_id": int(folder_match.group(1)) if folder_match else None, "cells": cells, "clusters": clusters,
        "result_types": result_types, "dynamics": trace_dynamics(clusters),
    }


def save_cluster_map_summary(sta_dir: str | Path, source_dir: str | Path, summary_csv: str | Path | None = None, log=None) -> Path:
    """Build the summary and save it as <sta_dir>/cluster_map_summary.json. What the STA job and tools.cluster_map_summary call."""
    path = path_for(PathKind.CLUSTER_MAP_SUMMARY, sta_dir)
    write_json(path, build_cluster_map_summary(sta_dir, source_dir, summary_csv, log))
    return path


def load_cluster_map_summary(sta_dir: str | Path, run_label=None) -> dict:
    """The saved summary of this STA run. Raises FileNotFoundError (saying which step makes it) when it is missing, ValueError when it was made for another run folder.
    run_label: how the error names the run (default: its folder name)."""
    path = path_for(PathKind.CLUSTER_MAP_SUMMARY, sta_dir)
    label = run_label if run_label is not None else run_dir_of_sta_dir(sta_dir).name
    data = read_json(path, f"The cluster map summary of STA run {label}", missing=missing_output_message("The cluster map summary", label))
    if data["run_folder"] != run_dir_of_sta_dir(sta_dir).name:
        raise ValueError(f"The cluster map summary of STA run {label} was made for another STA run folder: {path}. Rerun STA Maps.")
    return data
