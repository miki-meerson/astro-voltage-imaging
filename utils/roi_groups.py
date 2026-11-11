"""Groups of Suite2p ROIs whose traces are alike: ROIs that may be parts of one cell (the branches of a dendrite that Suite2p split into several ROIs), to be looked at and
analysed together.

The groups come from the Suite2p fluorescence of the accepted ROIs (iscell = 1): the correlation between the traces of every pair, then a hierarchical clustering (average
linkage on 1 - correlation, utils/sta_trace_utils.py `correlation_linkage`) cut where the average correlation inside a group falls below the threshold. So the members of a group
correlate at least `threshold` on average, which is stricter for a big group than a chain of pairs would be. Only groups of two or more ROIs are kept; a ROI that is alone has no group.
Nothing about the spikes or the astrocyte movie is used, so a grouping cannot depend on the result it will be used for.

Everything is saved in ONE file per experiment (roi_groups.json, pipeline_results/roi_groups/), by cell number (the position among the accepted ROIs: the numbering of the spikes
and the analyses). A file made for other Suite2p ROIs than the current ones (Suite2p was run again) is not applied.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster

from utils import globals_utils as g
from utils.file_utils import read_json, write_json
from utils.paths_utils import PathKind, path_for
from utils.sta_trace_utils import correlation_linkage

DEFAULT_THRESHOLD = 0.8  # the average correlation between the members of a group, at least


def find_groups(traces: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """The groups of ROIs (rows of `traces`, one per cell) whose traces correlate at least `threshold` on average, biggest first: [{id, cells, size, min_corr, mean_corr, corr}] with
    `corr` the members' correlation matrix (rounded). Raises ValueError for a threshold outside (0, 1]."""
    if not 0 < threshold <= 1:
        raise ValueError(f"The correlation threshold must be above 0 and at most 1, not {threshold!r}.")
    if len(traces) < 2:
        return []
    corr, tree = correlation_linkage(traces)
    labels = fcluster(tree, 1 - threshold, criterion="distance")
    members = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    groups = []
    for cells in sorted((m for m in members if len(m) > 1), key=lambda m: (-len(m), m[0])):
        sub = corr[np.ix_(cells, cells)]
        off_diagonal = sub[~np.eye(len(cells), dtype=bool)]
        groups.append({"id": len(groups), "cells": [int(c) for c in cells], "size": int(len(cells)), "min_corr": round(float(off_diagonal.min()), 3),
                       "mean_corr": round(float(off_diagonal.mean()), 3), "corr": np.round(sub, 3).tolist()})
    return groups


def load_groups(experiment: str | Path, fingerprint: str) -> dict | None:
    """The saved {threshold, groups}, or None when there is none. Raises ValueError when it was made for other Suite2p ROIs than the current ones."""
    saved = read_json(path_for(PathKind.ROI_GROUPS, experiment), "The ROI groups file", default=None)
    if saved is None:
        return None
    if saved["fingerprint"] != fingerprint:
        raise ValueError("The ROI groups were made for other Suite2p ROIs than the current ones (Suite2p was run again). Find the groups again: it makes the file anew.")
    return {"threshold": saved["threshold"], "groups": saved["groups"]}


def save_groups(experiment: str | Path, fingerprint: str, threshold: float, groups: list[dict]) -> None:
    """Save the groups of an experiment (its one file), replacing the earlier one."""
    write_json(path_for(PathKind.ROI_GROUPS, experiment), {"fingerprint": fingerprint, "threshold": threshold, "groups": groups})


def anatomy_group_of_part(part: int) -> str:
    """The result type of the anatomy around part `part` (1, 2, ...) of a merged ROI: "anatomy" for the first (the largest) part, "anatomy_part2", ... for the others."""
    return g.ANATOMY_GROUP if int(part) == 1 else g.ANATOMY_PART_TEMPLATE.format(part=int(part))


def part_of_anatomy_group(result_type: str) -> int | None:
    """The part (1, 2, ...) an anatomy result type is about, None for a result type that is not an anatomy one."""
    if result_type == g.ANATOMY_GROUP:
        return 1
    prefix, _, number = g.ANATOMY_PART_TEMPLATE.partition("{part}")
    return int(result_type[len(prefix):]) if result_type.startswith(prefix) and result_type[len(prefix):].isdigit() else None


def part_centers(stat: dict) -> list[tuple[float, float]]:
    """The centres (y, x) of the parts of a ROI of Suite2p's stat that the anatomy is done around: a merged ROI has one per ROI it was made of, the largest first; any other ROI has
    its own centre."""
    if "part_centers" in stat:
        return [(float(y), float(x)) for y, x in np.asarray(stat["part_centers"])]
    return [(float(np.mean(stat["ypix"])), float(np.mean(stat["xpix"])))]


def merged_roi_stat(members: list[dict]) -> dict:
    """The stat of the ROI that is the union of the `members` (their stats). The pixels are the union (a pixel in several members has the sum of their weights); the numbers about
    the shape that Suite2p saved (compactness, skew, ...) and the centre `med` are the largest member's, so the merged ROI is centred on its largest part. `part_centers` are the centres of all the members (the
    largest first): the anatomy is done around each of them. The arrays that follow the pixels are made again."""
    largest = max(members, key=lambda stat: len(stat["ypix"]))
    weights: dict[tuple[int, int], float] = {}
    for stat in members:
        for y, x, lam in zip(stat["ypix"], stat["xpix"], stat["lam"]):
            weights[(int(y), int(x))] = weights.get((int(y), int(x)), 0.0) + float(lam)
    pixels = sorted(weights)
    merged = dict(largest)
    ordered = sorted(members, key=lambda stat: -len(stat["ypix"]))
    merged.update(part_centers=np.array([[float(np.mean(stat["ypix"])), float(np.mean(stat["xpix"]))] for stat in ordered]),
                  ypix=np.array([y for y, _ in pixels], dtype=np.int32), xpix=np.array([x for _, x in pixels], dtype=np.int32), lam=np.array([weights[p] for p in pixels], dtype=np.float32),
                  npix=len(pixels), overlap=np.zeros(len(pixels), dtype=bool), soma_crop=np.ones(len(pixels), dtype=bool))
    return merged


def merge_groups(stat: np.ndarray, traces: dict[str, np.ndarray], iscell: np.ndarray, groups: list[list[int]]) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Suite2p's result with every group of ROIs (`groups`: lists of ORIGINAL ROI indices) merged into one ROI, as the Suite2p GUI's merge does: the merged ROI is added after the
    last ROI, its members stop being cells (iscell = 0), its traces are the members' traces weighted by their pixel counts (the mean intensity over the union of the pixels; the noise
    of each member is averaged out) and its stat is `merged_roi_stat` with `merged_from` (the members' ROI indices). `traces`: {name: (n ROIs, frames)}, merged the same way
    (the spikes (spks) are averaged too, not deconvolved again). Returns (stat, traces, iscell) with the merged ROIs appended in the order of `groups`."""
    new_stat, rows, new_iscell = list(stat), {name: [] for name in traces}, np.array(iscell, dtype=np.float64)
    appended = []
    for members in groups:
        weights = np.array([len(stat[i]["ypix"]) for i in members], dtype=np.float64)
        merged = merged_roi_stat([stat[i] for i in members])
        merged["merged_from"] = np.array(sorted(members), dtype=np.int32)
        new_stat.append(merged)
        for name, array in traces.items():
            rows[name].append(np.average(array[members].astype(np.float64), axis=0, weights=weights).astype(array.dtype))
        new_iscell[members, 0] = 0.0
        appended.append([1.0, 1.0])
    stat_array = np.empty(len(new_stat), dtype=object)
    stat_array[:] = new_stat
    merged_traces = {name: np.vstack([traces[name], np.array(rows[name])]) if rows[name] else traces[name] for name in traces}
    return stat_array, merged_traces, np.vstack([new_iscell, np.array(appended)]) if appended else new_iscell


def fold_anatomy_parts(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of an STA summary table with the anatomy result of every part of a merged ROI counted as the cell's one anatomy result: the rows of the parts after the first are
    dropped and their clusters (and significant clusters) are added to the row of part 1. Each part was tested at the run's cluster alpha, so the cell is
    anatomy-significant when any part is (more likely by chance the more parts it has: the group test accounts for that). A table without parts is returned as it is."""
    if "anatomy_part" not in table:
        return table
    later = table["anatomy_part"].fillna(1) > 1
    if not later.any():
        return table
    extra = table[later].groupby("cell_id")[["n_clusters", "n_significant_clusters"]].sum()
    kept = table[~later].copy()
    first = (kept["spike_group"].astype(str) == g.ANATOMY_GROUP) & kept["cell_id"].isin(extra.index)
    for column in extra:
        kept.loc[first, column] = kept.loc[first, column].fillna(0).to_numpy() + extra.loc[kept.loc[first, "cell_id"], column].to_numpy()
    return kept
