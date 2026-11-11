"""The groups of ROIs with alike traces, for the ROI groups tab of the Preprocessing page (utils/roi_groups.py).

The groups are found on request from the Suite2p fluorescence and saved in one file; the page only reads what is saved: the ROIs with their group, the groups with the correlation of
their members, and (for the group that is open) the members' traces from the Suite2p fluorescence.

Two numberings are used. A `cell` is the position among the accepted ROIs, as everywhere in the app (spikes, analyses, pages). A `roi` is the index in Suite2p's own files (stat, F, ...),
accepted or not. In a group, `cells` are cell numbers and, after the merge, `members` are the ROI indices the merged ROI was made of.

Confirming the groups merges each of them into ONE ROI in the Suite2p result itself, as the Suite2p GUI's merge does (utils/roi_groups.py `merge_groups`): after that every step
(spike detection, the analyses, the pages) sees the merged ROIs as ordinary cells with a cleaner trace. The Suite2p files as they were are kept in a backup folder, and the merge can
be undone. It changes the cell numbers, so it is only possible before Suite2p is confirmed."""
from __future__ import annotations

import shutil
from datetime import datetime

import numpy as np

from app.db import experiment_repository as repo
from app.services.suite2p_data import experiment_rois, experiment_stat_cells, experiment_suite2p_context, forget_suite2p
from utils import roi_groups as rg
from utils.file_utils import read_json, write_json
from utils.movie_utils import read_frame_rate
from utils.paths_utils import PathKind, path_for
from utils.roi_classification import stat_fingerprint
from utils.suite2p_utils import trace_to_dff

TRACE_FILES = {"F": PathKind.SUITE2P_F, "Fneu": PathKind.SUITE2P_FNEU, "spks": PathKind.SUITE2P_SPKS}  # the (ROIs x frames) arrays of Suite2p that are merged with the ROIs
SUITE2P_FILES = {**TRACE_FILES, "stat": PathKind.SUITE2P_STAT, "iscell": PathKind.SUITE2P_ISCELL}  # everything the merge changes


def _merged_record(source: str) -> dict | None:
    return read_json(path_for(PathKind.ROI_GROUPS_MERGED, source), "The merged ROI groups file", default=None)


def _accepted_rois(source: str) -> tuple[np.ndarray, np.ndarray]:
    """(roi_of_cell, stat): the ROI index of every cell, and the stat of every ROI. Raises FileNotFoundError when Suite2p has not run."""
    mask, stat = experiment_rois(source)
    return np.flatnonzero(mask), stat


def _groups_state(source: str) -> dict:
    """{threshold (the saved one, else the default), saved, merged ({confirmed_at} once the groups are merged, else None), warning, groups}: the groups of the experiment as the page shows them.
    Before the merge they are the groups found (saved). After it they are the merged ROIs: each has `cells` [the merged ROI's cell number], `roi` and `members`. Groups saved for other Suite2p
    ROIs are not applied (`warning` says why). Raises FileNotFoundError when Suite2p has not run."""
    roi_of_cell, stat = _accepted_rois(source)
    merged = _merged_record(source)
    if merged is not None:
        cell_of_roi = {int(roi): cell for cell, roi in enumerate(roi_of_cell)}
        groups = [{**group, "cells": [cell_of_roi[group["roi"]]]} for group in merged["groups"] if group["roi"] in cell_of_roi]
        return {"threshold": merged["threshold"], "saved": True, "merged": {"confirmed_at": merged["confirmed_at"]}, "warning": "", "groups": groups}
    try:
        found = rg.load_groups(source, stat_fingerprint(stat))
    except ValueError as exc:  # groups of another Suite2p result: shown as missing, with the reason
        return {"threshold": rg.DEFAULT_THRESHOLD, "saved": False, "merged": None, "warning": str(exc), "groups": []}
    if found is None:
        return {"threshold": rg.DEFAULT_THRESHOLD, "saved": False, "merged": None, "warning": "", "groups": []}
    return {"threshold": found["threshold"], "saved": True, "merged": None, "warning": "", "groups": found["groups"]}


def roi_group_payload(exp: dict) -> dict:
    """What the page loads: `_groups_state` and `rois`: [{roi, cell, ypix, xpix, group (id or None)}] of the accepted ROIs. Raises FileNotFoundError when Suite2p has not run."""
    source = exp["source_path"]
    state = _groups_state(source)
    roi_of_cell, stat = _accepted_rois(source)
    group_of = {cell: group["id"] for group in state["groups"] for cell in group["cells"]}
    rois = [{"roi": int(roi), "cell": cell, "ypix": [int(v) for v in stat[roi]["ypix"]], "xpix": [int(v) for v in stat[roi]["xpix"]], "group": group_of.get(cell)}
            for cell, roi in enumerate(roi_of_cell)]
    return {**state, "rois": rois}


def merged_group_sizes(source: str) -> dict[int, int]:
    """{cell: number of ROIs it was merged from} for the cells of the current Suite2p result that are merged groups (cells that are not groups are absent). Raises FileNotFoundError when
    Suite2p has not run."""
    return {cell: len(stat["merged_from"]) for cell, stat in enumerate(experiment_stat_cells(source)) if "merged_from" in stat}


def find_roi_groups(exp: dict, threshold: float) -> dict:
    """Find the groups of the accepted ROIs whose traces correlate at least `threshold` on average and save them as the experiment's one groups file, replacing the earlier one.
    Returns the payload. Raises FileNotFoundError when Suite2p has not run and ValueError for a threshold outside (0, 1] or when the groups are merged already."""
    source = exp["source_path"]
    if _merged_record(source) is not None:
        raise ValueError("The groups are merged already: undo the merge to find them again.")
    _, stat = _accepted_rois(source)
    groups = rg.find_groups(experiment_suite2p_context(source)["F"], threshold)
    rg.save_groups(source, stat_fingerprint(stat), threshold, groups)
    return roi_group_payload(exp)


def _check_not_confirmed(exp: dict, what: str) -> None:
    if exp["workflow_stage"] == repo.STAGE_ANALYSIS_READY:
        raise ValueError(f"Suite2p was already confirmed for this experiment, and {what} changes the cell numbers. Unconfirm Suite2p first: the runs made with the old cells are then cancelled.")


def _restore_suite2p(source: str) -> None:
    """Put the Suite2p files back from the backup folder, remove the folder and the record of the merge (if it was written), and drop what the app kept of the experiment."""
    backup, plane0 = path_for(PathKind.SUITE2P_MERGE_BACKUP, source), path_for(PathKind.SUITE2P_PLANE0, source)
    for kind in SUITE2P_FILES.values():
        shutil.copy2(path_for(kind, backup), path_for(kind, plane0))
    shutil.rmtree(backup)
    path_for(PathKind.ROI_GROUPS_MERGED, source).unlink(missing_ok=True)
    forget_suite2p(source)


def _write_merge(source: str, stat: np.ndarray, group_rois: list[list[int]], record: dict) -> None:
    """Merge the groups (`group_rois`: the ROI indices of each) in the Suite2p files of the experiment and write the record of the merge, keeping the files as they were in the backup
    folder. The merge is made in memory first. If anything fails while the files are written, they are put back and the backup folder is removed, so the merge can be tried again."""
    plane0, backup = path_for(PathKind.SUITE2P_PLANE0, source), path_for(PathKind.SUITE2P_MERGE_BACKUP, source)
    traces = {name: np.load(path_for(kind, plane0)) for name, kind in TRACE_FILES.items()}
    new_stat, new_traces, new_iscell = rg.merge_groups(stat, traces, np.load(path_for(PathKind.SUITE2P_ISCELL, plane0)), group_rois)
    backup.mkdir(parents=True)
    try:
        for kind in SUITE2P_FILES.values():
            shutil.copy2(path_for(kind, plane0), path_for(kind, backup))
    except OSError:
        shutil.rmtree(backup)
        raise
    try:
        for name, kind in TRACE_FILES.items():
            np.save(path_for(kind, plane0), new_traces[name])
        np.save(path_for(PathKind.SUITE2P_STAT, plane0), new_stat, allow_pickle=True)
        np.save(path_for(PathKind.SUITE2P_ISCELL, plane0), new_iscell)
        write_json(path_for(PathKind.ROI_GROUPS_MERGED, source), record)
    except Exception:
        _restore_suite2p(source)
        raise


def confirm_roi_groups(exp: dict) -> dict:
    """Merge every saved group of ROIs into one ROI in the Suite2p result (F, Fneu, spks, stat and iscell in the plane0 folder), keeping the files as they were in a backup folder, and
    save which ROIs were merged. Returns the payload. Raises ValueError when Suite2p is confirmed already, the groups are merged already or there are none, and FileNotFoundError when
    they were not found yet."""
    source = exp["source_path"]
    _check_not_confirmed(exp, "merging ROIs")
    state = _groups_state(source)
    if state["merged"] is not None:
        raise ValueError("The groups are merged already.")
    if not state["saved"]:
        raise FileNotFoundError(state["warning"] or "No ROI groups are saved for this experiment: find them first.")
    if not state["groups"]:
        raise ValueError(f"There are no groups to merge: no ROIs correlate at least {state['threshold']}. Find the groups again with a lower correlation.")
    backup = path_for(PathKind.SUITE2P_MERGE_BACKUP, source)
    if backup.exists():
        raise ValueError(f"The backup folder {backup} exists already: it holds the Suite2p result from before an earlier merge. Move it away first.")
    roi_of_cell, stat = _accepted_rois(source)
    group_rois = [[int(roi_of_cell[cell]) for cell in group["cells"]] for group in state["groups"]]
    groups = [{**{key: value for key, value in group.items() if key != "cells"}, "roi": len(stat) + k, "members": rois, "cells_before": group["cells"]}  # the merged ROIs follow the last ROI
              for k, (group, rois) in enumerate(zip(state["groups"], group_rois))]
    _write_merge(source, stat, group_rois, {"threshold": state["threshold"], "confirmed_at": datetime.now().isoformat(timespec="seconds"), "groups": groups})
    forget_suite2p(source)
    return roi_group_payload(exp)


def undo_roi_groups(exp: dict) -> dict:
    """Put the Suite2p result back as it was before the groups were merged and forget the merge (the groups found stay saved). Returns the payload. Raises ValueError when Suite2p is
    confirmed or nothing is merged, and FileNotFoundError when the backup folder is gone."""
    source = exp["source_path"]
    _check_not_confirmed(exp, "undoing the merge")
    if _merged_record(source) is None:
        raise ValueError("The groups of this experiment are not merged.")
    backup = path_for(PathKind.SUITE2P_MERGE_BACKUP, source)
    if not backup.exists():
        raise FileNotFoundError(f"The backup of the Suite2p result from before the merge is missing ({backup}): it cannot be undone. Run Suite2p again.")
    _restore_suite2p(source)
    return roi_group_payload(exp)


def group_traces(exp: dict, group_id: int) -> dict:
    """{group, fps, cells: [{label, roi, trace, merged}]}: the dF/F (percent) over the whole recording of every member of a saved group, or, after the merge, of the merged ROI (merged)
    and of the ROIs it was made of (from the backup). Raises FileNotFoundError when no groups are saved and ValueError for a group that does not exist."""
    source = exp["source_path"]
    state = _groups_state(source)
    if not state["saved"]:
        raise FileNotFoundError("No ROI groups are saved for this experiment: find the groups first.")
    group = next((g for g in state["groups"] if g["id"] == group_id), None)
    if group is None:
        raise ValueError(f"There is no group {group_id}: the saved groups are 0 to {len(state['groups']) - 1}.")
    roi_of_cell, _ = _accepted_rois(source)
    fluorescence = experiment_suite2p_context(source)["F"]  # one row per cell

    def entry(label: str, roi: int, raw_trace: np.ndarray, merged: bool) -> dict:
        return {"label": label, "roi": roi, "merged": merged, "trace": np.round(trace_to_dff(raw_trace) * 100, 3).tolist()}

    if state["merged"] is None:
        cells = [entry(f"cell {cell}", int(roi_of_cell[cell]), fluorescence[cell], False) for cell in group["cells"]]
    else:
        cell = group["cells"][0]
        before = np.load(path_for(PathKind.SUITE2P_F, path_for(PathKind.SUITE2P_MERGE_BACKUP, source)), mmap_mode="r")  # the traces of the ROIs it was made of, by ROI index
        cells = [entry(f"merged (cell {cell})", int(roi_of_cell[cell]), fluorescence[cell], True)]
        cells += [entry(f"ROI {roi}", roi, before[roi], False) for roi in group["members"]]
    return {"group": group_id, "fps": read_frame_rate(source), "cells": cells}
