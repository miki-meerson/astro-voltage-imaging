"""What the saved results of a hypothesis say, for the experiment pages: one summary per hypothesis (settings, the status of its newest run, and
the numbers of its newest completed run: cells tested, spikes used, cells with a significant cluster, how many are expected by chance), the flags
of every cell, and the anatomy clusters of the experiment. Reads saved files only."""
from __future__ import annotations

import json

import pandas as pd

from app.db import run_repository
from app.services import caches
from app.services.runs import default_sta_step, feeding_steps, hypothesis_for, run_dir, sta_steps
from utils import globals_utils as g
from utils.file_utils import read_json
from utils.hypotheses import hypothesis_name, sta_step
from utils.output_errors import missing_output_message
from utils.sta_stats_utils import significant_fraction
from utils.paths_utils import PathKind, path_for, sta_dir_of_run


def completed_sta_run(settings, experiment_id: int, step: str) -> tuple[dict, str]:
    """The newest completed run of an STA step and the folder of its results (<run>/sta_algorithm_home/sta-analysis). Raises when there is none."""
    run = run_repository.latest_run(settings, int(experiment_id), step)
    if run is None:
        raise FileNotFoundError(f"No completed run of hypothesis '{hypothesis_name(step)}' for this experiment: run the analysis for it first.")
    return run, sta_dir_of_run(run_dir(settings, int(run["id"]), bool(run["is_test"])))


def tested_cells(rows: pd.DataFrame, used: dict) -> pd.DataFrame:
    """The rows of the cells that were tested: not skipped, with at least the minimum number of spikes."""
    return rows[(rows["n_peaks"] >= int(used["min_spikes_for_cluster"])) & (rows["status"].astype(str) != "skipped")]


def _result_type_summary(rows: pd.DataFrame, used: dict) -> dict:
    """The numbers of one result type: `rows` = the summary csv rows of that spike group (one per cell), `used` = the parameters the job used."""
    min_spikes = int(used["min_spikes_for_cluster"])
    alpha = float(used["cluster_alpha"])
    skipped = rows["status"].astype(str) == "skipped"
    enough = rows["n_peaks"] >= min_spikes
    tested = tested_cells(rows, used)
    n_sig = int((tested["n_significant_clusters"].fillna(0) > 0).sum())
    result = {
        "cells_in_run": int(len(rows)), "cells_tested": int(len(tested)), "cells_few_spikes": int((~enough).sum()), "cells_skipped": int(skipped.sum()),
        "spikes_median": float(tested["n_peaks"].median()) if len(tested) else None,
        "spikes_removed_by_silence": int((rows["n_peaks_before_silence"] - rows["n_peaks"]).sum()),
        "cells_significant": n_sig, "cells_expected_by_chance": len(tested) * alpha,
        "p_versus_chance": significant_fraction(n_sig, len(tested), alpha)["p_vs_chance"],
    }
    if g.MOTION_STATUS_COLUMN in tested:
        result["cells_significant_motion_suspected"] = int(((tested["n_significant_clusters"].fillna(0) > 0) & (tested[g.MOTION_STATUS_COLUMN] == g.MOTION_SUSPECTED)).sum())
    return result


def _pooled_summary(rows: pd.DataFrame) -> dict:
    """The one test of a pooled hypothesis (all cells' spikes as one list of events): {pooled, spikes, clusters, significant_clusters, motion, motion_p}."""
    row = rows.iloc[0]
    return {
        "pooled": True, "spikes": int(row["n_peaks"]), "clusters": int(row["n_clusters"]), "significant_clusters": int(row["n_significant_clusters"]),
        "motion": str(row[g.MOTION_STATUS_COLUMN]) if g.MOTION_STATUS_COLUMN in rows else g.MOTION_NOT_TESTED,
        "motion_p": float(row["motion_p"]) if "motion_p" in rows and pd.notna(row["motion_p"]) else None,
    }


def hypothesis_results(settings, experiment_id: int, step: str) -> dict:
    """The numbers of the newest completed run of one hypothesis: {run_id, result_types: {name: numbers}, group_test (when the group test job has run), excluded_segments (the
    [start, end) frames left out of the analysis when it ran; [] for a run made before that could be set)}."""
    run, sta_dir = completed_sta_run(settings, experiment_id, step)
    table = pd.read_csv(path_for(PathKind.STA_SUMMARY_CSV, sta_dir))
    used = read_json(path_for(PathKind.STA_USED_PARAMETERS_JSON, sta_dir), "The STA used-parameters file", missing=missing_output_message("The used-parameters file", run["id"]))  # what the job really used
    pooled = hypothesis_for(step).pool_cells
    results = {str(name): _pooled_summary(rows) if pooled else _result_type_summary(rows, used) for name, rows in table.groupby(table["spike_group"].astype(str))}
    group_test = read_json(path_for(PathKind.GROUP_TEST, sta_dir), "The real-vs-fake group test", default=None)
    return {"run_id": int(run["id"]), "result_types": results, "group_test": group_test, "excluded_segments": used.get("excluded_segments", [])}


def waiting_for(settings, experiment_id: int, run: dict) -> list[str]:
    """The feeding steps a waiting run still waits for."""
    needed = feeding_steps(json.loads(run["parameters_json"])["result_types"])
    return sorted(needed - run_repository.steps_with_status(settings, experiment_id, ("completed",)))


def hypothesis_status(settings, experiment_id: int, step: str) -> dict:
    """One hypothesis without its results: name, description, settings, the status of its newest run (or 'not_run'), the run id and, for a
    waiting run, the feeding steps it still waits for."""
    hypothesis = hypothesis_for(step)
    newest = run_repository.latest_run(settings, int(experiment_id), step, status=None)
    waits_for = waiting_for(settings, experiment_id, newest) if newest and newest["status"] == "waiting" else []
    return {"name": hypothesis.name, "step": step, "description": hypothesis.description, "settings": hypothesis.settings(), "pool_cells": hypothesis.pool_cells,
            "status": newest["status"] if newest else "not_run", "run_id": int(newest["id"]) if newest else None, "waiting_for": waits_for}


def hypothesis_summary(settings, experiment_id: int, step: str) -> dict:
    """hypothesis_status plus, when a run completed, its results. A rerun in progress shows its own status next to the results of the earlier run."""
    summary = {**hypothesis_status(settings, experiment_id, step), "results": None}
    if run_repository.latest_run(settings, int(experiment_id), step) is not None:  # some run of this hypothesis completed
        try:
            summary["results"] = hypothesis_results(settings, experiment_id, step)
        except FileNotFoundError:
            summary["results_warning"] = missing_output_message("The summary table", summary["run_id"])
    return summary


def hypothesis_summaries(settings, experiment_id: int) -> list[dict]:
    """The summary of every hypothesis of the catalogue, in catalogue order."""
    return [hypothesis_summary(settings, experiment_id, step) for step in sta_steps()]


def chosen_step(name: str | None) -> str:
    """The STA step of the hypothesis named in the address (?h=): the default hypothesis when none is named. Raises for a name not in the catalogue."""
    if not name:
        return default_sta_step()
    step = sta_step(name)
    if step not in sta_steps():
        raise ValueError(f"Unknown hypothesis '{name}'. The hypotheses are: {', '.join(hypothesis_name(s) for s in sta_steps())}.")
    return step


def cell_table(settings, experiment_id: int, step: str) -> dict:
    """The flags of every cell of the newest completed run of a hypothesis, from its summary csv: {run_id, cells: {cell: {spikes, spikes_before (before the
    silence rule), clusters, significant, anatomy, motion}}}. clusters: the significant clusters over all result types; significant: a significant cluster in any result type; anatomy: in the anatomy result type. Raises FileNotFoundError when the
    hypothesis has no completed run."""
    run, sta_dir = completed_sta_run(settings, experiment_id, step)
    csv = path_for(PathKind.STA_SUMMARY_CSV, sta_dir)
    stat = csv.stat()
    key = (int(run["id"]), int(stat.st_mtime_ns), int(stat.st_size))
    cached = caches.STA_CELL_TABLE.get(key)
    if cached is not None:
        return cached
    table = pd.read_csv(csv)
    table["significant"] = table["n_significant_clusters"].fillna(0) > 0
    cells = {}
    for cell, rows in table.groupby("cell_id"):
        cells[int(cell)] = {
            "spikes": int(rows["n_peaks"].max()), "spikes_before": int(rows["n_peaks_before_silence"].max()), "clusters": int(rows["n_significant_clusters"].fillna(0).sum()), "significant": bool(rows["significant"].any()),
            "anatomy": bool(rows.loc[rows["spike_group"].astype(str) == g.ANATOMY_GROUP, "significant"].any()),
            "motion": str(rows[g.MOTION_STATUS_COLUMN].iloc[0]) if g.MOTION_STATUS_COLUMN in rows else g.MOTION_NOT_TESTED,
        }
    result = {"run_id": int(run["id"]), "cells": cells}
    caches.STA_CELL_TABLE.put(key, result)
    return result
