"""One summary per hypothesis of an experiment: its settings, the status of its newest run and, when a run completed, the numbers read from that
run's saved summary csv (cells tested, spikes used, cells with a significant cluster, how many are expected by chance). Reads saved files only."""
from __future__ import annotations

import json

import pandas as pd
from scipy.stats import binomtest

from app.db import run_repository
from app.services.runs import feeding_steps, hypothesis_for, run_dir, sta_steps
from utils import globals_utils as g
from utils.file_utils import read_json
from utils.hypotheses import hypothesis_name
from utils.output_errors import missing_output_message
from utils.paths_utils import PathKind, path_for, sta_dir_of_run


def completed_sta_run(settings, experiment_id: int, step: str) -> tuple[dict, str]:
    """The newest completed run of an STA step and the folder of its results (<run>/sta_algorithm_home/sta-analysis). Raises when there is none."""
    run = run_repository.latest_run(settings, int(experiment_id), step)
    if run is None:
        raise FileNotFoundError(f"No completed run of {step} for this experiment: run the analysis for hypothesis '{hypothesis_name(step)}' first.")
    return run, sta_dir_of_run(run_dir(settings, int(run["id"]), bool(run["is_test"])))


def _result_type_summary(rows: pd.DataFrame, params: dict) -> dict:
    """The numbers of one result type: `rows` = the summary csv rows of that spike group (one per cell)."""
    min_spikes = int(params["min_spikes_for_cluster"])
    alpha = float(params["cluster_alpha"])
    skipped = rows["status"].astype(str) == "skipped"
    enough = rows["n_peaks"] >= min_spikes
    tested = rows[enough & ~skipped]
    n_sig = int((tested["n_significant_clusters"].fillna(0) > 0).sum())
    result = {
        "cells_in_run": int(len(rows)), "cells_tested": int(len(tested)), "cells_few_spikes": int((~enough).sum()), "cells_skipped": int(skipped.sum()),
        "spikes_median": float(tested["n_peaks"].median()) if len(tested) else None,
        "spikes_removed_by_silence": int((rows["n_peaks_before_silence"] - rows["n_peaks"]).sum()),
        "cells_significant": n_sig, "cells_expected_by_chance": len(tested) * alpha,
        "p_versus_chance": float(binomtest(n_sig, len(tested), alpha, alternative="greater").pvalue) if len(tested) else None,
    }
    if g.MOTION_STATUS_COLUMN in tested:
        result["cells_significant_motion_suspected"] = int(((tested["n_significant_clusters"].fillna(0) > 0) & (tested[g.MOTION_STATUS_COLUMN] == g.MOTION_SUSPECTED)).sum())
    return result


def hypothesis_results(settings, experiment_id: int, step: str) -> dict:
    """The numbers of the newest completed run of one hypothesis: {run_id, result_types: {name: numbers}, group_test (when the group test job has run)}."""
    run, sta_dir = completed_sta_run(settings, experiment_id, step)
    table = pd.read_csv(path_for(PathKind.STA_SUMMARY_CSV, sta_dir))
    params = json.loads(run["parameters_json"])
    results = {str(name): _result_type_summary(rows, params) for name, rows in table.groupby(table["spike_group"].astype(str))}
    group_test = read_json(path_for(PathKind.GROUP_TEST, sta_dir), "The real-vs-fake group test", default=None)
    return {"run_id": int(run["id"]), "result_types": results, "group_test": group_test}


def _waiting_for(settings, experiment_id: int, run: dict) -> list[str]:
    """The feeding steps a waiting run still waits for."""
    needed = feeding_steps(json.loads(run["parameters_json"])["result_types"])
    return sorted(needed - run_repository.steps_with_status(settings, experiment_id, ("completed",)))


def hypothesis_summary(settings, experiment_id: int, step: str) -> dict:
    """One hypothesis: name, description, settings, the status of its newest run (with what a waiting run waits for) and, when a run completed,
    its results. A rerun in progress shows its own status next to the results of the earlier completed run."""
    hypothesis = hypothesis_for(step)
    newest = run_repository.latest_run(settings, int(experiment_id), step, status=None)
    summary = {"name": hypothesis.name, "step": step, "description": hypothesis.description, "settings": hypothesis.settings(),
               "status": newest["status"] if newest else "not_run", "run_id": int(newest["id"]) if newest else None, "waiting_for": [], "results": None}
    if newest and newest["status"] == "waiting":
        summary["waiting_for"] = _waiting_for(settings, experiment_id, newest)
    if run_repository.latest_run(settings, int(experiment_id), step) is not None:  # some run of this hypothesis completed
        try:
            summary["results"] = hypothesis_results(settings, experiment_id, step)
        except FileNotFoundError:
            summary["results_warning"] = missing_output_message("The summary table", summary["run_id"])
    return summary


def hypothesis_summaries(settings, experiment_id: int) -> list[dict]:
    """The summary of every hypothesis of the catalogue, in catalogue order."""
    return [hypothesis_summary(settings, experiment_id, step) for step in sta_steps()]
