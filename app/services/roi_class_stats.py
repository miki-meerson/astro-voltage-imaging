"""Soma vs dendrite: do the accepted ROIs of one class have significant clusters more often than the other? Reads the saved ROI classification (roi_classes.json) and the
saved summary table of a hypothesis' newest completed run; nothing is computed from movies.

A class with more cells has more chances to show a significant cell, so counts are never compared as they are. Per class the page gets the FRACTION of its tested cells that
are significant, with a Wilson 95% interval and a test against the false-positive rate alone (`cluster_alpha`). The two classes are compared on those fractions by Fisher's
exact test (which allows for the groups having different sizes), and again by a permutation test that keeps the group sizes and, since a cell with more spikes is more likely
to be significant, shuffles the class only among cells with a similar number of spikes. Cells whose class is "uncertain", or that have none, are listed and not compared."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from app.services.hypothesis_summary import completed_sta_run, tested_cells
from app.services.runs import hypothesis_for
from app.services.suite2p_data import experiment_rois
from utils import roi_classification as rc
from utils.file_utils import read_json
from utils.sta_stats_utils import significant_fraction
from utils.output_errors import missing_output_message
from utils.paths_utils import PathKind, path_for
from utils.roi_groups import fold_anatomy_parts

PERMUTATIONS = 10000
STRATA = 3  # cells are grouped by the number of spikes into this many equal groups, when there are enough cells for it
MIN_CELLS_PER_STRATUM = 4
NO_LABEL = "unclassified"


def _class_row(label: str, tested: pd.DataFrame, alpha: float) -> dict:
    """The numbers of the tested cells of one class."""
    n, k = len(tested), int(tested["significant"].sum())
    return {"label": label, "tested": n, "significant": k, **significant_fraction(k, n, alpha), "spikes_median": float(tested["n_peaks"].median()) if n else None}


def _soma_vs_dendrite(tested: pd.DataFrame) -> dict:
    """Fisher's exact p and the spike-matched permutation p of the difference between the significant fractions of the soma and the dendrite cells."""
    soma = tested.loc[tested["label"] == rc.SOMA]
    dendrite = tested.loc[tested["label"] == rc.DENDRITE]
    k_soma, k_dendrite = int(soma["significant"].sum()), int(dendrite["significant"].sum())
    both = pd.concat([soma, dendrite])
    significant, is_soma, spikes = both["significant"].to_numpy(float), (both["label"] == rc.SOMA).to_numpy(), both["n_peaks"].to_numpy(float)
    n_strata = max(1, min(STRATA, len(both) // MIN_CELLS_PER_STRATUM))
    stratum = np.digitize(spikes, np.quantile(spikes, np.arange(1, n_strata) / n_strata)) if n_strata > 1 else np.zeros(len(both), dtype=int)
    rng = np.random.default_rng(0)  # fixed: the same saved results always give the same p
    soma_significant = np.zeros(PERMUTATIONS)
    for s in np.unique(stratum):
        idx = stratum == s
        soma_significant += (rng.permuted(np.tile(is_soma[idx], (PERMUTATIONS, 1)), axis=1) * significant[idx]).sum(axis=1)
    n_soma, n_dendrite = len(soma), len(dendrite)
    difference = soma_significant / n_soma - (significant.sum() - soma_significant) / n_dendrite
    observed = k_soma / n_soma - k_dendrite / n_dendrite
    return {
        "difference": observed, "fisher_p": float(fisher_exact([[k_soma, n_soma - k_soma], [k_dendrite, n_dendrite - k_dendrite]])[1]),
        "matched_p": float((1 + np.count_nonzero(np.abs(difference) >= abs(observed) - 1e-12)) / (PERMUTATIONS + 1)), "strata": n_strata,
    }


def roi_class_statistics(settings, exp: dict, step: str) -> dict:
    """{available, message} or {available, run_id, alpha, result_types: [{name, classes: [per class numbers], comparison: soma vs dendrite or None, comparison_message}]} for the newest
    completed run of an STA step. Raises FileNotFoundError when the hypothesis has no completed run."""
    if hypothesis_for(step).pool_cells:
        return {"available": False, "message": "A pooled hypothesis has one result for the whole experiment, not one per cell."}
    source = exp["source_path"]
    _, stat = experiment_rois(source)
    try:
        saved = rc.load_classes(source, rc.stat_fingerprint(stat))
    except ValueError as exc:
        return {"available": False, "message": str(exc)}
    if saved is None:
        return {"available": False, "message": "This experiment has no ROI classification yet: run it on the ROI types tab of the Preprocessing page."}
    run, sta_dir = completed_sta_run(settings, exp["id"], step)
    table = fold_anatomy_parts(pd.read_csv(path_for(PathKind.STA_SUMMARY_CSV, sta_dir)))
    used = read_json(path_for(PathKind.STA_USED_PARAMETERS_JSON, sta_dir), "The STA used-parameters file", missing=missing_output_message("The used-parameters file", run["id"]))
    alpha = float(used["cluster_alpha"])
    labels = {row["cell"]: row["label"] for row in saved["rows"].values()}
    result_types = []
    for name, rows in table.groupby(table["spike_group"].astype(str)):
        tested = tested_cells(rows, used).copy()
        tested["significant"] = tested["n_significant_clusters"].fillna(0) > 0
        tested["label"] = tested["cell_id"].map(labels).fillna(NO_LABEL)
        classes = [_class_row(label, tested[tested["label"] == label], alpha) for label in (rc.SOMA, rc.DENDRITE, rc.UNCERTAIN, NO_LABEL)]
        classes = [row for row in classes if row["tested"] or row["label"] in rc.MANUAL_LABELS]
        enough = all(row["tested"] for row in classes if row["label"] in rc.MANUAL_LABELS)
        result_types.append({
            "name": str(name), "classes": classes, "comparison": _soma_vs_dendrite(tested) if enough else None,
            "comparison_message": "" if enough else "Soma and dendrite cells need at least one tested cell each to be compared.",
        })
    return {"available": True, "run_id": int(run["id"]), "alpha": alpha, "result_types": result_types}
