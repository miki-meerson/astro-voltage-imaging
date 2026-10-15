"""Experiment-level view of the significant anatomy-configuration clusters: READS the saved summary only.

The summary is computed where the data is created (utils/anatomy_summary.py, written by the STA Maps step) and saved as
<latest completed STA run>/sta_algorithm_home/sta-analysis/anatomy_summary.json.
This module opens that one file: no listing of cell folders, no recomputation, no cache file. When the file is missing it raises a
FileNotFoundError that says which step writes it. The cluster signs are stored as saved ('positive' / 'negative'); they are mapped to
depolarizing / hyperpolarizing here (app/services/polarity.py).
"""
from __future__ import annotations

from app.db import experiment_repository
from app.services import polarity
from app.services.hypothesis_summary import completed_sta_run
from app.services.runs import default_sta_step
from utils.anatomy_summary import load_anatomy_summary
from utils.file_utils import read_json
from utils.paths_utils import PathKind, path_for


def anatomy_cluster_summary(settings, experiment_id: int, step: str | None = None) -> dict:
    """The saved summary of the latest completed run of an STA step (default: the default hypothesis), with polarity labels, neuron ids and the
    real-vs-fake group test when a cluster job has made it."""
    run, sta_dir = completed_sta_run(settings, experiment_id, step or default_sta_step())
    saved = load_anatomy_summary(sta_dir, run_label=int(run["id"]))
    if saved["warning"]:
        return {"available": False, "message": saved["warning"]}
    db_ids = experiment_repository.neuron_ids_by_cell(settings, int(experiment_id))
    clusters = [dict(c, sign=polarity.polarity_from_sign(str(c.get("sign", ""))), neuron_id=db_ids.get(int(c["cell"]))) for c in saved["clusters"]]
    payload = {
        "available": True, "run_id": int(run["id"]), "cells": saved["cells"], "clusters": clusters,
        "counts": saved["counts"], "excluded": saved["excluded"],
    }
    group_test = read_json(path_for(PathKind.GROUP_TEST, sta_dir), "The real-vs-fake group test", default=None)  # a separate cluster job: shown when it has run
    if group_test is not None:
        payload["group_test"] = group_test
    return payload
