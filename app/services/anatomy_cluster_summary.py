"""Experiment-level view of the significant anatomy-configuration clusters: READS the saved summary only.

The summary is computed where the data is created (utils/anatomy_summary.py, written by the STA Maps step) and saved as
<latest completed STA run>/sta_algorithm_home/sta-analysis/anatomy_summary.json.
This module opens that one file: no listing of cell folders, no recomputation, no cache file. When the file is missing it raises a
FileNotFoundError that says which step writes it. The cluster signs are stored as saved ('positive' / 'negative'); they are mapped to
depolarizing / hyperpolarizing here (app/services/polarity.py).
"""
from __future__ import annotations

import json

from app.db import experiment_repository, run_repository
from app.services import polarity
from app.services.runs import run_dir
from utils.anatomy_summary import load_anatomy_summary
from utils.paths_utils import PathKind, path_for, sta_dir_of_run


def anatomy_cluster_summary(settings, experiment_id: int) -> dict:
    """The saved summary of the latest completed STA run (with polarity labels, neuron ids and the real-vs-fake group test when a cluster job has made it)."""
    run = run_repository.latest_completed_run_row(settings, int(experiment_id), "sta")
    if run is None:
        raise FileNotFoundError("No completed STA run for this experiment: run STA Maps first.")
    sta_dir = sta_dir_of_run(run_dir(settings, int(run["id"]), bool(run["is_test"])))
    saved = load_anatomy_summary(sta_dir, run_label=int(run["id"]))
    if saved["warning"]:
        return {"available": False, "message": saved["warning"]}
    db_ids = experiment_repository.neuron_ids_by_cell(settings, int(experiment_id))
    clusters = [dict(c, sign=polarity.polarity_from_sign(str(c.get("sign", ""))), neuron_id=db_ids.get(int(c["cell"]))) for c in saved["clusters"]]
    payload = {
        "available": True, "run_id": int(run["id"]), "cells": saved["cells"], "clusters": clusters,
        "counts": saved["counts"], "excluded": saved["excluded"],
    }
    try:  # the real-vs-fake test is a separate cluster job: shown when it has run
        payload["group_test"] = json.loads(path_for(PathKind.GROUP_TEST, sta_dir).read_text(encoding="utf-8"))
    except FileNotFoundError:  # not made yet
        pass
    return payload
