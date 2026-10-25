"""The pages of the experiment workspace: overview, spikes, hypotheses, cells and runs (the preprocessing page is in preprocessing.py).

Every page shows the same frame (tabs and step bar, app/services/workspace.py). The hypothesis chosen in the address (?h=<name>) only matters on the
pages that show STA results: Hypotheses and Cells."""
from __future__ import annotations

import json

from flask import redirect, render_template, request, url_for

from app.db import experiment_repository as repo
from app.db import neuron_repository
from app.db.run_repository import latest_run
from app.services.cell_views import pooled_row
from app.services.experiments import start_spike_detection
from app.services.hypothesis_summary import cell_table, chosen_step, hypothesis_status, hypothesis_summaries, waiting_for
from app.services.roi_class_stats import roi_class_statistics
from app.services.run_views import build_validation_rows, run_view, validation_all_ok
from app.services.runs import STA_RESULT_TYPES, hypothesis_for, sta_steps
from app.services.spikes import current_suite2p_neurons, experiment_spike_review, spikes_edited_after_sta
from app.services.visualizations import cell_motion, cell_result_cards
from app.services.workspace import page_data
from app.web.common import FlatBlueprint, get_settings
from utils import globals_utils as g
from utils.hypotheses import hypothesis_name

bp = FlatBlueprint("workspace", __name__)


def _missing():
    return render_template("error.html", title="Missing Experiment", message="Experiment not found"), 404


@bp.get("/experiments/<int:experiment_id>/overview")
def experiment_overview(experiment_id):
    settings = get_settings()
    data = page_data(settings, experiment_id, "overview")
    if data is None:
        return _missing()
    rows = build_validation_rows(json.loads(data["exp"]["validation_report"] or "{}"))
    return render_template(
        "overview.html", **data,
        hypotheses=[hypothesis_status(settings, experiment_id, step) for step in sta_steps()],
        result_types=STA_RESULT_TYPES, sta_defaults=dict(settings.defaults.get("sta", {})),
        analysis_ready=data["exp"]["workflow_stage"] == repo.STAGE_ANALYSIS_READY,
        validation_rows=rows, validation_ok=validation_all_ok(rows) and data["exp"]["validation_state"] == "valid",
        notice=request.args.get("notice"),
    )


@bp.get("/experiments/<int:experiment_id>/spikes")
def experiment_spikes(experiment_id):
    settings = get_settings()
    data = page_data(settings, experiment_id, "spikes")
    if data is None:
        return _missing()
    analysis_ready = data["exp"]["workflow_stage"] == repo.STAGE_ANALYSIS_READY
    review = experiment_spike_review(settings, experiment_id) if analysis_ready else {"has_spike_run": False, "reviewed_count": 0, "total_count": 0, "cells": []}
    unreviewed_only = request.args.get("unreviewed") == "1"
    shown = [cell for cell in review["cells"] if not (unreviewed_only and cell["reviewed"])]
    wanted = request.args.get("cell", type=int)
    selected = next((cell for cell in review["cells"] if cell["cell"] == wanted), None)
    return render_template("spikes.html", **data, review=review, cells=shown, selected=selected, unreviewed_only=unreviewed_only, analysis_ready=analysis_ready,
                           notice=request.args.get("notice"))


@bp.post("/experiments/<int:experiment_id>/spikes/redetect")
def redetect_spikes(experiment_id):
    settings = get_settings()
    try:
        if repo.workflow_stage_of(settings, experiment_id) != repo.STAGE_ANALYSIS_READY:
            raise ValueError("Spike detection needs the confirmed Suite2p cells: confirm Suite2p on the Preprocessing tab first.")
        start_spike_detection(settings, experiment_id)
        return redirect(url_for("experiment_spikes", experiment_id=experiment_id, notice="redetecting"))
    except (ValueError, OSError) as exc:
        return render_template("error.html", title="Spike Detection Failed", message=str(exc)), 400


@bp.get("/experiments/<int:experiment_id>/hypotheses")
def experiment_hypotheses(experiment_id):
    settings = get_settings()
    data = page_data(settings, experiment_id, "hypotheses")
    if data is None:
        return _missing()
    try:
        step = chosen_step(request.args.get("h"))
    except ValueError as exc:
        return render_template("error.html", title="Unknown Hypothesis", message=str(exc)), 400
    try:
        class_stats = roi_class_statistics(settings, data["exp"], step)
    except FileNotFoundError as exc:  # the hypothesis has no completed run, or Suite2p results are missing
        class_stats = {"available": False, "message": str(exc)}
    summaries = hypothesis_summaries(settings, experiment_id)
    n_tests = sum(len(summary["results"]["result_types"]) for summary in summaries if summary["results"])  # hypothesis x result type: each is one test per cell
    return render_template("hypotheses.html", **data, summaries=summaries, selected=hypothesis_name(step), class_stats=class_stats, n_tests=n_tests, result_types=STA_RESULT_TYPES,
                           sta_defaults=dict(settings.defaults.get("sta", {})), analysis_ready=data["exp"]["workflow_stage"] == repo.STAGE_ANALYSIS_READY)


def _cell_list(settings, experiment_id: int, step: str, args) -> tuple[list[dict], dict | None, str]:
    """(the cells to list, the run of the hypothesis' table or None, a message when there is no table): the current Suite2p cells with their flags,
    cut by the filters in the address (?significant=1, ?anatomy=1)."""
    neurons = current_suite2p_neurons(repo.experiment_source_path(settings, experiment_id), repo.experiment_neurons(settings, experiment_id))
    try:
        table = cell_table(settings, experiment_id, step)
    except FileNotFoundError as exc:
        return [{"id": int(n["id"]), "cell": int(n["pipeline_neuron_id"])} for n in neurons], None, str(exc)
    flags = table["cells"]
    cells = [{"id": int(n["id"]), "cell": int(n["pipeline_neuron_id"]), **flags.get(int(n["pipeline_neuron_id"]), {})} for n in neurons]
    if args.get("significant") == "1":
        cells = [cell for cell in cells if cell.get("significant")]
    if args.get("anatomy") == "1":
        cells = [cell for cell in cells if cell.get("anatomy")]
    return cells, table, ""


@bp.get("/experiments/<int:experiment_id>/cells")
def experiment_cells(experiment_id):
    settings = get_settings()
    data = page_data(settings, experiment_id, "cells")
    if data is None:
        return _missing()
    try:
        step = chosen_step(request.args.get("h"))
    except ValueError as exc:
        return render_template("error.html", title="Unknown Hypothesis", message=str(exc)), 400
    cells, table, message = _cell_list(settings, experiment_id, step, request.args)
    pooled = hypothesis_for(step).pool_cells  # one result for the whole experiment: the only "cell" of the list is the pooled one
    wanted = g.POOLED_CELL_ID if pooled else request.args.get("cell", type=int)
    if pooled and table is not None:
        cells = [{"id": None, "cell": g.POOLED_CELL_ID, **table["cells"].get(g.POOLED_CELL_ID, {})}]
    selected = next((cell for cell in cells if cell["cell"] == wanted), None) if table is not None else None
    sta_run, stale, cards, motion, motion_message = latest_run(settings, experiment_id, step), None, [], None, ""
    spike_run = latest_run(settings, experiment_id, g.SPIKE_STEP)
    if selected is not None:
        edited = None if pooled else spikes_edited_after_sta(settings, spike_run, sta_run, selected["cell"])
        stale = f"The spikes of this cell were edited on {edited:%Y-%m-%d %H:%M}, after this run started: the results used the old spikes. Rerun this hypothesis." if edited else None
        row = pooled_row(settings, experiment_id) if pooled else neuron_repository.neuron_row(settings, selected["id"])
        try:
            cards = cell_result_cards(settings, row, step)
        except FileNotFoundError as exc:
            message = str(exc)
        try:
            motion = cell_motion(settings, row, step)
        except FileNotFoundError as exc:  # the cards can exist without the motion test (an older run): say so next to the test
            motion_message = str(exc)
    map_base = f"/experiments/{experiment_id}/pooled-sta-map.png?h={hypothesis_name(step)}&type=" if pooled else (f"/neurons/{selected['id']}/sta-map.png?h={hypothesis_name(step)}&type=" if selected else "")
    return render_template("cells.html", **data, hypotheses=[hypothesis_name(s) for s in sta_steps()], selected_hypothesis=hypothesis_name(step), cells=cells, message=message,
                           selected=selected, pooled=pooled, map_base=map_base, sta_run=sta_run, stale=stale, cards=cards, motion=motion, motion_message=motion_message, only_significant=request.args.get("significant") == "1", only_anatomy=request.args.get("anatomy") == "1")


@bp.get("/experiments/<int:experiment_id>/runs")
def experiment_runs(experiment_id):
    settings = get_settings()
    data = page_data(settings, experiment_id, "runs")
    if data is None:
        return _missing()
    runs = [run_view(row, settings) for row in data["runs"]]
    for run in runs:
        run["waiting_for"] = waiting_for(settings, experiment_id, run) if run["status"] == "waiting" else []
    return render_template("runs.html", **{**data, "runs": runs}, status_filter=request.args.get("status", ""))
