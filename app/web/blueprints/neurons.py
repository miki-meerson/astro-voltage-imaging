"""The data of a cell for the Cells and Spikes tabs: summary images, the cropped STA movie and the review mark."""
from __future__ import annotations

from io import BytesIO

from flask import jsonify, render_template, request, send_file

from app.db import neuron_repository
from app.services.cell_views import cell_movie_files, sta_all_stack_payload, sta_crop_stack_payload
from app.services.hypothesis_summary import chosen_step
from app.services.runs import run_dir
from app.services.spikes import (
    load_spike_payload,
    refresh_review_entry,
    save_spike_payload,
    stale_suite2p_cell_message,
)
from app.services.visualizations import render_result_map
from app.web.common import FlatBlueprint, get_settings
from utils import globals_utils as g
from utils.paths_utils import PathKind, path_for

bp = FlatBlueprint("neurons", __name__)


def _step() -> str:
    """The hypothesis step of the request (?h=<name>; the default hypothesis when none is given)."""
    return chosen_step(request.args.get("h"))


@bp.get("/neurons/<int:neuron_id>/sta-map.png")
def neuron_sta_map(neuron_id):
    settings = get_settings()
    n = neuron_repository.neuron_row(settings, neuron_id)
    if n is None:
        return render_template("error.html", title="Missing Cell", message="Cell not found"), 404
    return send_file(render_result_map(settings, n, _step(), request.args.get("type", "")), mimetype="image/png")


@bp.get("/neurons/<int:neuron_id>/anatomy-crop/stack")
def neuron_anatomy_crop_stack(neuron_id):
    settings = get_settings()
    try:
        files = cell_movie_files(settings, neuron_id, _step())
        if request.args.get("type", g.ANATOMY_GROUP) == g.ALL_GROUP:  # ?type=all: the whole-field STA movie of the 'all' result; the default is the anatomy crop
            payload = sta_all_stack_payload(files.all_npz, files.source_path, files.pipeline_neuron_id)
        else:
            payload = sta_crop_stack_payload(files.movie_tif, files.anatomy_npz, files.source_path, files.pipeline_neuron_id)
        return send_file(BytesIO(payload), mimetype="application/octet-stream", max_age=300)
    except FileNotFoundError as exc:  # a saved file is missing: the text says which step writes it
        return jsonify({"error": str(exc)}), 404
    except (ValueError, KeyError, OSError) as exc:  # unknown cell, movie of another STA run, no fps in Experiment.xml, a file that cannot be read
        return jsonify({"error": str(exc)}), 400


@bp.post("/neurons/<int:neuron_id>/review/mark")
def mark_cell_reviewed(neuron_id):
    settings = get_settings()
    n, spike_run = neuron_repository.neuron_with_latest_run(settings, neuron_id, g.SPIKE_STEP)
    if n is None:
        return jsonify({"error": "Cell not found"}), 404
    if spike_run is None:
        return jsonify({"error": "No completed spike run."}), 400
    stale_message = stale_suite2p_cell_message(n["source_path"], n["pipeline_neuron_id"])
    if stale_message:
        return jsonify({"error": stale_message}), 400
    run_folder = run_dir(settings, int(spike_run["id"]), bool(spike_run["is_test"]))
    cell = n["pipeline_neuron_id"]
    try:
        payload = load_spike_payload(path_for(PathKind.SPIKE_LABELS, run_folder, cell=cell))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    save_spike_payload(run_folder, cell, payload["peaks_all"], payload["labels"])
    return jsonify({"ok": True, **refresh_review_entry(settings, int(n["experiment_id"]), spike_run, cell)})
