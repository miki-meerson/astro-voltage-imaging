"""Run detail page, logs and artifact download; rerun of failed runs."""
from __future__ import annotations

import sqlite3

from flask import jsonify, redirect, render_template, request, send_file, url_for

from app.db import run_repository
from app.services.run_views import run_stderr, run_stdout, run_view
from app.services.runs import rerun_failed_steps
from app.web.common import FlatBlueprint, get_settings

bp = FlatBlueprint("runs", __name__)


@bp.post("/runs/rerun-failed")
def rerun_failed():
    settings = get_settings()
    try:
        step = request.form.get("step") or None
        created = rerun_failed_steps(settings, step)
        return redirect(url_for("index", rerun_created=len(created)))
    except (ValueError, OSError, sqlite3.Error) as exc:
        return render_template("error.html", title="Rerun Failed", message=str(exc)), 400


@bp.get("/runs/<int:run_id>")
def run_detail(run_id):
    settings = get_settings()
    run = run_repository.run_with_experiment(settings, run_id)
    if run is None:
        return render_template("error.html", title="Missing Run", message="Run not found"), 404
    rv = run_view(run, settings)
    return render_template("run.html", run=rv, stdout=run_stdout(rv), stderr=run_stderr(rv), active=rv["status"] in {"pending", "queued", "running"})


@bp.get("/runs/<int:run_id>/logs")
def run_logs(run_id):
    settings = get_settings()
    run = run_repository.run_row(settings, run_id)
    if run is None:
        return jsonify({"error": "Run not found"}), 404
    return jsonify(
        {
            "status": run["status"],
            "stdout": run_stdout(run),
            "stderr": run_stderr(run),
        }
    )


@bp.get("/runs/<int:run_id>/artifact/<path:rel_path>")
def artifact(run_id, rel_path):
    settings = get_settings()
    run = run_repository.run_row(settings, run_id)
    if run is None:
        return render_template("error.html", title="Missing Run", message="Run not found"), 404
    root = settings.test_output_root if run["is_test"] else settings.output_root
    target = (root / rel_path).resolve()
    allowed = root.resolve()
    if allowed not in target.parents and target != allowed:
        return render_template("error.html", title="Unsafe Path", message="Artifact path is outside the output root."), 400
    if not target.exists() or not target.is_file():
        return render_template("error.html", title="Missing Artifact", message=str(target)), 404
    return send_file(target)
