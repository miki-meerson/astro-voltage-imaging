"""Shared plumbing for the route modules."""
from __future__ import annotations

from io import BytesIO

from flask import Blueprint, current_app, jsonify, request, send_file
from flask.blueprints import BlueprintSetupState

from app.db.experiment_repository import experiment_source_path
from app.services import polarity
from app.services.movies import movie_chunk, movie_info, pb_movie_path
from utils.movie_utils import read_frame_rate


class _FlatSetupState(BlueprintSetupState):
    """Register rules under their plain endpoint name ("neuron", not "neurons.neuron").

    The templates and the redirects use url_for("experiment_overview", ...) etc.; keeping the endpoint names
    unchanged means the split into blueprints does not touch any of them.
    """

    def add_url_rule(self, rule, endpoint=None, view_func=None, **options):
        if self.url_prefix is not None:
            rule = f"{self.url_prefix.rstrip('/')}/{rule.lstrip('/')}" if rule else self.url_prefix
        options.setdefault("subdomain", self.subdomain)
        if endpoint is None:
            endpoint = view_func.__name__
        defaults = self.url_defaults
        if "defaults" in options:
            defaults = dict(defaults, **options.pop("defaults"))
        self.app.add_url_rule(rule, endpoint, view_func, defaults=defaults, **options)


class FlatBlueprint(Blueprint):
    def make_setup_state(self, app, options, first_registration=False):
        return _FlatSetupState(self, app, options, first_registration)


def get_settings():
    """The Settings object of the running app (set once in create_app)."""
    return current_app.config["SETTINGS"]


def _movie_of(experiment_id: int):
    """(movie file, frame rate, flip the sign, error response) of ?kind= of the experiment; the error response is None when all is well."""
    kind = request.args.get("kind", "astro")
    source_path = experiment_source_path(get_settings(), experiment_id)
    if source_path is None:
        return None, None, None, (jsonify({"available": False, "error": "Experiment not found.", "message": "Experiment not found."}), 404)
    path = pb_movie_path(source_path, kind)
    if path is None:
        return None, None, None, (jsonify({"available": False, "error": "Unknown movie type.", "message": "Unknown movie type."}), 400)
    if not path.exists():
        message = f"{path.name} was not found. The photobleaching correction step writes it: rerun that step for this experiment."
        return None, None, None, (jsonify({"available": False, "error": message, "message": message}), 404)
    return path, read_frame_rate(source_path), polarity.flip_for_kind(kind), None


def movie_meta_response(experiment_id: int):
    """JSON description of a corrected movie (?kind=astro|neuron): size, frame count, frame rate, contrast and the number of frames per chunk."""
    path, fps, flip, error = _movie_of(experiment_id)
    if error:
        return error
    try:
        return jsonify(movie_info(path, fps, flip))
    except (ValueError, OSError) as exc:
        return jsonify({"available": False, "message": str(exc)}), 400


def movie_chunk_response(experiment_id: int):
    """One chunk of frames of a corrected movie (?kind, ?chunk) as raw 8-bit values: the page plays the movie by asking for chunk after chunk."""
    path, fps, flip, error = _movie_of(experiment_id)
    if error:
        return error
    try:
        data = movie_chunk(path, fps, flip, request.args.get("chunk", 0, type=int))
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400
    return send_file(BytesIO(data), mimetype="application/octet-stream", max_age=3600)
