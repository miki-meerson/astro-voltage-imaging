"""Shared plumbing for the route modules."""
from __future__ import annotations

from io import BytesIO

from flask import Blueprint, current_app, jsonify, request, send_file
from flask.blueprints import BlueprintSetupState

from app.db.experiment_repository import experiment_source_path
from app.services import polarity
from app.services.movies import movie_frame_jpeg, movie_metadata
from utils.movie_utils import read_frame_rate


class _FlatSetupState(BlueprintSetupState):
    """Register rules under their plain endpoint name ("neuron", not "neurons.neuron").

    The templates and the redirects use url_for("experiment_analysis", ...) etc.; keeping the endpoint names
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


def _movie_path_for(experiment_id: int, kind: str, path_of):
    """(source_path, movie path, error response). path_of(source_path, kind) gives the movie file, or None for an unknown kind."""
    source_path = experiment_source_path(get_settings(), experiment_id)
    if source_path is None:
        return None, None, "Experiment not found.", 404
    path = path_of(source_path, kind)
    if path is None:
        return None, None, "Unknown movie type.", 400
    return source_path, path, None, 200


def movie_meta_response(experiment_id: int, path_of):
    """JSON metadata of a movie of the experiment (kind = ?kind=astro|neuron), for the experiment and preprocessing viewers."""
    kind = request.args.get("kind", "astro")
    source_path, path, error, status = _movie_path_for(experiment_id, kind, path_of)
    if error:
        return jsonify({"available": False, "message": error}), status
    if not path.exists():
        return jsonify({"available": False, "message": f"{path.name} was not found."})
    try:
        payload = movie_metadata(path, read_frame_rate(source_path))
        payload["kind"] = kind
        payload["gevi_flipped"] = polarity.flip_for_kind(kind)
        return jsonify(payload)
    except (ValueError, OSError) as exc:
        return jsonify({"available": False, "message": str(exc)}), 500


def movie_frame_response(experiment_id: int, path_of):
    """One frame of a movie of the experiment as JPEG (?kind, ?frame, ?max_size, ?contrast), sign-flipped for astro."""
    kind = request.args.get("kind", "astro")
    try:
        frame_index = max(0, int(request.args.get("frame", "0")))
        max_size = min(1400, max(240, int(request.args.get("max_size", "720"))))
    except ValueError:
        return jsonify({"error": "Invalid frame or max_size."}), 400
    _, path, error, status = _movie_path_for(experiment_id, kind, path_of)
    if error:
        return jsonify({"error": error}), status
    if not path.exists():
        return jsonify({"error": f"{path.name} was not found."}), 404
    try:
        jpeg = movie_frame_jpeg(path, frame_index, max_size, request.args.get("contrast", "auto"), invert_sign=polarity.flip_for_kind(kind))
        return send_file(BytesIO(jpeg), mimetype="image/jpeg", max_age=600)
    except (ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 500
