"""Flask application factory. Routes live in app/web/blueprints/, logic in app/services/."""
from __future__ import annotations

from flask import Flask

from app.db.core import migrate
from app.settings import load_settings
from app.web.blueprints import experiments, files, neurons, preprocessing, runs, spikes
from app.web.logging_setup import configure_logging, init_error_handling
from app.web.security import init_security

BLUEPRINTS = (experiments.bp, preprocessing.bp, neurons.bp, spikes.bp, runs.bp, files.bp)


def create_app() -> Flask:
    settings = load_settings()
    migrate(settings)
    app = Flask(__name__)
    app.config["SETTINGS"] = settings
    configure_logging(app, settings)
    init_security(app, settings)
    init_error_handling(app)
    for blueprint in BLUEPRINTS:
        app.register_blueprint(blueprint)
    return app
