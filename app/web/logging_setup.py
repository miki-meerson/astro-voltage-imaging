"""File logging (runtime/logs/web.log) and the 500 handler that returns an error id."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
import uuid

from flask import Flask, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from app.web.security import wants_json

_HANDLER_NAME = "astro-web-file"


def configure_logging(app: Flask, settings) -> Path:
    log_dir = Path(settings.project_root) / "runtime" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "web.log"
    level_name = str(((settings.raw.get("logging") or {}).get("level")) or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger("astro")
    logger.setLevel(level)
    if not any(getattr(h, "name", "") == _HANDLER_NAME for h in logger.handlers):
        handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    app.logger.parent = logger  # app.logger ("app.web.server") messages also go to the file
    return log_path


def init_error_handling(app: Flask) -> None:
    log = logging.getLogger("astro.web")

    @app.before_request
    def _start_timer():
        g._t0 = time.time()

    @app.after_request
    def _log_request(response):
        if not request.path.startswith("/static/"):
            log.info("%s %s -> %s (%.0f ms)", request.method, request.full_path.rstrip("?"), response.status_code,
                     (time.time() - getattr(g, "_t0", time.time())) * 1000)
        return response

    @app.errorhandler(Exception)
    def _unexpected_error(exc):
        if isinstance(exc, HTTPException):
            return exc  # 404, 405, ... keep Flask's normal handling
        error_id = uuid.uuid4().hex[:8]
        log.exception("unhandled error %s on %s %s", error_id, request.method, request.full_path)
        message = f"Unexpected server error ({type(exc).__name__}). Error id: {error_id}. Details are in runtime/logs/web.log."
        if wants_json():
            return jsonify({"error": message, "error_id": error_id}), 500
        return render_template("error.html", title="Server Error", message=message), 500
