"""Small protective checks for the local web app: cross-origin POST rejection, /browse roots, redirect targets."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, jsonify, render_template, request

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def is_safe_next(target: str | None) -> bool:
    """True only for a same-site relative URL ("/experiments/6#notes"); rejects absolute URLs, "//host" and backslash tricks."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return False
    parts = urlsplit(target)
    return not parts.scheme and not parts.netloc


def safe_next(target: str | None, default: str) -> str:
    return target if is_safe_next(target) else default


def _norm(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def browse_roots(settings) -> list[Path]:
    """Folders /browse may show: the data root, the scan parent and optional `browse_roots` from the config."""
    roots = [settings.source_data_root, settings.scan_parent]
    roots += [Path(p) for p in (settings.raw.get("browse_roots") or [])]
    return roots


def path_within(path: str | Path, roots) -> bool:
    target = _norm(path)
    for root in roots:
        base = _norm(root)
        if target == base or target.startswith(base.rstrip(os.sep + "/") + os.sep):
            return True
    return False


def wants_json() -> bool:
    return request.is_json or request.path.endswith((".json", "/data")) or request.path.endswith(
        ("/detect", "/save", "/mark", "/stack", "/meta"))


def init_security(app: Flask, settings) -> None:
    allowed_hosts = {str(h).lower() for h in ((settings.raw.get("security") or {}).get("allowed_origins") or [])}

    @app.before_request
    def reject_cross_origin_writes():
        if request.method not in UNSAFE_METHODS:
            return None
        if request.headers.get("Sec-Fetch-Site") == "cross-site":
            blocked = True
        else:
            origin = request.headers.get("Origin") or request.headers.get("Referer")
            if not origin:
                return None  # not a browser request (scripts, curl, tests): nothing to protect against
            host = urlsplit(origin).netloc.lower()
            blocked = host != request.host.lower() and host not in allowed_hosts
        if not blocked:
            return None
        message = "Cross-origin request blocked: write requests must come from this app's own pages."
        app.logger.warning("blocked cross-origin %s %s (Origin=%r)", request.method, request.path, request.headers.get("Origin"))
        if wants_json():
            return jsonify({"error": message}), 403
        return render_template("error.html", title="Request Blocked", message=message), 403
