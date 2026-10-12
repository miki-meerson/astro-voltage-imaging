"""Folder browser used by the "add experiment" form."""
from __future__ import annotations

from pathlib import Path

from flask import render_template, request

from app.web.common import FlatBlueprint, get_settings
from app.web.security import browse_roots, path_within

bp = FlatBlueprint("files", __name__)


@bp.get("/browse")
def browse():
    settings = get_settings()
    roots = browse_roots(settings)
    current = Path(request.args.get("path") or settings.scan_parent)
    if not path_within(current, roots):
        return render_template("error.html", title="Browse Failed", message="This folder is outside the configured data roots."), 403
    if not current.exists():
        current = settings.scan_parent
    if current.is_file():
        current = current.parent
    dirs = []
    try:
        dirs = sorted([p for p in current.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    except OSError as exc:
        return render_template("error.html", title="Browse Failed", message=str(exc)), 400
    parent = current.parent if current.parent != current else None
    if parent is not None and not path_within(parent, roots):
        parent = None  # no ".." above the data roots
    return render_template("browse.html", current=current, parent=parent, dirs=dirs)
