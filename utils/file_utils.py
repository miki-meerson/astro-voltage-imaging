"""The one way to open a small saved json or npz file: a missing file raises an error that names it and says to rerun the step that writes it,
unless the caller says the file may be absent (then `default` is returned). No Flask, no app imports."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REQUIRED = object()  # default of the readers below: the file must exist


def _missing(path, what: str, missing: str | None, default):
    """`default` when the caller allows it, else the FileNotFoundError that names the file (`missing`: the caller's own full message)."""
    if default is not REQUIRED:
        return default
    raise FileNotFoundError(missing or f"{what} is missing: {path}. It is written by an earlier analysis step: rerun that step.")


def read_json(path: str | Path, what: str, default=REQUIRED, missing: str | None = None):
    """The content of a saved json file. A missing file gives `default` (when given) or a FileNotFoundError ("<what> is missing", or `missing`);
    a file that is not valid JSON gives a ValueError naming it."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return _missing(path, what, missing, default)
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ValueError(f"{what} {path} is not valid JSON ({exc}).") from exc


def read_npz_arrays(path: str | Path, what: str, default=REQUIRED, missing: str | None = None) -> dict:
    """Every array of a saved npz file as {key: array}. A missing file gives `default` (when given) or a FileNotFoundError, as read_json."""
    try:
        data = np.load(path, allow_pickle=True)
    except FileNotFoundError:
        return _missing(path, what, missing, default)
    with data as d:
        return {key: np.asarray(d[key]) for key in d.files}
