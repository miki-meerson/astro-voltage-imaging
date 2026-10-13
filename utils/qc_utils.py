"""Readers of the external / photobleaching QC files in <experiment>/pipeline_results/qc/. The one place these files are opened."""
from __future__ import annotations

import json
from pathlib import Path

from utils.paths_utils import PathKind, path_for


def read_bad_segments(experiment_dir: str | Path) -> list:
    """[start, end) frame ranges removed outside the app (numbering before the removal); [] when the experiment has none (no file)."""
    path = path_for(PathKind.BAD_SEGMENTS, experiment_dir)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except ValueError as exc:
        raise ValueError(f"The bad-segments file {path} is not valid JSON ({exc}).") from exc


def read_pb_qc_metrics(experiment_dir: str | Path, pb_kind: str) -> dict:
    """Photobleaching QC numbers of `pb_kind` (astro / neuron); {} when the file was not written (the photobleaching step produces it)."""
    path = path_for(PathKind.PB_QC_METRICS, experiment_dir, pb_kind=pb_kind)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ValueError(f"The photobleaching QC file {path} is not valid JSON ({exc}); rerun the photobleaching correction step.") from exc


def read_trimmed_frames(experiment_dir: str | Path) -> int:
    """Frames the photobleaching step dropped from the start of the astro movie (0 when it recorded none)."""
    return int(read_pb_qc_metrics(experiment_dir, "astro").get("trimmed", 0))
