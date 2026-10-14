"""Reader of a number from the photobleaching QC file in <experiment>/pipeline_results/qc/."""
from __future__ import annotations

from pathlib import Path

from utils.file_utils import read_json
from utils.paths_utils import PathKind, path_for


def read_trimmed_frames(experiment_dir: str | Path) -> int:
    """Frames the photobleaching step dropped from the start of the astro movie (0 when it recorded none or wrote no QC file)."""
    metrics = read_json(path_for(PathKind.PB_QC_METRICS, experiment_dir, pb_kind="astro"), "The photobleaching QC file", default={})
    return int(metrics.get("trimmed", 0))
