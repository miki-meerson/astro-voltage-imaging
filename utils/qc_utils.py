"""Readers of the QC files in <experiment>/pipeline_results/qc/: the frames the photobleaching step trimmed, and the segments the user excluded from the analysis."""
from __future__ import annotations

from pathlib import Path

from utils.file_utils import read_json
from utils.paths_utils import PathKind, path_for


def read_trimmed_frames(experiment_dir: str | Path) -> int:
    """Frames the photobleaching step dropped from the start of the astro movie (0 when it recorded none or wrote no QC file)."""
    metrics = read_json(path_for(PathKind.PB_QC_METRICS, experiment_dir, pb_kind="astro"), "The photobleaching QC file", default={})
    return int(metrics.get("trimmed", 0))


def merge_segments(segments, n_frames: int | None = None) -> list[list[int]]:
    """The [start, end) frame ranges, sorted, with the ones that overlap or touch merged. Raises ValueError for a range that does not have
    0 <= start < end (and end <= n_frames when it is given)."""
    cleaned = []
    for segment in segments or []:
        start, end = (int(v) for v in segment)
        if not 0 <= start < end or (n_frames is not None and end > n_frames):
            raise ValueError(f"The segment [{start}, {end}) must lie inside the movie: 0 <= start < end" + (f" <= {n_frames} frames." if n_frames is not None else "."))
        cleaned.append([start, end])
    merged: list[list[int]] = []
    for start, end in sorted(cleaned):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def read_excluded_segments(experiment_dir: str | Path) -> list[list[int]]:
    """The segments the user excluded from the analysis, [start, end) frames of the analysed movie, merged ([] when there are none). The movie is not cut: its frame numbers,
    and so the spikes, the Suite2p traces and the cleaning, stay as they are; the STA leaves the frames out."""
    return merge_segments(read_json(path_for(PathKind.EXCLUDED_SEGMENTS, experiment_dir), "The excluded-segments file", default=[]))
