"""The segments of the analysed movie the user leaves out of the analysis (utils/qc_utils.py), shown in seconds and frames and set in seconds on the Hypotheses tab.

The movie is not cut: its frame numbers, and so the spikes, the Suite2p traces and the cleaning, stay as they are. The STA job leaves the frames out (no spike window and no
chance copy overlaps them, and the image shifts inside them do not count for the border), so the hypotheses that were run before have to be run again."""
from __future__ import annotations

import re

from app.db import experiment_repository
from app.services import polarity
from app.services.movies import movie_info, pb_movie_path
from utils.file_utils import write_json
from utils.movie_utils import read_frame_rate
from utils.paths_utils import PathKind, path_for
from utils.qc_utils import merge_segments, read_excluded_segments

SEGMENT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*$")  # "120-180", "120.5 - 180", "120 to 180"


def _movie(settings, experiment_id: int) -> tuple[str, float, int]:
    """(experiment folder, frame rate, frames of the analysed astro movie). Raises FileNotFoundError saying which step writes the movie."""
    source = experiment_repository.experiment_source_path(settings, int(experiment_id))
    if source is None:
        raise ValueError("Experiment not found.")
    path = pb_movie_path(source, "astro")
    if not path.exists():
        raise FileNotFoundError("The photobleaching corrected astro movie was not found. The photobleaching correction step writes it: rerun that step for this experiment.")
    fps = read_frame_rate(source)
    return source, fps, int(movie_info(path, fps, polarity.flip_for_kind("astro"))["frames"])


def excluded_segments(settings, experiment_id: int) -> dict:
    """{fps, frames (of the movie), segments: [[start, end) frames], text (the segments as the form takes them: seconds)}."""
    source, fps, frames = _movie(settings, experiment_id)
    segments = read_excluded_segments(source)
    return {"fps": fps, "frames": frames, "segments": segments, "text": ", ".join(f"{round(a / fps, 2):g}-{round(b / fps, 2):g}" for a, b in segments)}


def set_excluded_segments(settings, experiment_id: int, text: str) -> list[list[int]]:
    """Save the segments written as seconds ("120-180, 400-430": start-end, separated by commas, semicolons or lines; empty clears them) for the experiment as the [start, end)
    frames of its analysed movie, merged. Returns them. Raises ValueError for a part that cannot be read, a segment that is empty or ends after the movie."""
    source, fps, frames = _movie(settings, experiment_id)
    segments = []
    for part in (p for p in re.split(r"[,;\n]", text or "") if p.strip()):
        match = SEGMENT.match(part)
        if match is None:
            raise ValueError(f"Cannot read '{part.strip()}': write each segment as start-end in seconds, for example 120-180.")
        start, end = round(float(match.group(1)) * fps), round(float(match.group(2)) * fps)
        if end > frames:
            raise ValueError(f"The segment {part.strip()} ends after the movie, which is {frames / fps:g} s ({frames} frames) long.")
        segments.append([start, end])
    merged = merge_segments(segments, frames)
    write_json(path_for(PathKind.EXCLUDED_SEGMENTS, source), merged)
    return merged
