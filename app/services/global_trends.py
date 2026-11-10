"""The global trends of an experiment for the Hypotheses tab: the traces of the signals the whole field shares and the maps of where each is strong,
read from the global_trends.npz of the newest completed run of a hypothesis (written by the STA job, utils/global_trends.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.services import caches
from app.services.hypothesis_summary import completed_sta_run
from utils.global_trends import GLOBAL_SERIES
from utils.output_errors import missing_output_message
from utils.paths_utils import PathKind, path_for

MAX_POINTS = 3000  # traces are averaged in blocks down to at most this many points
SERIES_TEXT = {  # name -> (label, unit of the trace)
    GLOBAL_SERIES: ("Field mean intensity", "% change, up = depolarization"),
    "dy": ("Vertical shift (dy)", "px"),
    "dx": ("Horizontal shift (dx)", "px"),
    "shift_mag": ("Shift size", "px"),
    "reg_corr": ("Registration quality", "correlation with the mean image"),
}
MAP_UNIT = "% of the pixel's mean per standard deviation of the series"


def _blocks(values: np.ndarray, size: int) -> np.ndarray:
    """The series averaged in blocks of `size` frames (the last, shorter block too)."""
    edges = np.arange(0, len(values), size)
    return np.add.reduceat(values, edges) / np.diff(np.append(edges, len(values)))


def global_trends(settings, experiment_id: int, step: str) -> dict:
    """{run_id, fps, frames, spatial_bin, border_bins {top, bottom, left, right}, times (s), cuts (s), excluded ([start, end) in s), series: [{name, label, unit, values, map}]} of the newest completed run of
    the hypothesis `step`. map = the amplitude map of the series, as rows of numbers, with its upper limit `map_max` (99th percentile).
    Raises FileNotFoundError saying which step to rerun when the run has no global trends file."""
    run, sta_dir = completed_sta_run(settings, experiment_id, step)
    path = path_for(PathKind.GLOBAL_TRENDS, sta_dir)
    if not path.exists():
        raise FileNotFoundError(missing_output_message("The global trends file", run["id"]) + " Choose the 'all' result type for the run: the trends are computed from its movie.")
    key = (str(path), Path(path).stat().st_mtime_ns)
    hit = caches.GLOBAL_TRENDS.get(key)
    if hit is not None:
        return dict(hit, run_id=int(run["id"]))
    with np.load(path, allow_pickle=True) as z:
        names, traces, amplitude = [str(n) for n in z["names"]], np.asarray(z["traces"], dtype=np.float64), np.asarray(z["amplitude"], dtype=np.float64)
        fps, spatial_bin = float(z["fps"]), int(z["spatial_bin"])
        cuts, margins = [int(c) for c in np.asarray(z["cuts"]).ravel()], [int(m) for m in np.asarray(z["border_margins_px"]).ravel()]
        excluded = np.asarray(z["excluded_segments"], dtype=int).reshape(-1, 2) if "excluded_segments" in z else np.zeros((0, 2), dtype=int)  # a run made before they could be set has none
    frames = traces.shape[1]
    size = max(1, int(np.ceil(frames / MAX_POINTS)))
    times = (np.arange(0, frames, size) + (np.minimum(size, frames - np.arange(0, frames, size)) - 1) / 2) / fps
    series = []
    for name, values, amp in zip(names, traces, amplitude):
        label, unit = SERIES_TEXT[name]
        series.append({
            "name": name, "label": label, "unit": unit, "values": [round(float(v), 4) for v in _blocks(values, size)],
            "map": [[round(float(v), 3) for v in row] for row in amp], "map_max": round(float(np.percentile(amp, 99)), 3),
        })
    result = {
        "frames": frames, "fps": fps, "spatial_bin": spatial_bin, "map_unit": MAP_UNIT, "border_bins": dict(zip(("top", "bottom", "left", "right"), (int(np.ceil(m / spatial_bin)) for m in margins))),
        "times": [round(float(t), 3) for t in times], "cuts": [round(c / fps, 3) for c in cuts], "excluded": [[round(a / fps, 3), round(b / fps, 3)] for a, b in excluded], "series": series,
    }
    caches.GLOBAL_TRENDS.put(key, result)
    return dict(result, run_id=int(run["id"]))
