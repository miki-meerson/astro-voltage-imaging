"""Cell page data: the cropped STA movie of a cell and its single-response stack.

The movie is a saved file (sta_anatomy_bin1.tif in the experiment folder, written by the STA Maps step); this module only reads it. When a
file is missing it raises an error that says which step writes it (utils/output_errors.py); nothing is made here on first view.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from app.db import experiment_repository, run_repository
from app.services import caches, polarity
from app.services.runs import run_dir
from app.services.sta_results import anatomy_clusters_in_crop
from utils import globals_utils as g
from utils.movie_utils import read_frame_rate
from utils.output_errors import missing_output_message
from utils.paths_utils import PathKind, path_for, run_dir_of_sta_dir, sta_dir_of_run
from utils.sta_trace_utils import event_triggered_mean
from utils.suite2p_utils import load_suite2p_context, roi_pixels_in_crop, suite2p_stat_cells, suite2p_trace_dff
from utils.tif_export import saved_run_name


@dataclass(frozen=True)
class CellMovieFiles:
    """Where the saved files of one cell's cropped STA movie are (the latest completed STA run of its experiment)."""
    movie_tif: Path
    anatomy_npz: Path
    source_path: str
    pipeline_neuron_id: int
    sta_run_id: int


def cell_movie_files(settings, neuron_id: int) -> CellMovieFiles:
    """Look up the cell and its experiment's latest completed STA run and return the paths of the saved files (nothing is read from them)."""
    cell = experiment_repository.neuron_with_experiment(settings, neuron_id)
    if cell is None:
        raise ValueError("Cell not found.")
    run = run_repository.latest_completed_run_row(settings, int(cell["experiment_id"]), "sta")
    if run is None:
        raise FileNotFoundError("No completed STA run was found for this experiment: run STA Maps first.")
    sta_dir = sta_dir_of_run(run_dir(settings, int(run["id"]), bool(run["is_test"])))
    number = int(cell["pipeline_neuron_id"])
    return CellMovieFiles(
        movie_tif=path_for(PathKind.STA_MOVIE_TIF, cell["source_path"], cell=number, group=g.ANATOMY_GROUP, bin=1),
        anatomy_npz=path_for(PathKind.ANATOMY_RESULT, sta_dir, cell=number),
        source_path=cell["source_path"],
        pipeline_neuron_id=number,
        sta_run_id=int(run["id"]),
    )


def _check_saved_files(movie_tif: Path, anatomy_npz: Path, cell: int) -> None:
    """The cropped movie and the anatomy result must exist, and the movie must come from the same STA run as the result."""
    run_name = run_dir_of_sta_dir(anatomy_npz.parent.parent).name  # <run>/sta_algorithm_home/sta-analysis/cell_<N>/<file>
    if not anatomy_npz.exists():
        raise FileNotFoundError(
            f"The anatomy STA result of cell {cell} is missing for STA run {run_name} ({anatomy_npz}). The STA Maps step writes it when the "
            "'anatomy' result type is chosen: rerun STA Maps with that result type."
        )
    if not movie_tif.exists():
        raise FileNotFoundError(missing_output_message(f"The cropped STA movie of cell {cell} ({movie_tif.name})", run_name))
    found = saved_run_name(movie_tif)
    if found != run_name:
        raise ValueError(
            f"The cropped STA movie of cell {cell} is from another STA run ({found or 'no run tag'}), not from STA run {run_name}. "
            "Rerun STA Maps for this experiment."
        )


def _read_anatomy_header(anatomy_npz: Path) -> dict:
    """The small keys of the anatomy result that describe its movie (never the permutations)."""
    with np.load(anatomy_npz, allow_pickle=True) as d:
        spatial_bin = int(np.asarray(d["spatial_bin"]).item())
        if spatial_bin != 1:
            raise ValueError(f"The cropped STA movie is shown for anatomy results at spatial bin 1 (this one is bin {spatial_bin}).")
        y0, y1, x0, x1 = (int(v) for v in np.asarray(d["anatomy_crop_bounds"]).ravel())
        return {
            "pre_frames": int(np.asarray(d["pre_frames"]).item()),
            "post_frames": int(np.asarray(d["post_frames"]).item()),
            "peaks": np.asarray(d["peaks"], dtype=np.int32),
            "crop": {"y0": y0, "y1": y1, "x0": x0, "x1": x1, "width": x1 - x0, "height": y1 - y0},
        }


def _roi_outline(source_path, pipeline_neuron_id, crop: dict) -> list:
    """The cell's Suite2p ROI pixels inside the crop (an optional marker: [] when the Suite2p files cannot be read)."""
    try:
        return roi_pixels_in_crop(suite2p_stat_cells(source_path)[int(pipeline_neuron_id)], crop)
    except (OSError, IndexError, KeyError):
        return []


def _neuron_trace(source_path, pipeline_neuron_id, peaks: np.ndarray, pre: int, post: int, rel: np.ndarray) -> dict:
    """The neuron's own dF/F (neuron movie, Suite2p) averaged around the same spikes as the astro movie (an optional marker)."""
    neuron = {"trace": [], "trace_label": "", "trace_peak": None}
    try:
        ctx = load_suite2p_context(path_for(PathKind.SUITE2P_PLANE0, source_path))
        dff = np.asarray(suite2p_trace_dff(int(pipeline_neuron_id), suite2p_ctx=ctx), dtype=float)
    except (OSError, IndexError, KeyError, ValueError):
        return neuron
    ok = peaks[(peaks >= pre) & (peaks + post < len(dff))]
    if ok.size:
        sta_trace = event_triggered_mean(dff, ok, rel)
        neuron["trace"] = sta_trace.round(6).tolist()
        neuron["trace_label"] = "Neuron dF/F (neuron movie), spike-triggered average"
        neuron["trace_peak"] = float(sta_trace[rel == 0][0]) if (rel == 0).any() else None
    return neuron


def sta_crop_stack_payload(movie_tif: Path, anatomy_npz: Path, source_path: str | Path, pipeline_neuron_id: int) -> bytes:
    """The whole cropped STA movie as ONE binary payload: [u32 header length][json header][uint16 frames], plus the markers.

    One request instead of one JPEG per frame; contrast is applied in the browser. Reads the saved tif (movie_tif) and the cell's anatomy
    result (anatomy_npz), both from utils/paths_utils.py; no database. Raises FileNotFoundError (what is missing and which step writes it), ValueError (the movie is from another STA run, or the fps
    cannot be read from Experiment.xml).
    """
    _check_saved_files(movie_tif, anatomy_npz, int(pipeline_neuron_id))
    key = (int(pipeline_neuron_id), str(movie_tif), movie_tif.stat().st_mtime_ns, anatomy_npz.stat().st_mtime_ns, 7)
    hit = caches.STA_STACK.get(key)
    if hit is not None:
        return hit

    head = _read_anatomy_header(anatomy_npz)
    pre, post, crop = head["pre_frames"], head["post_frames"], head["crop"]
    fps = read_frame_rate(source_path)
    movie = np.asarray(tifffile.imread(movie_tif), dtype=np.float32)  # relative change, already in the biological sign (brighter = depolarised)
    lo, hi = (float(v) for v in np.nanpercentile(movie[:: max(1, movie.shape[0] // 40)], [0.05, 99.95]))
    if not hi > lo:
        hi = lo + 1.0
    q = (np.clip((movie - lo) / (hi - lo), 0, 1) * 65535 + 0.5).astype("<u2")
    rel = np.arange(-pre, post + 1, dtype=int)
    neuron = {"pixels": _roi_outline(source_path, int(pipeline_neuron_id), crop),
              **_neuron_trace(source_path, int(pipeline_neuron_id), head["peaks"], pre, post, rel)}
    header = {
        "frames": int(movie.shape[0]), "height": int(movie.shape[1]), "width": int(movie.shape[2]),
        "lo": lo, "hi": hi, "relative_frames": rel.tolist(), "fps": fps,
        "n_peaks": int(head["peaks"].size), "crop": crop, "neuron": neuron, "gevi_flipped": polarity.RAW_SIGN_IS_INVERTED,
        "units": "relative_change", "anatomy_clusters": anatomy_clusters_in_crop(anatomy_npz, crop),
    }
    header_bytes = json.dumps(header).encode("utf-8")
    payload = struct.pack("<I", len(header_bytes)) + header_bytes + q.tobytes()
    caches.STA_STACK.put(key, payload)
    return payload
