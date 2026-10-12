"""Cell page data: the cropped STA movie (the anatomy STA result of the STA run, saved once as a tif) and its single-response stack."""
from __future__ import annotations

import json
from pathlib import Path
import re
import threading

import numpy as np

from app.db.core import connect
from app.services import caches, polarity
from app.services.movies import experiment_fps
from app.services.run_views import latest_completed_run
from app.services.runs import run_dir
from app.services.spikes import suite2p_plane0, suite2p_stat_cells
from app.services.sta_results import anatomy_clusters_in_crop
from utils.tif_export import fiji_cell_dir, sta_note, write_imagej_stack

def latest_anatomy_npz(settings, conn, experiment_id: int, pipeline_neuron_id: str | int) -> Path:
    """The anatomy STA result of this cell in the latest completed STA run: it holds the cropped, full-resolution STA movie."""
    run = latest_completed_run(conn, experiment_id, "sta")
    if run is None:
        raise FileNotFoundError("No completed STA run was found.")
    path = (
        run_dir(settings, int(run["id"]), bool(run["is_test"]))
        / "sta_algorithm_home"
        / "sta-analysis"
        / f"cell_{int(pipeline_neuron_id)}"
        / "astro_sta_cluster_permutation_anatomy.npz"
    )
    if not path.exists():
        raise FileNotFoundError(f"This cell has no anatomy STA result in STA run {int(run['id'])} (the run needs the 'anatomy' result type): {path}")
    return path


def sta_crop_path(source_path: str | Path, pipeline_neuron_id: str | int, spatial_bin: int = 1) -> Path:
    """The ONE saved copy of the cropped STA movie: sta_anatomy_bin<B>.tif in the experiment folder (Fiji-readable, biological sign).
    The STA job writes it when it creates the movie; for older runs the app writes it once from the run's anatomy result."""
    return fiji_cell_dir(source_path, pipeline_neuron_id) / f"sta_anatomy_bin{int(spatial_bin)}.tif"


def _saved_run_name(tif: Path) -> str | None:
    """Name of the STA run folder a saved tif was made from (kept in its ImageJ info text); None when missing or unreadable."""
    try:
        import tifffile

        with tifffile.TiffFile(tif) as t:
            info = (t.imagej_metadata or {}).get("Info", "")
    except Exception:
        return None
    m = re.search(r"run=(\S+)", str(info))
    return m.group(1) if m else None


_BUILD_LOCKS: dict[int, threading.Lock] = {}
_BUILD_LOCKS_GUARD = threading.Lock()


def _build_lock(neuron_id: int) -> threading.Lock:
    """One lock per cell: concurrent requests for the same cell wait for one conversion instead of each doing it."""
    with _BUILD_LOCKS_GUARD:
        return _BUILD_LOCKS.setdefault(int(neuron_id), threading.Lock())


def ensure_saved_cropped_sta_movie(settings, neuron_id: int) -> dict:
    """The cropped STA movie of a cell and where it is saved. It is the observed STA movie the STA job already computed for the cell's
    anatomy result (the same grid as the clusters, relative change vs the pre-spike baseline, biological sign): a small file, not a rebuild
    from the astro movie. Written to the experiment folder once (older runs) or by the STA job itself (new runs)."""
    with connect(settings) as conn:
        n = conn.execute(
            """
            SELECT n.*,e.source_path,e.canonical_id,e.id AS experiment_id
            FROM neurons n
            JOIN experiments e ON e.id=n.experiment_id
            WHERE n.id=?
            """,
            (neuron_id,),
        ).fetchone()
        if n is None:
            raise ValueError("Cell not found.")
        anat = latest_anatomy_npz(settings, conn, int(n["experiment_id"]), n["pipeline_neuron_id"])
    key = (int(neuron_id), str(anat), int(anat.stat().st_mtime_ns))
    cached = caches.ANATOMY_CROP_INFO.get(key)
    if cached is not None:
        return dict(cached)
    with _build_lock(neuron_id):
        cached = caches.ANATOMY_CROP_INFO.get(key)
        if cached is not None:
            return dict(cached)
        try:
            fps = experiment_fps(n["source_path"])
        except Exception:
            fps = 1.0
        with np.load(anat, allow_pickle=True) as d:
            spatial_bin = int(np.asarray(d["spatial_bin"]).item()) if "spatial_bin" in d.files else 1
            if spatial_bin != 1:
                raise ValueError(f"The cropped STA movie is shown for anatomy results at spatial bin 1 (this one is bin {spatial_bin}).")
            pre_frames = int(np.asarray(d["pre_frames"]).item())
            post_frames = int(np.asarray(d["post_frames"]).item())
            peaks = np.asarray(d["peaks"], dtype=np.int32)
            y0, y1, x0, x1 = (int(v) for v in np.asarray(d["anatomy_crop_bounds"]).ravel())
            run_name = anat.parents[3].name  # <run folder>/sta_algorithm_home/sta-analysis/cell_<N>/<file>
            out_tif = sta_crop_path(n["source_path"], n["pipeline_neuron_id"], spatial_bin)
            if _saved_run_name(out_tif) != run_name:  # missing, or made from another STA run
                movie = np.asarray(d["observed_sta_movie"], dtype=np.float32)
                write_imagej_stack(out_tif, movie, fps, pre_frames, sta_note(n["pipeline_neuron_id"], peaks.size, "anatomy spatial bin 1 (relative change)", run=run_name))
        info = {
            "available": True,
            "movie_path": out_tif,
            "source_sta_npz": str(anat),
            "pipeline_neuron_id": int(n["pipeline_neuron_id"]),
            "frames": int(pre_frames + post_frames + 1),
            "fps": fps,
            "relative_frames": np.arange(-pre_frames, post_frames + 1, dtype=int).tolist(),
            "pre_frames": pre_frames,
            "post_frames": post_frames,
            "n_peaks": int(peaks.size),
            "crop": {"y0": y0, "y1": y1, "x0": x0, "x1": x1, "width": x1 - x0, "height": y1 - y0},
            "spatial_bin": spatial_bin,
            "units": "relative_change",
        }
        caches.ANATOMY_CROP_INFO.put(key, dict(info))
        return info


def suite2p_roi_pixels(source_path: str | Path, pipeline_neuron_id: str | int) -> tuple[np.ndarray, np.ndarray]:
    roi = suite2p_stat_cells(source_path)[int(pipeline_neuron_id)]
    return np.asarray(roi["ypix"], dtype=int), np.asarray(roi["xpix"], dtype=int)


def sta_crop_stack_payload(settings, neuron_id: int):
    """The whole cropped STA movie as ONE binary payload (bytes): [u32 header length][json header][uint16 frames], plus the markers.

    One request instead of one JPEG per frame; contrast is applied in the browser.
    """
    import struct

    import tifffile

    info = ensure_saved_cropped_sta_movie(settings, neuron_id)
    path = Path(info["movie_path"])
    anat = Path(info["source_sta_npz"])
    key = (int(neuron_id), str(path), path.stat().st_mtime_ns, anat.stat().st_mtime_ns if anat.exists() else 0, 6)
    hit = caches.STA_STACK.get(key)
    if hit is not None:
        return hit

    movie = np.asarray(tifffile.imread(path), dtype=np.float32)  # relative change, already in the biological sign (brighter = depolarised)
    lo, hi = (float(v) for v in np.nanpercentile(movie[:: max(1, movie.shape[0] // 40)], [0.05, 99.95]))
    if not hi > lo:
        hi = lo + 1.0
    q = np.clip((movie - lo) / (hi - lo), 0, 1)
    q = (q * 65535 + 0.5).astype("<u2")
    crop = info["crop"]
    rel = np.asarray(info["relative_frames"], dtype=int)
    with connect(settings) as conn:
        src = conn.execute("SELECT e.source_path FROM neurons n JOIN experiments e ON e.id=n.experiment_id WHERE n.id=?", (neuron_id,)).fetchone()["source_path"]
    neuron = {"pixels": [], "trace": [], "trace_label": "", "trace_peak": None}
    try:  # ROI outline (optional marker only)
        ys, xs = suite2p_roi_pixels(src, info["pipeline_neuron_id"])
        ys, xs = ys - int(crop["y0"]), xs - int(crop["x0"])
        keep = (ys >= 0) & (ys < movie.shape[1]) & (xs >= 0) & (xs < movie.shape[2])
        if keep.any():
            neuron["pixels"] = np.stack([ys[keep], xs[keep]], axis=1).tolist()
    except Exception:
        pass
    try:  # the neuron's own dF/F (neuron movie, Suite2p), averaged around the same spikes as the astro movie
        from utils.suite2p_utils import load_suite2p_context
        from utils.spikes_utils import _suite2p_trace_dff

        ctx = load_suite2p_context(suite2p_plane0(src))
        dff = np.asarray(_suite2p_trace_dff(int(info["pipeline_neuron_id"]), suite2p_ctx=ctx), dtype=float)
        with np.load(anat, allow_pickle=True) as d:
            peaks = np.asarray(d["peaks"], dtype=int)
        pre, post = int(info["pre_frames"]), int(info["post_frames"])
        ok = peaks[(peaks >= pre) & (peaks + post < len(dff))]
        if ok.size:
            sta_trace = np.mean([dff[q - pre: q + post + 1] for q in ok], axis=0)
            neuron["trace"] = sta_trace.round(6).tolist()
            neuron["trace_label"] = "Neuron dF/F (neuron movie), spike-triggered average"
            neuron["trace_peak"] = float(sta_trace[rel == 0][0]) if (rel == 0).any() else None
    except Exception:
        pass
    header = {
        "frames": int(movie.shape[0]), "height": int(movie.shape[1]), "width": int(movie.shape[2]),
        "lo": lo, "hi": hi, "relative_frames": rel.tolist(), "fps": float(info["fps"]),
        "n_peaks": int(info["n_peaks"]), "crop": crop, "neuron": neuron, "gevi_flipped": polarity.RAW_SIGN_IS_INVERTED,
        "units": info["units"], "anatomy_clusters": anatomy_clusters_in_crop(anat, crop),
    }
    hb = json.dumps(header).encode("utf-8")
    payload = struct.pack("<I", len(hb)) + hb + q.tobytes()
    caches.STA_STACK.put(key, payload)
    return payload
