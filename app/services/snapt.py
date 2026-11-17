"""The SNAPT propagation maps for the pages (the SNAPT step, utils/snapt_run.py): what the newest completed run saved for one cell, as plain numbers for the page to draw. A page only reads what
the step saved."""
from __future__ import annotations

import base64

import numpy as np

from app.db import run_repository
from app.services.cell_views import binned_roi_pixels
from app.services.runs import run_dir
from utils import globals_utils as g
from utils.file_utils import read_json
from utils.movie_utils import read_frame_rate
from utils.paths_utils import PathKind, path_for
from utils.snapt_run import load_cell_snapt

MAPS = {"delay": ("delay_seconds", 4), "width": ("width", 3), "amplitude": ("amplitude", 3), "r2": ("r2", 3)}  # name on the page: (saved array, decimals)


def cell_snapt(settings, experiment_id: int, cell: int, source_path: str, movie_sign: str | None = None) -> dict:
    """The maps of one cell from the newest completed run of the step: {run_id, fps, spatial_bin, height, width, roi ([[y, x], ...] in the pixels of the maps), components, clusters ([{sign, type ('all' or 'anatomy'), y, x: the pixels of its footprint on the grid of the maps}, ...], those the kernels were made from), signs: {sign:
    {n_clusters, origin ([y, x] in the pixels of the maps: the centre of the clusters' footprints), kernel (values 0..1), kernel_seconds, maps: {delay (seconds, later is positive), width (1 = as wide as the kernel), amplitude, r2: flat lists of the H x W map}, accepted (flat
    list of 0 and 1), accepted_fraction, converged_fraction, movie (only for `movie_sign`): {seconds, frames, data (base64 of the uint8 values, frames x H x W)}}}}. Raises FileNotFoundError
    naming the step when it was not run, or giving the reason when the cell was left out."""
    run = run_repository.latest_run(settings, int(experiment_id), g.SNAPT_STEP)
    if run is None:
        raise FileNotFoundError("SNAPT has not been run for this experiment: start the SNAPT step (Overview tab).")
    out = run_dir(settings, int(run["id"]), bool(run["is_test"]))
    path = path_for(PathKind.SNAPT_CELL, out, cell=int(cell))
    if not path.exists():
        reason = read_json(path_for(PathKind.SNAPT_SUMMARY, out), "The SNAPT summary", default={}).get("skipped", {}).get(str(int(cell)))
        if reason:
            raise FileNotFoundError(f"This cell was left out of SNAPT (run {run['id']}): {reason}")
    d = load_cell_snapt(path)
    fps = read_frame_rate(source_path)
    height, width = next(iter(d["results"].values()))["amplitude"].shape
    signs = {}
    for sign, r in d["results"].items():
        signs[sign] = {"n_clusters": int(r["n_clusters"]), "origin": [round(float(v), 2) for v in r["origin"]], "kernel": [round(float(v), 4) for v in r["kernel"]], "kernel_seconds": [round(float(v) / fps, 4) for v in r["kernel_rels"]],
                       "maps": {name: [round(float(v), decimals) for v in r[key].ravel()] for name, (key, decimals) in MAPS.items()}, "accepted": r["accepted"].ravel().astype(int).tolist(),
                       "accepted_fraction": round(float(r["accepted"].mean()), 4), "converged_fraction": round(float(r["converged"].mean()), 4)}
        if sign == movie_sign:
            signs[sign]["movie"] = {"seconds": [round(float(v) / fps, 4) for v in r["movie_rels"]], "frames": int(r["movie"].shape[0]), "data": base64.b64encode(r["movie"].tobytes()).decode("ascii")}
    clusters = [{"sign": sign, "type": kind, "y": ys.tolist(), "x": xs.tolist()} for mask, sign, kind in zip(d["clusters"]["masks"], d["clusters"]["signs"], d["clusters"]["types"]) for ys, xs in [np.nonzero(mask)]]
    return {"run_id": int(run["id"]), "fps": fps, "spatial_bin": d["spatial_bin"], "height": int(height), "width": int(width), "components": d["components"],
            "roi": binned_roi_pixels(source_path, cell, d["spatial_bin"], int(height), int(width)), "clusters": clusters, "signs": signs}
