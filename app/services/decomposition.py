"""The PCA / ICA of the STA movies for the pages (the decomposition step, utils/decomposition_run.py): the status of the step for an experiment, and what the newest completed run saved
for one cell, as plain numbers for the page to draw. A page only reads what the step saved."""
from __future__ import annotations

import numpy as np

from app.db import run_repository
from app.services.cell_views import binned_roi_pixels
from app.services.runs import run_dir
from utils import globals_utils as g
from utils.decomposition_run import load_cell_decomposition
from utils.file_utils import read_json
from utils.movie_utils import read_frame_rate
from utils.paths_utils import PathKind, path_for

METHODS = ("pca", "ica")


def decomposition_status(settings, experiment_id: int) -> dict:
    """{experiment_id, status (of the newest run of the step, or 'not_run'), run_id} of an experiment."""
    newest = run_repository.latest_run(settings, int(experiment_id), g.DECOMPOSITION_STEP, status=None)
    return {"experiment_id": int(experiment_id), "status": newest["status"] if newest else "not_run", "run_id": int(newest["id"]) if newest else None}


def _map_rows(maps: np.ndarray) -> tuple[list[list[float]], list[float]]:
    """([each map as a flat list of values between -1 and 1], [the largest absolute value of each map]): a map is its list times its scale. Keeps the size of the answer small."""
    flat = maps.reshape(maps.shape[0], -1).astype(np.float64)
    scales = np.maximum(np.abs(flat).max(axis=1), 1e-30)
    return [[round(float(v), 3) for v in row] for row in flat / scales[:, None]], [float(s) for s in scales]


def cell_decomposition(settings, experiment_id: int, cell: int, source_path: str) -> dict:
    """The decomposition of one cell from the newest completed run of the step: {run_id, fps, rels (frames from the spike), spatial_bin, height, width, roi ([[y, x], ...] in the pixels of
    the maps), n_peaks, n_components,
    variance_explained (shares, the first ones), pca / ica: {maps (flat lists, values -1..1), scales (map = list x scale), courses (one list of values per component), variance (ICA: the share
    of the movie each component accounts for)}, ica_converged, seed}. Raises FileNotFoundError naming the step when it was not run, or giving the reason when the cell was left out."""
    run = run_repository.latest_run(settings, int(experiment_id), g.DECOMPOSITION_STEP)
    if run is None:
        raise FileNotFoundError("The PCA / ICA has not been run for this experiment: start the decomposition step (Overview tab).")
    out = run_dir(settings, int(run["id"]), bool(run["is_test"]))
    path = path_for(PathKind.DECOMPOSITION_CELL, out, cell=int(cell))
    if not path.exists():
        summary = read_json(path_for(PathKind.DECOMPOSITION_SUMMARY, out), "The decomposition summary", default={})
        reason = summary.get("skipped", {}).get(str(int(cell)))
        if reason:
            raise FileNotFoundError(f"This cell was left out of the PCA / ICA (run {run['id']}): {reason}.")
    d = load_cell_decomposition(path)
    height, width = d["pca_maps"].shape[1:]
    result = {"run_id": int(run["id"]), "fps": read_frame_rate(source_path), "rels": [int(v) for v in d["rels"]], "spatial_bin": int(d["spatial_bin"]), "height": int(height), "width": int(width),
              "roi": binned_roi_pixels(source_path, cell, int(d["spatial_bin"]), int(height), int(width)), "n_peaks": int(d["n_peaks"]),
              "n_components": int(d["n_components"]), "variance_explained": [round(float(v), 5) for v in d["variance_explained"]], "ica_converged": bool(d["ica_converged"]), "seed": int(d["seed"]),
              "baseline_window": [int(v) for v in d["baseline_window"]]}
    for method in METHODS:
        maps, scales = _map_rows(d[f"{method}_maps"])
        result[method] = {"maps": maps, "scales": scales, "courses": [[round(float(v), 5) for v in d[f"{method}_courses"][:, k]] for k in range(int(d["n_components"]))],
                          "variance": [round(float(v), 5) for v in d["ica_variance"]] if method == "ica" else [round(float(v), 5) for v in d["variance_explained"][:int(d["n_components"])]]}
    return result
