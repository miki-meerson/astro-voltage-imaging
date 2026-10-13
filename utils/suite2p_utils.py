from pathlib import Path

import numpy as np

from utils.paths_utils import PathKind, path_for
from utils.sta_trace_utils import pixels_in_crop


def read_suite2p_traces(suite2p_path, cell_mask, neuropil_correction=False):
    F = np.load(path_for(PathKind.SUITE2P_F, suite2p_path), allow_pickle=True)
    F_cells = F[cell_mask, :]

    if neuropil_correction:
        Fneu = np.load(path_for(PathKind.SUITE2P_FNEU, suite2p_path), allow_pickle=True)
        Fneu_cells = Fneu[cell_mask, :]

        neuropil_coeff = 0.7
        F_corr = F_cells - neuropil_coeff * Fneu_cells
    else:
        F_corr = F_cells

    F_norm = (F_corr - np.min(F_corr, axis=1, keepdims=True)) / (np.ptp(F_corr, axis=1, keepdims=True) + 1e-12)
    return F_cells, F_norm


def load_cell_selection(suite2p_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(cell_mask, stat_cells): the accepted cells of a Suite2p plane0 folder and their ROI stats. The only place iscell.npy and stat.npy are read."""
    cell_mask = np.load(path_for(PathKind.SUITE2P_ISCELL, suite2p_path), allow_pickle=True)[:, 0].astype(bool)
    return cell_mask, np.load(path_for(PathKind.SUITE2P_STAT, suite2p_path), allow_pickle=True)[cell_mask]


def load_suite2p_context(suite2p_path: str | Path, cell_selection: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """Accepted cells, their ROI stats and the fluorescence. Pass an already loaded cell_selection to avoid reading iscell/stat again."""
    suite2p_path = Path(suite2p_path)
    cell_mask, stat_cells = cell_selection if cell_selection is not None else load_cell_selection(suite2p_path)
    f_raw, f_norm = read_suite2p_traces(suite2p_path, cell_mask)
    return {
        "cell_mask": cell_mask,
        "stat_cells": stat_cells,
        "F": f_raw,
        "F_norm": f_norm,
        "n_cells": int(cell_mask.sum()),
    }


def suite2p_stat_cells(source_dir: str | Path) -> np.ndarray:
    """Suite2p ROI stats of the accepted cells only (cell ids are indices into this array); source_dir is the experiment folder."""
    return load_cell_selection(path_for(PathKind.SUITE2P_PLANE0, source_dir))[1]


def roi_pixels(stat_cell) -> tuple[np.ndarray, np.ndarray]:
    """(ypix, xpix) of one ROI, full-resolution pixels."""
    return np.asarray(stat_cell["ypix"], dtype=int), np.asarray(stat_cell["xpix"], dtype=int)


def roi_center(stat_cell) -> tuple[float, float]:
    """Centre (y, x) of one ROI as the mean of its pixels, in full-resolution pixels."""
    ys, xs = roi_pixels(stat_cell)
    return float(ys.mean()), float(xs.mean())


def roi_pixels_in_crop(stat_cell, crop: dict) -> list:
    """The ROI pixels [[y, x], ...] that fall inside a crop box {y0, x0, height, width}, in the crop's own coordinates."""
    return pixels_in_crop(*roi_pixels(stat_cell), crop)


def suite2p_trace_dff(
    neuron_id: int,
    *,
    suite2p_ctx: dict | None = None,
    suite2p_path=None,
    F_raw: np.ndarray | None = None,
    baseline_percentile: float = 10.0,
) -> np.ndarray:
    """Convert a raw Suite2p fluorescence trace to dF/F using a robust per-neuron baseline."""
    if F_raw is None:
        if suite2p_ctx is None:
            if suite2p_path is None:
                raise ValueError("Pass suite2p_ctx, suite2p_path, or F_raw to suite2p_trace_dff.")
            suite2p_ctx = load_suite2p_context(suite2p_path)
        F_raw = np.asarray(suite2p_ctx["F"], dtype=np.float32)
    else:
        F_raw = np.asarray(F_raw, dtype=np.float32)
    trace = np.asarray(F_raw[int(neuron_id)], dtype=np.float32).ravel()
    f0 = float(np.nanpercentile(trace, baseline_percentile))
    if not np.isfinite(f0) or abs(f0) <= 1e-6:
        f0 = float(np.nanmedian(trace))
    if not np.isfinite(f0) or abs(f0) <= 1e-6:
        return np.full_like(trace, np.nan, dtype=np.float32)
    return ((trace - f0) / (abs(f0) + 1e-6)).astype(np.float32)


def neuron_centroid_binned(cell_id, spatial_bin, map_shape, suite2p_ctx):
    s = suite2p_ctx["stat_cells"][cell_id]
    if "med" in s and len(s["med"]) >= 2:
        y, x = float(s["med"][0]), float(s["med"][1])
    else:
        y, x = roi_center(s)
    yb = int(np.clip(np.floor(y / spatial_bin), 0, map_shape[0] - 1))
    xb = int(np.clip(np.floor(x / spatial_bin), 0, map_shape[1] - 1))
    return yb, xb


def get_neuron_outline(neuron_id, suite2p_path, image_shape, spatial_bin):
    suite2p_ctx = load_suite2p_context(suite2p_path)
    return suite2p_roi_mask(neuron_id, suite2p_ctx["stat_cells"], image_shape, spatial_bin=spatial_bin)


def suite2p_roi_mask(neuron_id, stat_cells, image_shape, spatial_bin=1, exclude_overlap=False):
    s = stat_cells[int(neuron_id)]
    ypix, xpix = roi_pixels(s)
    if exclude_overlap and "overlap" in s:
        keep = ~np.asarray(s["overlap"], dtype=bool)
        ypix = ypix[keep]
        xpix = xpix[keep]

    map_h, map_w = image_shape
    yb = np.clip(np.floor_divide(ypix, int(spatial_bin)), 0, map_h - 1)
    xb = np.clip(np.floor_divide(xpix, int(spatial_bin)), 0, map_w - 1)

    neuron_mask2d = np.zeros((map_h, map_w), dtype=bool)
    neuron_mask2d[yb, xb] = True

    return neuron_mask2d


def add_neuron_outlines(ax, cell_ids, stat_cells, image_shape, color="yellow", linewidth=1.0, spatial_bin=1, alpha=1.0):
    for cell_id in cell_ids:
        mask = suite2p_roi_mask(
            int(cell_id),
            stat_cells,
            image_shape,
            spatial_bin=spatial_bin,
            exclude_overlap=True,
        )
        if mask.any():
            ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=linewidth, alpha=alpha)
