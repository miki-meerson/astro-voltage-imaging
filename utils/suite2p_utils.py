from pathlib import Path

import numpy as np

from utils.paths_utils import PathKind, path_for


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


def load_suite2p_context(suite2p_path: str | Path) -> dict:
    suite2p_path = Path(suite2p_path)
    iscell = np.load(path_for(PathKind.SUITE2P_ISCELL, suite2p_path), allow_pickle=True)
    cell_mask = iscell[:, 0].astype(bool)
    stat = np.load(path_for(PathKind.SUITE2P_STAT, suite2p_path), allow_pickle=True)
    stat_cells = stat[cell_mask]
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
    plane0 = path_for(PathKind.SUITE2P_PLANE0, source_dir)
    iscell = np.load(path_for(PathKind.SUITE2P_ISCELL, plane0), allow_pickle=True)[:, 0].astype(bool)
    return np.load(path_for(PathKind.SUITE2P_STAT, plane0), allow_pickle=True)[iscell]


def roi_center(stat_cell) -> tuple[float, float]:
    """Centre (y, x) of one ROI as the mean of its pixels, in full-resolution pixels."""
    return float(np.mean(stat_cell["ypix"])), float(np.mean(stat_cell["xpix"]))


def neuron_centroid_binned(cell_id, spatial_bin, map_shape, suite2p_ctx):
    s = suite2p_ctx["stat_cells"][cell_id]
    if "med" in s and len(s["med"]) >= 2:
        y, x = float(s["med"][0]), float(s["med"][1])
    else:
        y = float(np.mean(s["ypix"]))
        x = float(np.mean(s["xpix"]))
    yb = int(np.clip(np.floor(y / spatial_bin), 0, map_shape[0] - 1))
    xb = int(np.clip(np.floor(x / spatial_bin), 0, map_shape[1] - 1))
    return yb, xb


def get_neuron_outline(neuron_id, suite2p_path, image_shape, spatial_bin):
    suite2p_ctx = load_suite2p_context(suite2p_path)
    return suite2p_roi_mask(neuron_id, suite2p_ctx["stat_cells"], image_shape, spatial_bin=spatial_bin)


def suite2p_roi_mask(neuron_id, stat_cells, image_shape, spatial_bin=1, exclude_overlap=False):
    s = stat_cells[int(neuron_id)]
    ypix = np.asarray(s["ypix"], dtype=int)
    xpix = np.asarray(s["xpix"], dtype=int)
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
