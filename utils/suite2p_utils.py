import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_suite2p_anatomy(suite2p_path, n_cells):
    stat = np.load(suite2p_path + 'stat.npy', allow_pickle=True)
    ops = np.load(suite2p_path + 'ops.npy', allow_pickle=True)
    ops = ops.item()

    im = np.zeros((ops['Ly'], ops['Lx']))

    for n in range(0, n_cells):
        ypix = stat[n]['ypix'][~stat[n]['overlap']]
        xpix = stat[n]['xpix'][~stat[n]['overlap']]
        im[ypix, xpix] = n + 1

    plt.figure(figsize=(6, 6))
    plt.imshow(im)
    plt.show(block=True)


def get_roi_image(cell_id, suite2p_path):
    iscell = np.load(suite2p_path + 'iscell.npy', allow_pickle=True)
    stat = np.load(suite2p_path + 'stat.npy', allow_pickle=True)
    ops = np.load(suite2p_path + 'ops.npy', allow_pickle=True).item()
    cell_mask = iscell[:, 0].astype(bool)
    stat_cells = stat[cell_mask]

    roi = stat_cells[cell_id]
    im = np.zeros((ops['Ly'], ops['Lx']))
    ypix = roi['ypix'][~roi['overlap']]
    xpix = roi['xpix'][~roi['overlap']]
    im[ypix, xpix] = cell_id + 1

    return im


def read_suite2p_traces(suite2p_path, cell_mask, neuropil_correction=False):
    F = np.load(suite2p_path + 'F.npy', allow_pickle=True)
    F_cells = F[cell_mask, :]

    if neuropil_correction:
        Fneu = np.load(suite2p_path + 'Fneu.npy', allow_pickle=True)
        Fneu_cells = Fneu[cell_mask, :]

        neuropil_coeff = 0.7
        F_corr = F_cells - neuropil_coeff * Fneu_cells
    else:
        F_corr = F_cells

    F_norm = (F_corr - np.min(F_corr, axis=1, keepdims=True)) / (np.ptp(F_corr, axis=1, keepdims=True) + 1e-12)
    return F_cells, F_norm


def load_suite2p_context(suite2p_path: str | Path) -> dict:
    suite2p_path = Path(suite2p_path)
    iscell = np.load(suite2p_path / "iscell.npy", allow_pickle=True)
    cell_mask = iscell[:, 0].astype(bool)
    stat = np.load(suite2p_path / "stat.npy", allow_pickle=True)
    stat_cells = stat[cell_mask]
    f_raw, f_norm = read_suite2p_traces(f"{suite2p_path.as_posix()}/", cell_mask)
    return {
        "cell_mask": cell_mask,
        "stat_cells": stat_cells,
        "F": f_raw,
        "F_norm": f_norm,
        "n_cells": int(cell_mask.sum()),
    }


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
    s = suite2p_ctx["stat_cells"][neuron_id]
    ypix = np.asarray(s["ypix"], dtype=int)
    xpix = np.asarray(s["xpix"], dtype=int)

    map_h, map_w = image_shape
    yb = np.clip(np.floor_divide(ypix, spatial_bin), 0, map_h - 1)
    xb = np.clip(np.floor_divide(xpix, spatial_bin), 0, map_w - 1)

    neuron_mask2d = np.zeros((map_h, map_w), dtype=bool)
    neuron_mask2d[yb, xb] = True

    return neuron_mask2d