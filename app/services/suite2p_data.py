"""Suite2p data of an experiment for the web app: read from disk once, then kept for the last few experiments that were opened (rolling
caches in caches.py; the oldest experiment is dropped when a new one comes in). iscell.npy and stat.npy are read once per experiment (the
cell selection) and shared by the stats and the context. The data is reread after the user confirms the Suite2p extraction
(`forget_suite2p`, called by confirm_suite2p). Cluster jobs and adapter processes read the files themselves (utils/suite2p_utils.py).
A failed read is not cached."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.services import caches
from utils.paths_utils import PathKind, path_for
from utils.suite2p_utils import load_cell_selection, load_suite2p_context


def _cell_selection(source_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    key = str(source_path)
    selection = caches.SUITE2P_STAT.get(key)
    if selection is None:
        selection = load_cell_selection(path_for(PathKind.SUITE2P_PLANE0, source_path))
        caches.SUITE2P_STAT.put(key, selection)
    return selection


def experiment_stat_cells(source_path: str | Path) -> np.ndarray:
    """Suite2p ROI stats of the accepted cells (cell ids are indices into this array)."""
    return _cell_selection(source_path)[1]


def experiment_suite2p_context(source_path: str | Path) -> dict:
    """The Suite2p context (accepted cells, ROI stats, fluorescence) of an experiment. Shared between requests: never modify it."""
    key = str(source_path)
    context = caches.SUITE2P_CONTEXT.get(key)
    if context is None:
        context = load_suite2p_context(path_for(PathKind.SUITE2P_PLANE0, source_path), _cell_selection(source_path))
        caches.SUITE2P_CONTEXT.put(key, context)
    return context


def forget_suite2p(source_path: str | Path) -> None:
    """Drop what is kept of this experiment so the next use rereads the Suite2p files."""
    caches.SUITE2P_STAT.discard(str(source_path))
    caches.SUITE2P_CONTEXT.discard(str(source_path))
