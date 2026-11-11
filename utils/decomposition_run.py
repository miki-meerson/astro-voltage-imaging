"""The decomposition step of the pipeline: the PCA and ICA (utils/sta_decomposition.py) of the whole-field STA movie of every cell of a finished STA run, saved one file per cell in the
run folder of the step (DECOMPOSITION_CELL) with a summary of what was done and which cells were left out and why (DECOMPOSITION_SUMMARY). The cluster job (tools/sta_decomposition.py)
calls `decompose_sta_run`; the pages read the files back with `load_cell_decomposition` (app/services/decomposition.py). No Flask, no app imports."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from utils import globals_utils as g
from utils.cluster_result import read_result_header
from utils.file_utils import write_json
from utils.movie_utils import read_frame_rate
from utils.paths_utils import PathKind, cell_dirs, path_for, result_file_name, run_dir_of_sta_dir
from utils.sta_decomposition import MAX_COMPONENTS, VARIANCE_TARGET, decompose_sta_movie

SEED = 0  # the start of the ICA: saved with every result


def save_cell_decomposition(path: Path, result: dict, head: dict) -> None:
    """Write the decomposition of one cell (`decompose_sta_movie`) with what describes its movie (`head`: `read_result_header` of the STA result it was made from)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result, spatial_bin=head["spatial_bin"], pre_frames=head["pre_frames"], post_frames=head["post_frames"], baseline_window=head["baseline_window"],
                        n_peaks=len(head["peaks"]))


def load_cell_decomposition(path: Path) -> dict:
    """What `save_cell_decomposition` wrote, as plain arrays and numbers. Raises FileNotFoundError saying which step makes the file when it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"The decomposition of this cell is missing ({path.name}). The decomposition step writes it: run that step for this experiment.")
    with np.load(path) as d:
        return {key: (d[key].item() if d[key].ndim == 0 else d[key]) for key in d.files}


def decompose_sta_run(sta_dir: str | Path, output: str | Path, source_dir: str | Path, log: Callable[[str], None] = print) -> Path:
    """Decompose the whole-field STA movie (the 'all' result) of every cell of the STA run whose sta-analysis folder is `sta_dir`; the files go to the run folder `output`. A cell without
    that result, or whose movie cannot be decomposed, is left out and named with the reason in the summary (and in `log`). Returns the summary file."""
    sta_dir, output = Path(sta_dir), Path(output)
    fps = read_frame_rate(source_dir)
    done: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    spatial_bins: set[int] = set()
    for cell, folder in cell_dirs(sta_dir).items():
        result_file = folder / result_file_name(g.ALL_GROUP)
        if not result_file.exists():
            skipped[str(cell)] = f"no 'all' result ({result_file.name}): the STA Maps run was made without that result type"
            log(f"cell {cell}: {skipped[str(cell)]}")
            continue
        head = read_result_header(result_file)
        with np.load(result_file, allow_pickle=True) as d:
            if "observed_sta_movie" not in d:
                skipped[str(cell)] = "its result has no STA movie"
                log(f"cell {cell}: {skipped[str(cell)]}")
                continue
            movie = np.asarray(d["observed_sta_movie"], dtype=np.float32)
        try:
            result = decompose_sta_movie(movie, head["pre_frames"], head["post_frames"], head["baseline_window"], seed=SEED)
        except ValueError as exc:
            skipped[str(cell)] = str(exc)
            log(f"cell {cell}: {exc}")
            continue
        save_cell_decomposition(path_for(PathKind.DECOMPOSITION_CELL, output, cell=cell), result, head)
        spatial_bins.add(head["spatial_bin"])
        done[str(cell)] = {"n_components": result["n_components"], "explained_by_kept": round(float(result["variance_explained"][:result["n_components"]].sum()), 4), "ica_converged": result["ica_converged"]}
        log(f"cell {cell}: {result['n_components']} components, {100 * done[str(cell)]['explained_by_kept']:.1f}% of the variance, ICA {'converged' if result['ica_converged'] else 'did not converge'}")
    summary = path_for(PathKind.DECOMPOSITION_SUMMARY, output)
    write_json(summary, {"sta_run": run_dir_of_sta_dir(sta_dir).name, "fps": fps, "spatial_bins": sorted(spatial_bins), "variance_target": VARIANCE_TARGET, "max_components": MAX_COMPONENTS,
                         "seed": SEED, "cells": done, "skipped": skipped})
    return summary
