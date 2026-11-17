"""The SNAPT step of the pipeline: the propagation maps (utils/snapt.py) of every cell of the decomposition of a finished STA run, saved one file per cell in the run folder of the step
(SNAPT_CELL) with a summary of what was done and which cells were left out and why (SNAPT_SUMMARY). The cluster job (tools/snapt.py) calls `snapt_sta_run`; the pages read the files back with
`load_cell_snapt` (app/services/snapt.py). It reads only saved files: the decomposition of each cell and the significant clusters of its 'all' and anatomy STA results. No Flask, no app imports."""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Callable

import numpy as np

from utils import globals_utils as g
from utils.cluster_result import cluster_pixels_fullres, load_cluster_result, read_result_header
from utils.decomposition_run import load_cell_decomposition
from utils.file_utils import read_json, write_json
from utils.movie_utils import read_frame_rate
from utils.paths_utils import PathKind, path_for, result_file_name, run_dir_of_sta_dir
from utils.snapt import DENOISE_COMPONENTS, snapt_cell
from utils.sta_decomposition import rebuild_movie

SIGN_SEPARATOR = "__"  # a saved array is named <sign>__<name>


def save_cell_snapt(path: Path, results: dict[str, dict], clusters: list[dict], spatial_bin: int, components: int) -> None:
    """Write the maps of one cell (`snapt_cell`), for each of its signs, with all the clusters its kernels were made from (footprints on the grid of the maps, sign, result type), the bin of
    its maps and the number of PCA components its movie was rebuilt from."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"{sign}{SIGN_SEPARATOR}{name}": value for sign, result in results.items() for name, value in result.items()}
    np.savez_compressed(path, signs=np.array(list(results)), spatial_bin=spatial_bin, components=components, cluster_masks=np.array([c["mask"] for c in clusters]),
                        cluster_signs=np.array([c["sign"] for c in clusters]), cluster_types=np.array([c["type"] for c in clusters]), **arrays)


def load_cell_snapt(path: Path) -> dict:
    """{spatial_bin, components, results: {sign: {name: array or number}}, clusters: {masks (n x H x W), signs, types}} as `save_cell_snapt` wrote it. Raises FileNotFoundError saying which
    step makes the file when it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"The SNAPT maps of this cell are missing ({path.name}). The SNAPT step writes them: run that step for this experiment.")
    with np.load(path) as d:
        signs = [str(sign) for sign in d["signs"]]
        value = lambda key: d[key].item() if d[key].ndim == 0 else d[key]
        results = {sign: {key[len(sign) + len(SIGN_SEPARATOR):]: value(key) for key in d.files if key.startswith(sign + SIGN_SEPARATOR)} for sign in signs}
        clusters = {"masks": d["cluster_masks"], "signs": [str(v) for v in d["cluster_signs"]], "types": [str(v) for v in d["cluster_types"]]}
        return {"spatial_bin": int(d["spatial_bin"]), "components": int(d["components"]), "results": results, "clusters": clusters}


def _result_clusters(result_file: Path, result_type: str, grid_bin: int, shape: tuple[int, int]) -> list[dict]:
    """The significant clusters of a saved result ('all': the whole field; anatomy: the crop around the cell) with their wide traces, their footprints moved onto the pixel grid of the whole
    field binned by `grid_bin` (`shape`)."""
    head = read_result_header(result_file)
    clusters = []
    for cluster, mask in load_cluster_result(result_file)[2]:
        if "wide_trace" not in cluster:
            continue
        ys, xs = cluster_pixels_fullres(mask, head["crop"], head["spatial_bin"])
        ys, xs = ys // grid_bin, xs // grid_bin
        inside = (ys < shape[0]) & (xs < shape[1])
        footprint = np.zeros(shape, dtype=bool)
        footprint[ys[inside], xs[inside]] = True
        clusters.append({"sign": cluster["sign"], "type": result_type, "t": cluster["wide_trace"][0], "trace": cluster["wide_trace"][1], "mask": footprint})
    return clusters


def _snapt_one_cell(job: tuple) -> dict:
    """The maps of one cell from its saved decomposition and the significant clusters of its 'all' and anatomy results (the anatomy one only when it was made): {cell, results, clusters,
    spatial_bin, components} or {cell, reason}."""
    cell, decomposition_file, result_files, fps = job
    try:
        d = load_cell_decomposition(decomposition_file)
        shape = tuple(d["pca_maps"].shape[1:])
        clusters = [c for result_type, file in result_files.items() if file.exists() for c in _result_clusters(file, result_type, int(d["spatial_bin"]), shape)]
        components = min(DENOISE_COMPONENTS, int(d["n_components"]))
        results = snapt_cell(rebuild_movie(d["pca_maps"], d["pca_courses"], components), d["rels"], clusters, fps)
    except ValueError as exc:
        return {"cell": cell, "reason": str(exc)}
    return {"cell": cell, "results": results, "clusters": clusters, "spatial_bin": int(d["spatial_bin"]), "components": components}


def _answers(jobs: list[tuple], workers: int):
    """The answers of `_snapt_one_cell` for the jobs, as they finish: `workers` processes when there are several, else in this one."""
    if workers <= 1:
        yield from map(_snapt_one_cell, jobs)
        return
    with mp.Pool(workers) as pool:
        yield from pool.imap_unordered(_snapt_one_cell, jobs)


def snapt_sta_run(sta_dir: str | Path, decomposition_dir: str | Path, output: str | Path, source_dir: str | Path, log: Callable[[str], None] = print, workers: int = 1) -> Path:
    """The propagation maps of every cell that the decomposition run `decomposition_dir` did, from the significant clusters of the STA run whose sta-analysis folder is `sta_dir` (the run the
    decomposition was made from: anything else is a ValueError); the files go to the run folder `output`. A cell that gives no map is left out and named with the reason in the summary (and in
    `log`). `workers` cells are done at a time. Returns the summary file."""
    sta_dir, decomposition_dir, output = Path(sta_dir), Path(decomposition_dir), Path(output)
    decomposed = read_json(path_for(PathKind.DECOMPOSITION_SUMMARY, decomposition_dir), "The decomposition summary")
    sta_run = run_dir_of_sta_dir(sta_dir).name
    if decomposed["sta_run"] != sta_run:
        raise ValueError(f"The newest decomposition was made from the STA run {decomposed['sta_run']}, not from the newest one ({sta_run}): run the decomposition again first.")
    fps = read_frame_rate(source_dir)
    jobs = []
    for cell in map(int, decomposed["cells"]):
        result_files = {g.ALL_GROUP: path_for(PathKind.CELL_DIR, sta_dir, cell=cell) / result_file_name(g.ALL_GROUP), g.ANATOMY_GROUP: path_for(PathKind.ANATOMY_RESULT, sta_dir, cell=cell)}
        jobs.append((cell, path_for(PathKind.DECOMPOSITION_CELL, decomposition_dir, cell=cell), result_files, fps))
    done: dict[str, dict] = {}
    skipped = {cell: f"left out of the decomposition: {reason}" for cell, reason in decomposed["skipped"].items()}
    for answer in _answers(jobs, workers):
        cell = str(answer["cell"])
        if "reason" in answer:
            skipped[cell] = answer["reason"]
            log(f"cell {cell}: {answer['reason']}")
            continue
        save_cell_snapt(path_for(PathKind.SNAPT_CELL, output, cell=answer["cell"]), answer["results"], answer["clusters"], answer["spatial_bin"], answer["components"])
        done[cell] = {sign: {"n_clusters": int(r["n_clusters"]), "n_anatomy": int(r["n_anatomy"]), "accepted_fraction": round(float(r["accepted"].mean()), 4)} for sign, r in answer["results"].items()}
        log(f"cell {cell}: " + ", ".join(f"{sign} {r['n_clusters']} clusters ({r['n_anatomy']} anatomy), {100 * r['accepted'].mean():.0f}% of the pixels accepted" for sign, r in answer["results"].items()))
    summary = path_for(PathKind.SNAPT_SUMMARY, output)
    write_json(summary, {"sta_run": sta_run, "decomposition_run": decomposition_dir.name, "fps": fps, "components": DENOISE_COMPONENTS, "cells": done, "skipped": skipped})
    return summary
