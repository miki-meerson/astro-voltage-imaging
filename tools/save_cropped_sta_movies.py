"""Backfill the STA movies of an experiment as Fiji-readable .tif files in its folder (older STA runs).

    python tools/save_cropped_sta_movies.py --experiment-id 13

New STA runs write these by themselves; the app writes the cropped movie of a cell when it first opens it. This does it for every cell of an
experiment at once, from files the STA run already saved (no astro movie is read, so it is quick):
  <experiment>/sta_movies_tif/cell_<N>/sta_anatomy_bin1.tif   the cropped movie shown on the cell page: the anatomy result's observed STA movie
  <experiment>/sta_movies_tif/cell_<N>/sta_<group>_bin<B>.tif  the binned STA movies older runs saved as .npy (all, simple, complex, ...)
Float32 ImageJ stacks, biological sign (bright = depolarization), one slice label per frame (time relative to the spike).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]

from app.db.core import connect  # noqa: E402
from app.services.cell_views import ensure_saved_cropped_sta_movie  # noqa: E402
from app.services.movies import experiment_fps  # noqa: E402
from app.settings import load_settings  # noqa: E402
from utils.tif_export import fiji_cell_dir, sta_note, write_imagej_stack  # noqa: E402


def _write_binned_tifs(cell: int, cell_dir: Path, pre_frames: int, frames: int, n_spikes: int, fps: float, source_path, run_name: str) -> list[str]:
    """Convert the binned STA movies an older STA run saved as .npy (already in the biological sign) to .tif."""
    out_dir = fiji_cell_dir(source_path, cell)
    written = []
    for npy in sorted(Path(cell_dir).glob("astro_sta_movie*_bin*.npy")):
        stem = npy.stem.replace("astro_sta_movie", "", 1)  # "_bin4" | "_simple_bin4" | "_anatomy_bin1"
        group, _, binpart = stem.strip("_").rpartition("_")
        if group == "anatomy":  # identical to the cropped movie the app writes (sta_anatomy_bin1.tif): saved once
            continue
        movie = np.load(npy)
        if movie.ndim != 3 or movie.shape[0] != frames:
            continue
        path = out_dir / f"sta_{group or 'all'}_{binpart}.tif"
        write_imagej_stack(path, movie, fps, pre_frames, sta_note(cell, n_spikes, f"{group or 'all'} spikes, spatial {binpart}", run=run_name))
        written.append(path.name)
    return written


def _save_experiment(settings, experiment_id: int) -> int:
    with connect(settings) as conn:
        exp = conn.execute("SELECT source_path FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if exp is None:
            raise ValueError(f"Experiment {experiment_id} was not found.")
        neurons = conn.execute(
            "SELECT id,pipeline_neuron_id FROM neurons WHERE experiment_id=? ORDER BY CAST(pipeline_neuron_id AS INTEGER)", (experiment_id,)
        ).fetchall()
    source_path = exp["source_path"]
    try:
        fps = experiment_fps(source_path)
    except Exception:
        fps = 1.0
    ok = failed = 0
    for neuron in neurons:
        cell = int(neuron["pipeline_neuron_id"])
        try:
            info = ensure_saved_cropped_sta_movie(settings, neuron["id"])  # writes sta_anatomy_bin1.tif if it is missing or from another run
            anat = Path(info["source_sta_npz"])
            binned = _write_binned_tifs(cell, anat.parent, info["pre_frames"], info["frames"], info["n_peaks"], fps, source_path, anat.parents[3].name)
            print(f"OK cell={cell} crop={info['crop']['width']}x{info['crop']['height']} binned_tifs={len(binned)} -> {info['movie_path']}", flush=True)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP cell={cell}: {exc}", flush=True)
            failed += 1
    print(f"summary ok={ok} skipped={failed}", flush=True)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill the cropped and binned STA movies of an experiment as Fiji .tif files (see the module docstring).")
    parser.add_argument("--experiment-id", type=int, required=True)
    args = parser.parse_args()
    _save_experiment(load_settings(), args.experiment_id)


if __name__ == "__main__":
    main()
