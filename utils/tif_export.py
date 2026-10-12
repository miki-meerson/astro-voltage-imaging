"""Fiji / ImageJ copies of the STA movies, written next to the experiment's other outputs when the movies are created.

Layout: <experiment folder>/sta_movies_tif/cell_<N>/sta_<group>_bin<B>.tif, one file per binned STA movie of the STA run (all, simple,
complex, anatomy, behavioural groups). sta_anatomy_bin1.tif is the cropped full-resolution movie the app shows on the cell page. Float32 ImageJ stacks (TYX), biological sign (bright = depolarization),
one slice label per frame (time relative to the spike). Used by the STA cluster job, the app's crop-movie generator and the backfill job
tools/save_cropped_sta_movies.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FIJI_DIR_NAME = "sta_movies_tif"


def fiji_cell_dir(source_home, cell_id) -> Path:
    return Path(source_home) / FIJI_DIR_NAME / f"cell_{int(cell_id)}"


def sta_note(cell_id, n_spikes: int, what: str, run: str | None = None) -> str:
    """Text stored in the tif (ImageJ Info). `run` = name of the STA run folder it was made from: the app uses it to know whether a saved tif is current."""
    tag = f"; run={run}" if run else ""
    return f"cell {int(cell_id)}; spike-triggered average of the astro movie, {int(n_spikes)} spikes; frame 0 = spike; bright = depolarization; {what}{tag}"


def write_imagej_stack(path, movie, fps: float, pre_frames: int, note: str) -> None:
    """Float32 ImageJ stack (axes TYX) with one slice label per frame: its time relative to the spike."""
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    movie = np.asarray(movie, dtype=np.float32)
    rel = np.arange(-int(pre_frames), -int(pre_frames) + movie.shape[0])
    labels = [f"{r:+d} frames ({r / fps:+.3f} s)" if r else "0 (spike)" for r in rel]
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    tifffile.imwrite(tmp, movie, imagej=True, metadata={"axes": "TYX", "finterval": 1.0 / fps, "unit": "s", "Labels": labels, "Info": note})
    tmp.replace(path)
