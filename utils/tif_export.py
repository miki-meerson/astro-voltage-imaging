"""Fiji / ImageJ copies of the STA movies, written next to the experiment's other outputs when the movies are created.

Layout: <experiment folder>/sta_movies_tif/cell_<N>/sta_<group>_bin<B>.tif (paths: utils/paths_utils.py, PathKind.STA_MOVIE_TIF), one file per
binned STA movie of the STA run (all, simple, complex, anatomy, behavioural groups). sta_anatomy_bin1.tif is the cropped full-resolution movie
the app shows on the cell page. Float32 ImageJ stacks (TYX), biological sign (bright = depolarization), one slice label per frame (time
relative to the spike). Written by the STA cluster job (cluster-code/sta_maps_sparse.py). Every file carries `run=<STA run folder name>` in its ImageJ info text.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile


def sta_note(cell_id, n_spikes: int, what: str, run: str | None = None) -> str:
    """Text stored in the tif (ImageJ Info). `run` = name of the STA run folder it was made from: the app uses it to know whether a saved tif is current."""
    tag = f"; run={run}" if run else ""
    return f"cell {int(cell_id)}; spike-triggered average of the astro movie, {int(n_spikes)} spikes; frame 0 = spike; bright = depolarization; {what}{tag}"


def saved_run_name(tif: str | Path) -> str | None:
    """Name of the STA run folder a saved tif was made from (the `run=` tag of its ImageJ info text); None when the file has no tag.
    A missing or unreadable file raises (FileNotFoundError / the tifffile error)."""
    with tifffile.TiffFile(tif) as t:
        info = (t.imagej_metadata or {}).get("Info", "")
    m = re.search(r"run=(\S+)", str(info))
    return m.group(1) if m else None


def write_imagej_stack(path, movie, fps: float, pre_frames: int, note: str) -> None:
    """Float32 ImageJ stack (axes TYX) with one slice label per frame: its time relative to the spike."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    movie = np.asarray(movie, dtype=np.float32)
    rel = np.arange(-int(pre_frames), -int(pre_frames) + movie.shape[0])
    labels = [f"{r:+d} frames ({r / fps:+.3f} s)" if r else "0 (spike)" for r in rel]
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    tifffile.imwrite(tmp, movie, imagej=True, metadata={"axes": "TYX", "finterval": 1.0 / fps, "unit": "s", "Labels": labels, "Info": note})
    tmp.replace(path)
