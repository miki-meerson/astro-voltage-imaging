"""The invalid border of the motion-corrected movie, and the mask that leaves it out of the STA test.

The registration moves every frame by its shift (cv2.warpAffine with M = [[1, 0, dx], [0, 1, dy]]: the content moves by +dx, +dy) and fills what
slides into view with reflected content. So a frame shifted by dy > 0 has invalid rows at the top (the first dy rows), dy < 0 at the bottom, and
dx > 0 at the left, dx < 0 at the right; the values there follow the shift and not the tissue. The margin of a side is the largest shift toward
it over the analysed frames, rounded up, plus the pixels the cubic interpolation mixes in. The job and the Overview panel both take their margins
from here, so the picture shows what the test left out.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.file_utils import read_json
from utils.motion_diagnostics import align_shifts_to_analysed_frames, load_patch_extent, load_shifts
from utils.paths_utils import PathKind, path_for
from utils.qc_utils import excluded_mask, read_excluded_segments, read_trimmed_frames

INTERPOLATION_PX = 2  # the cubic interpolation of the registration reads this many pixels on each side of a pixel


def read_bad_segments(experiment_dir: str | Path) -> list:
    """The removed bad segments of the experiment, [start, end) frame ranges in the numbering before the removal ([] when there are none)."""
    return read_json(path_for(PathKind.BAD_SEGMENTS, experiment_dir), "The bad-segments file", default=[])


def analysed_shifts(experiment_dir: str | Path, n_frames: int, with_patches: bool = False) -> np.ndarray:
    """The applied registration shifts (dy, dx) of the analysed frames, shape (n_frames, 2).

    with_patches: for the border. After a non-rigid motion correction every patch moved by its own shift, so the rows are instead the extremes over the patches,
    (dy max, dx max) of every frame followed by (dy min, dx min) of every frame, shape (2 * n_frames, 2); a rigid correction gives its shifts as above.
    Raises FileNotFoundError when the experiment has no shifts file and ValueError when no reading of it (see align_shifts_to_analysed_frames)
    gives `n_frames` rows."""
    path = path_for(PathKind.MC_SHIFTS, experiment_dir)
    shifts = load_shifts(path)
    if shifts is None:
        raise FileNotFoundError(f"The motion-correction shifts file is missing ({path}). The motion correction step writes it: rerun that step.")
    trimmed, bad = read_trimmed_frames(experiment_dir), read_bad_segments(experiment_dir)
    aligned = align_shifts_to_analysed_frames(shifts, int(n_frames), trimmed, bad)
    if aligned is None:
        raise ValueError(f"The shifts file has {len(shifts)} rows but {n_frames} frames are analysed (photobleaching trimmed {trimmed}, bad segments {bad}): the shifts cannot be matched to the frames.")
    extent = load_patch_extent(path) if with_patches else None
    if extent is None:
        return aligned
    extent = align_shifts_to_analysed_frames(extent, int(n_frames), trimmed, bad)
    return np.concatenate([extent[:, [0, 2]], extent[:, [1, 3]]])


def border_margins_px(shifts: np.ndarray, extra_px: int = INTERPOLATION_PX, keep: np.ndarray | None = None) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) margins in full-resolution pixels for the shifts (dy, dx) of the analysed frames. keep: a boolean per analysed frame, the frames whose shifts
    count (all when None); for the stacked rows of analysed_shifts(with_patches=True) it applies to each block of rows."""
    shifts = np.asarray(shifts, dtype=np.float64)
    if keep is not None:
        keep = np.asarray(keep, dtype=bool)
        if not keep.any():
            raise ValueError("Every frame is excluded: there are no shifts left to take the border from.")
        shifts = shifts[np.tile(keep, len(shifts) // len(keep))]
    dy, dx = shifts[:, 0], shifts[:, 1]
    exposed = (max(0.0, dy.max()), max(0.0, -dy.min()), max(0.0, dx.max()), max(0.0, -dx.min()))
    return tuple(int(np.ceil(v)) + int(extra_px) for v in exposed)


def experiment_border_margins(experiment_dir: str | Path, n_frames: int, extra_px: int = INTERPOLATION_PX, excluded: list | None = None) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) margins of an experiment, from the shifts (of every patch after a non-rigid correction) of the analysed frames outside its excluded segments
    (`excluded`, or the experiment's own when None): what the STA job leaves out, and what the Overview panel shows. Raises what analysed_shifts raises."""
    keep = ~excluded_mask(n_frames, read_excluded_segments(experiment_dir) if excluded is None else excluded)
    return border_margins_px(analysed_shifts(experiment_dir, n_frames, with_patches=True), extra_px, keep)


def valid_mask(shape_hw: tuple[int, int], margins_px: tuple[int, int, int, int], spatial_bin: int) -> np.ndarray:
    """(H, W) boolean mask of the pixels that are analysed, False in the border. The margins (full-resolution pixels) are rounded up to whole
    pixels of the binned movie. Raises ValueError when they would cover the whole field."""
    height, width = (int(v) for v in shape_hw)
    top, bottom, left, right = (int(np.ceil(m / int(spatial_bin))) for m in margins_px)
    if top + bottom >= height or left + right >= width:
        raise ValueError(f"The border margins {tuple(margins_px)} px leave no pixels of a {height}x{width} field at spatial bin {spatial_bin}.")
    valid = np.ones((height, width), dtype=bool)
    valid[:top], valid[height - bottom:], valid[:, :left], valid[:, width - right:] = False, False, False, False
    return valid


def motion_border_margins(experiment_dir: str | Path, trimmed: int = 0) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) margins from the registration shifts of the frames after the first `trimmed` ones (the movie the photobleaching step writes), from every patch after a
    non-rigid correction. The photobleaching step blanks these margins, so it needs no photobleaching QC file yet. Raises FileNotFoundError when the shifts file is missing."""
    path = path_for(PathKind.MC_SHIFTS, experiment_dir)
    shifts = load_shifts(path)
    if shifts is None:
        raise FileNotFoundError(f"The motion-correction shifts file is missing ({path}). The motion correction step writes it: rerun that step.")
    extent = load_patch_extent(path)
    rows = shifts[int(trimmed):] if extent is None else np.concatenate([extent[int(trimmed):, [0, 2]], extent[int(trimmed):, [1, 3]]])
    return border_margins_px(rows)


def blank_border(frame: np.ndarray, margins_px: tuple[int, int, int, int]) -> np.ndarray:
    """`frame` (H, W) with its border set to the mean of the pixels inside it, in place; returned. Raises ValueError when the margins cover the whole field."""
    top, bottom, left, right = (int(m) for m in margins_px)
    height, width = frame.shape
    if top + bottom >= height or left + right >= width:
        raise ValueError(f"The border margins {tuple(margins_px)} px leave no pixels of a {height}x{width} frame.")
    inside = frame[top:height - bottom, left:width - right]
    fill = inside.mean()
    frame[:top], frame[height - bottom:], frame[:, :left], frame[:, width - right:] = fill, fill, fill, fill
    return frame
