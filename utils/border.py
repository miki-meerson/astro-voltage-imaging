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
from utils.motion_diagnostics import align_shifts_to_analysed_frames, load_shifts
from utils.paths_utils import PathKind, path_for
from utils.qc_utils import read_trimmed_frames

INTERPOLATION_PX = 2  # the cubic interpolation of the registration reads this many pixels on each side of a pixel


def read_bad_segments(experiment_dir: str | Path) -> list:
    """The removed bad segments of the experiment, [start, end) frame ranges in the numbering before the removal ([] when there are none)."""
    return read_json(path_for(PathKind.BAD_SEGMENTS, experiment_dir), "The bad-segments file", default=[])


def analysed_shifts(experiment_dir: str | Path, n_frames: int) -> np.ndarray:
    """The applied registration shifts (dy, dx) of the analysed frames, shape (n_frames, 2).

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
    return aligned


def border_margins_px(shifts: np.ndarray, extra_px: int = INTERPOLATION_PX) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) margins in full-resolution pixels for the shifts (dy, dx) of the analysed frames."""
    dy, dx = np.asarray(shifts, dtype=np.float64)[:, 0], np.asarray(shifts, dtype=np.float64)[:, 1]
    exposed = (max(0.0, dy.max()), max(0.0, -dy.min()), max(0.0, dx.max()), max(0.0, -dx.min()))
    return tuple(int(np.ceil(v)) + int(extra_px) for v in exposed)


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
