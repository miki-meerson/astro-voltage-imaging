"""The border masking for the Overview tab: the margins the STA test leaves out (the same function the job uses, utils/border.py) and the stretch of the
corrected astro movie that shows them best, the frames around the largest image shift. The frames themselves come through the movie routes."""
from __future__ import annotations

import numpy as np

from app.db import experiment_repository
from app.services import polarity
from app.services.movies import CHUNK_FRAMES, movie_info, pb_movie_path
from utils.border import INTERPOLATION_PX, analysed_shifts, border_margins_px
from utils.movie_utils import read_frame_rate

SNIPPET_CHUNKS = 2  # chunks of the movie shown (CHUNK_FRAMES frames each, about 2 s): the page plays them in a loop


def border_snippets(settings, experiment_id: int) -> dict:
    """{kind, first_chunk, chunks, chunk_frames, frames, fps, width, height, block, margins_px {top, bottom, left, right}, margins_shown (the same in
    pixels of the frames sent), shifts (dy, dx of each frame of the snippet), peak_frame, peak_time_s, start_time_s} of the experiment.

    Raises FileNotFoundError saying which step writes what is missing and ValueError when the shifts cannot be matched to the frames."""
    source = experiment_repository.experiment_source_path(settings, experiment_id)
    path = None if source is None else pb_movie_path(source, "astro")
    if path is None or not path.exists():
        raise FileNotFoundError("The photobleaching corrected astro movie was not found. The photobleaching correction step writes it: rerun that step for this experiment.")
    fps = read_frame_rate(source)
    info = movie_info(path, fps, polarity.flip_for_kind("astro"))
    shifts = analysed_shifts(source, info["frames"])
    extra = int(settings.raw.get("defaults", {}).get("sta", {}).get("border_extra_px", INTERPOLATION_PX))
    margins = dict(zip(("top", "bottom", "left", "right"), border_margins_px(shifts, extra)))
    peak = int(np.argmax(np.hypot(shifts[:, 0], shifts[:, 1])))
    n_chunks = int(np.ceil(info["frames"] / CHUNK_FRAMES))
    first = int(np.clip((peak - CHUNK_FRAMES) // CHUNK_FRAMES, 0, max(0, n_chunks - SNIPPET_CHUNKS)))
    chunks = min(SNIPPET_CHUNKS, n_chunks - first)
    start = first * CHUNK_FRAMES
    frames = min(chunks * CHUNK_FRAMES, info["frames"] - start)
    return {
        "kind": "astro", "first_chunk": first, "chunks": chunks, "chunk_frames": CHUNK_FRAMES, "frames": frames, "fps": fps,
        "width": info["width"], "height": info["height"], "block": info["block"],
        "margins_px": margins, "margins_shown": {side: value / info["block"] for side, value in margins.items()},
        "shifts": [[round(float(v), 2) for v in row] for row in shifts[start:start + frames]],
        "peak_frame": peak - start, "peak_time_s": round(peak / fps, 2), "start_time_s": round(start / fps, 2),
    }
