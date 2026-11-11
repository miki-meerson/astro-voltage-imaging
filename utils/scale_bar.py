"""Scale bars and lengths in micrometres. The size of one pixel of the recorded images comes from the experiment's Experiment.xml (utils/movie_utils.py `read_pixel_size_um`); an image
binned by `bin` has pixels `bin` times as large. app/web/static/js/scale_bar.js has the same rules for the images drawn in the browser."""
from __future__ import annotations

import math

BAR_FRACTION = 0.2  # a scale bar is about this share of the image width


def um_per_pixel(pixel_um: float, spatial_bin: int = 1) -> float:
    """Micrometres in one pixel of an image whose pixels are `spatial_bin` recorded pixels wide."""
    return float(pixel_um) * int(spatial_bin)


def scale_bar_length_um(width_um: float, fraction: float = BAR_FRACTION) -> float:
    """A round length (1, 2 or 5 x a power of ten micrometres) that is at most `fraction` of an image of `width_um` and as long as that allows. Raises ValueError for no width."""
    if not width_um > 0:
        raise ValueError(f"A scale bar needs an image width above zero, not {width_um!r}.")
    target = width_um * fraction
    power = 10.0 ** math.floor(math.log10(target))
    return next(step * power for step in (5, 2, 1) if step * power <= target * (1 + 1e-9))


def format_length_um(length_um: float) -> str:
    """A length for a label: '50 µm', '2.5 µm', '1 mm' from 1000 µm up."""
    return f"{length_um / 1000:g} mm" if length_um >= 1000 else f"{length_um:g} µm"
