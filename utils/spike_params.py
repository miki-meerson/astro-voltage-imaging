"""The detector settings of spike detection: names, defaults and limits, defined once.

Used by the spike adapter (defaults for a run without settings), by the spike editor's server side (app/services/spikes.py: limits and the
values it sends to the page) and, through that, by the editor page. There is no copy of a default in the JavaScript or in the config file."""
from __future__ import annotations

# name -> (kind, default, lo, hi): the limits are what the server enforces
SPIKE_PARAM_SPECS = {
    "mad_factor": (float, 4.0, 0.5, 8.0),
    "min_amplitude": (float, 0.30, 0.0, 2.0),
    "min_width": (float, 18.0, 1.0, 120.0),
    "min_subthreshold": (float, 0.10, 0.0, 1.0),
    "duplicate_window_frames": (int, 30, 1, 240),
    "align_radius_frames": (int, 5, 0, 80),
    "highpass_cutoff_hz": (float, 0.07, 0.0, 1.0),
}


def spike_params_with_defaults(params: dict) -> dict:
    """Every detector setting: the value in `params` when it has one, else the default."""
    return {name: kind(params.get(name, default)) for name, (kind, default, _, _) in SPIKE_PARAM_SPECS.items()}
