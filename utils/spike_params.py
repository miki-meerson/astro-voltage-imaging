"""The detector settings of spike detection: names, defaults and limits, defined once.

Used by the spike adapter (defaults for a run without settings), by the spike editor's server side (app/services/spikes.py: limits and the
values it sends to the page) and, through that, by the editor page. There is no copy of a default in the JavaScript or in the config file."""
from __future__ import annotations

# name -> (kind, default); the limits of what can be chosen are the min / max of the sliders in the editor page
SPIKE_PARAM_SPECS = {
    "mad_factor": (float, 4.0),
    "min_amplitude": (float, 0.30),
    "min_width": (float, 18.0),
    "min_subthreshold": (float, 0.10),
    "duplicate_window_frames": (int, 30),
    "align_radius_frames": (int, 5),
    "highpass_cutoff_hz": (float, 0.07),
}


def spike_params_with_defaults(params: dict) -> dict:
    """Every detector setting: the value in `params` when it has one, else the default."""
    return {name: kind(params.get(name, default)) for name, (kind, default) in SPIKE_PARAM_SPECS.items()}
