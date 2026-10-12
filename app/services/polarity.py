"""The one definition of the sign convention shown in the app (JS twin: app/web/static/js/polarity.js).

Biological convention: UP / brighter / red = depolarization, DOWN / darker / blue = hyperpolarization.
The astro GEVI is negative-going: depolarization LOWERS the raw fluorescence. Stored data keep whatever sign they were saved with
(raw tifs, cropped STA movies; the STA maps, effect movies and cluster signs of the cluster jobs are already flipped, so a
"positive" cluster is a raw fluorescence decrease = depolarization). Views convert at display time only, through this module.
"""
from __future__ import annotations

import numpy as np

# True: the raw astro fluorescence goes the opposite way to the membrane potential. The only switch for the whole app.
RAW_SIGN_IS_INVERTED = True
RAW_TO_BIOLOGICAL = -1.0 if RAW_SIGN_IS_INVERTED else 1.0

DEPOLARIZING = "depolarizing"
HYPERPOLARIZING = "hyperpolarizing"
COLORS = {DEPOLARIZING: "#ff5d6c", HYPERPOLARIZING: "#4fa3ff"}
AXIS_NOTE = "depolarization up"


def flip_for_kind(kind: str) -> bool:
    """Whether a raw movie of this kind must be flipped for display. Astro only: the neuron indicator's polarity is unverified."""
    return RAW_SIGN_IS_INVERTED and str(kind) == "astro"


def to_biological(values):
    """Raw astro fluorescence (or a quantity measured on it) -> biological sign (up = depolarization), as float32."""
    return np.asarray(values, dtype=np.float32) * np.float32(RAW_TO_BIOLOGICAL)


def polarity_from_sign(sign) -> str:
    """'positive' / 'negative' (cluster sign in the flipped STA convention) or a number -> 'depolarizing' / 'hyperpolarizing'."""
    if isinstance(sign, str):
        return DEPOLARIZING if sign.strip().lower() in ("positive", DEPOLARIZING) else HYPERPOLARIZING
    return DEPOLARIZING if float(sign) >= 0 else HYPERPOLARIZING


def polarity_color(polarity: str) -> str:
    return COLORS[polarity]
