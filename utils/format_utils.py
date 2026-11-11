"""How numbers are written for people (the pages and their texts)."""
from __future__ import annotations

import math
import re


def format_p(p) -> str:
    """A p value with up to 5 decimals (at least 3, trailing zeros dropped): 0.043, 0.00012; "<0.00001" below that; an en dash when there is none. The `pval` filter of the templates
    (app/web/static/js/plot_canvas.js has the same rule for the pages drawn in JS)."""
    if p is None or not math.isfinite(p):
        return "–"
    return "<0.00001" if p < 0.00001 else re.sub(r"(\.\d{3}\d*?)0+$", r"\1", f"{p:.5f}")
