"""The error text for a saved output an STA run does not have. Used by the app views and by utils that read saved outputs."""
from __future__ import annotations


def missing_output_message(what: str, sta_run_label) -> str:
    """What is missing for an STA run and which step makes it (rule: a view never makes a missing file, it says where it comes from)."""
    return f"{what} is missing for STA run {sta_run_label}. It is written by the STA Maps step: rerun STA Maps for this experiment."
