"""Soma / dendrite classification of the Suite2p ROIs of an experiment, from their shape. No training data is needed.

Suite2p has no soma / dendrite classifier: its only pretrained one (suite2p/classifiers/classifier.npy) separates cells from non-cells, and the shape numbers it
saves (compact, aspect_ratio) are measured on a cropped, soma-like core of each ROI, so a dendrite looks compact there. The shape is measured here on the whole ROI
(every pixel of stat.npy's ypix / xpix), after a clean-up: a Suite2p ROI is ragged (holes, stray pixels), which would make even a round cell look irregular, so
small gaps are closed, holes filled and only the main connected piece is kept. `npix` is the pixel count of the ROI as Suite2p saved it, the rest is measured
on the cleaned ROI:

    elongation   sqrt(largest / smallest variance of the pixel coordinates): 1 for a round ROI, large for a thin one
    solidity     the ROI's area / the area of its convex hull: 1 for a convex ROI, small for a branching or curved one
    max_width    twice the largest distance from a pixel of the ROI to its edge: how thick the ROI is where it is thickest (pixels)
    length       the major axis of the ROI's ellipse (pixels)

The rule is deliberately cautious: a soma is round and convex, a dendrite is elongated or branching, and everything between (and every ROI too small to judge) is
"uncertain" and is shown as such: a wrong label is worse than an honest "not sure". The thresholds are defaults of an untested rule: check them against ROIs you
know, correct the labels by hand, and change the thresholds in config (defaults.roi_classification) when they do not suit the data.

Only the ACCEPTED ROIs (iscell = 1) are classified: the feature is about the ROIs the analyses use. Everything is saved in ONE file per experiment (roi_classes.json,
pipeline_results/roi_classification/), one row per accepted ROI, by the ORIGINAL Suite2p ROI index (the position in stat.npy), with the automatic label, the label the
user set by hand (if any) and the final `label` (the manual one when there is one, else the automatic one): an analysis reads `label` and nothing else. The automatic
classification and a manual correction both change that one file; running the automatic classification again makes the whole file anew, the manual labels included. iscell.npy, stat.npy and the cell
numbering of the spikes and the analyses (`cell`, the position among the accepted ROIs) are not touched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt, label
from scipy.spatial import ConvexHull

from utils.file_utils import write_json
from utils.paths_utils import PathKind, path_for

SOMA, DENDRITE, UNCERTAIN = "soma", "dendrite", "uncertain"
CLASSES = (  # the labels in the order of the legend, with how a page shows them
    {"name": SOMA, "label": "Soma", "color": "#2fbf71"},
    {"name": DENDRITE, "label": "Dendrite", "color": "#d65db1"},
    {"name": UNCERTAIN, "label": "Uncertain", "color": "#f2b84b"},
)
MANUAL_LABELS = (SOMA, DENDRITE)  # what a user can set; "uncertain" is only something the automatic rule says

DEFAULT_SETTINGS = {
    "min_pixels": 20,  # an ROI with fewer pixels is too small to judge by shape
    "soma_max_elongation": 1.7,  # a soma is at most this elongated ...
    "soma_min_solidity": 0.75,  # ... and at least this convex
    "dendrite_min_elongation": 2.6,  # a dendrite is at least this elongated ...
    "dendrite_max_solidity": 0.55,  # ... or at most this convex
}
PIXEL_VARIANCE = 1 / 12  # the variance of a pixel's own width: keeps a one-pixel-wide ROI from having no width at all


def roi_settings(overrides: dict | None = None) -> dict:
    """The thresholds: the defaults, with `overrides` (config defaults.roi_classification) on top. Raises ValueError for a name that is not a threshold, a value that is not a
    number, or thresholds that overlap (an ROI would then be both a soma and a dendrite)."""
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(DEFAULT_SETTINGS))
    if unknown:
        raise ValueError(f"Unknown ROI classification setting(s) {', '.join(unknown)}. Available: {', '.join(DEFAULT_SETTINGS)}.")
    settings = {**DEFAULT_SETTINGS, **overrides}
    bad = [key for key, value in settings.items() if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)]
    if bad:
        raise ValueError(f"ROI classification setting(s) {', '.join(bad)} must be numbers.")
    if settings["soma_max_elongation"] >= settings["dendrite_min_elongation"] or settings["soma_min_solidity"] <= settings["dendrite_max_solidity"]:
        raise ValueError("The soma thresholds must stay clear of the dendrite thresholds: soma_max_elongation < dendrite_min_elongation and soma_min_solidity > dendrite_max_solidity.")
    return settings


def roi_features(ypix, xpix) -> dict:
    """{npix, elongation, solidity, max_width, length} of the ROI made of these pixels (see the module text)."""
    y0, x0 = int(np.min(ypix)), int(np.min(xpix))
    mask = np.zeros((int(np.max(ypix)) - y0 + 5, int(np.max(xpix)) - x0 + 5), dtype=bool)  # padded: the pixels outside the ROI are its edge
    mask[np.asarray(ypix, dtype=int) - y0 + 2, np.asarray(xpix, dtype=int) - x0 + 2] = True
    ragged = mask
    pieces, n_pieces = label(binary_fill_holes(binary_closing(ragged, structure=np.ones((3, 3), dtype=bool))), structure=np.ones((3, 3), dtype=int))
    cleaned = pieces == (1 + int(np.argmax(np.bincount(pieces.ravel())[1:])))  # the biggest connected piece
    y, x = (axis.astype(float) for axis in np.nonzero(cleaned))
    area = y.size
    var_y, var_x = np.mean((y - y.mean()) ** 2) + PIXEL_VARIANCE, np.mean((x - x.mean()) ** 2) + PIXEL_VARIANCE
    covariance = np.mean((y - y.mean()) * (x - x.mean()))
    spread = np.sqrt(((var_y - var_x) / 2) ** 2 + covariance ** 2)  # the two variances along the principal axes of the 2 x 2 covariance matrix, in closed form
    small, large = (var_y + var_x) / 2 - spread, (var_y + var_x) / 2 + spread
    corners = np.concatenate([np.stack([y + dy, x + dx], axis=1) for dy in (-0.5, 0.5) for dx in (-0.5, 0.5)])  # the hull of the pixels as squares, never of single points
    return {
        "npix": int(np.count_nonzero(ragged)), "elongation": float(np.sqrt(large / small)), "solidity": float(min(1.0, area / ConvexHull(corners).volume)),
        "max_width": float(2 * distance_transform_edt(cleaned).max() - 1), "length": float(4 * np.sqrt(large)),
    }


def classify_roi(features: dict, settings: dict) -> tuple[str, str]:
    """(label, why) of one ROI from its features: soma, dendrite or uncertain, and the sentence that says why."""
    if features["npix"] < settings["min_pixels"]:
        return UNCERTAIN, f"only {features['npix']} pixels: too small to judge by shape"
    elongated = features["elongation"] >= settings["dendrite_min_elongation"]
    branching = features["solidity"] <= settings["dendrite_max_solidity"]
    if elongated or branching:
        return DENDRITE, "elongated" if elongated and not branching else "branching or curved" if branching and not elongated else "elongated and branching"
    if features["elongation"] <= settings["soma_max_elongation"] and features["solidity"] >= settings["soma_min_solidity"]:
        return SOMA, "round and convex"
    return UNCERTAIN, "between a soma and a dendrite: " + ("somewhat elongated" if features["elongation"] > settings["soma_max_elongation"] else "somewhat irregular")


FEATURE_NAMES = ("npix", "elongation", "solidity", "max_width", "length")


def rounded(features: dict) -> dict:
    """The features as they are saved: plain numbers, rounded."""
    return {name: round(float(features[name]), 3) for name in FEATURE_NAMES}


def classify_rois(stat, accepted: np.ndarray, settings: dict) -> list[dict]:
    """The rows of the accepted ROIs (`accepted`: iscell as a boolean per ROI of stat.npy), by original ROI index: {roi, cell (the position among the accepted ROIs), label, auto,
    why, manual (none yet), and the features}."""
    rows = []
    for cell, roi in enumerate(np.flatnonzero(accepted)):
        features = roi_features(stat[roi]["ypix"], stat[roi]["xpix"])
        auto, why = classify_roi(features, settings)
        rows.append({"roi": int(roi), "cell": cell, "label": auto, "auto": auto, "why": why, "manual": None, **rounded(features)})
    return rows


def stat_fingerprint(stat) -> str:
    """A short fingerprint of the ROIs of stat.npy (their number, sizes and centres): classifications made for one Suite2p result must not be applied to another."""
    digest = hashlib.sha1()
    for roi in stat:
        digest.update(np.asarray([len(roi["ypix"]), *np.asarray(roi["med"], dtype=float)], dtype=np.float64).tobytes())
    return f"{len(stat)}-{digest.hexdigest()[:16]}"


def load_classes(experiment: str | Path, fingerprint: str) -> dict | None:
    """The saved classification {thresholds, rows: {original ROI index: row}}, or None when there is none. Raises ValueError when it was made for other Suite2p ROIs than the
    current ones (Suite2p was run again): the labels set by hand in it do not belong to the current ROIs."""
    path = path_for(PathKind.ROI_CLASSES, experiment)
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["fingerprint"] != fingerprint:
        raise ValueError(f"The ROI classification in {path.name} was made for other Suite2p ROIs than the current ones (Suite2p was run again). Run the automatic classification again: it "
                         "makes the file anew, and the labels set by hand in it are lost.")
    return {"thresholds": saved["thresholds"], "rows": {row["roi"]: row for row in saved["rows"]}}


def save_classes(experiment: str | Path, fingerprint: str, thresholds: dict, rows: list[dict]) -> None:
    """Save the classification of an experiment (its one file), rows by original ROI index."""
    unknown = sorted({row["manual"] for row in rows if row["manual"]} - set(MANUAL_LABELS))
    if unknown:
        raise ValueError(f"Unknown label(s) {', '.join(unknown)}: a label set by hand is one of {', '.join(MANUAL_LABELS)}.")
    write_json(path_for(PathKind.ROI_CLASSES, experiment), {"fingerprint": fingerprint, "thresholds": thresholds, "rows": sorted(rows, key=lambda row: row["roi"])})
