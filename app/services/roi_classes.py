"""The soma / dendrite labels of the accepted (iscell = 1) Suite2p ROIs of an experiment, for the ROI types tab of the Preprocessing page.

The labels are an extra property of each ROI, by its ORIGINAL Suite2p index (the position in stat.npy): iscell.npy, stat.npy and the cell numbering of the spikes and the
analyses (the position among the accepted ROIs, `cell` below) are not touched. They live in ONE saved file (utils/roi_classification.py), one row per accepted ROI with the
automatic label, the label set by hand and the final `label`; the automatic classification is run on request (and replaces the whole file) and a click on the page changes one row of it. The page only
reads and shows what is saved."""
from __future__ import annotations

import io
import json
from datetime import datetime

import numpy as np
from PIL import Image

from app.services import caches
from app.services.suite2p_data import experiment_rois
from utils import globals_utils as g
from utils import roi_classification as rc
from utils.file_utils import write_json
from utils.movie_utils import get_mean_images
from utils.paths_utils import PathKind, path_for

FOV_PERCENTILES = (0.5, 99.5)  # the contrast of the field of view


def _entry(roi: int, cell: int, stat, row: dict | None) -> dict:
    """One accepted ROI as the page shows it, from its saved row (None when it has none)."""
    row = row or {}
    return {
        "roi": roi, "cell": cell, "ypix": [int(v) for v in stat[roi]["ypix"]], "xpix": [int(v) for v in stat[roi]["xpix"]], "label": row.get("label"), "auto": row.get("auto"),
        "manual": row.get("manual"), "why": row.get("why"), "features": {name: row[name] for name in rc.FEATURE_NAMES} if row.get("npix") is not None else None,
    }


def roi_class_payload(settings, exp: dict) -> dict:
    """{classes (the legend), manual_labels, auto_available, warning, thresholds (the ones the saved classification used, else the ones a run would use), counts {label or
    "unclassified": number of accepted ROIs}, n_manual, rois: [{roi (original index), cell, ypix, xpix, label, auto, manual, why, features}]} of the accepted (iscell = 1) ROIs.
    Raises FileNotFoundError when Suite2p has not run."""
    source = exp["source_path"]
    mask, stat = experiment_rois(source)
    fingerprint = rc.stat_fingerprint(stat)
    warning = ""
    try:
        saved = rc.load_classes(source, fingerprint)
    except ValueError as exc:  # a classification of another Suite2p result: shown as missing, with the reason
        saved, warning = None, str(exc)
    rows = {} if saved is None else saved["rows"]
    entries = [_entry(int(roi), cell, stat, rows.get(int(roi))) for cell, roi in enumerate(np.flatnonzero(mask))]
    unlabelled = sum(1 for entry in entries if entry["label"] is None)
    if saved is not None and unlabelled:
        warning = f"{unlabelled} accepted ROIs have no label yet (the accepted ROIs changed after the classification): run the automatic classification."
    counts = {}
    for entry in entries:
        counts[entry["label"] or "unclassified"] = counts.get(entry["label"] or "unclassified", 0) + 1
    return {
        "classes": list(rc.CLASSES), "manual_labels": list(rc.MANUAL_LABELS), "auto_available": saved is not None, "warning": warning,
        "thresholds": saved["thresholds"] if saved is not None else rc.roi_settings(settings.defaults.get(g.ROI_CLASS_SETTINGS)), "counts": counts,
        "n_manual": sum(1 for entry in entries if entry["manual"]), "rois": entries,
    }


def run_auto_classification(settings, exp: dict) -> dict:
    """Classify the accepted ROIs of the experiment from their shape and save the result as the experiment's one classification file, replacing the earlier one: the labels set by
    hand in it are lost too. Returns the payload."""
    source = exp["source_path"]
    mask, stat = experiment_rois(source)
    fingerprint = rc.stat_fingerprint(stat)
    thresholds = rc.roi_settings(settings.defaults.get(g.ROI_CLASS_SETTINGS))
    rc.save_classes(source, fingerprint, thresholds, rc.classify_rois(stat, mask, thresholds))
    return roi_class_payload(settings, exp)


def set_manual_class(settings, exp: dict, roi: int, label: str | None) -> dict:
    """Set (label soma or dendrite) or clear (label None) the label of one accepted ROI by hand, in the experiment's classification file; returns the ROI as the page shows it.
    Raises ValueError for a ROI that is not an accepted cell, for a label the user cannot set, and for a file that belongs to an earlier Suite2p result."""
    source = exp["source_path"]
    mask, stat = experiment_rois(source)
    fingerprint = rc.stat_fingerprint(stat)
    if not 0 <= roi < len(stat) or not mask[roi]:
        raise ValueError(f"ROI {roi} is not an accepted cell (iscell = 1) of this experiment.")
    if label is not None and label not in rc.MANUAL_LABELS:
        raise ValueError(f"A label set by hand is one of {', '.join(rc.MANUAL_LABELS)}, not {label!r}.")
    saved = rc.load_classes(source, fingerprint)
    rows = {} if saved is None else saved["rows"]
    cell = int(np.count_nonzero(mask[:roi]))
    row = rows.get(roi) or {"roi": roi, "cell": cell, "label": None, "auto": None, "why": None, "manual": None, **{name: None for name in rc.FEATURE_NAMES}}  # an ROI the automatic run has not seen
    row["manual"], row["label"] = label, label or row["auto"]
    rows[roi] = row
    rc.save_classes(source, fingerprint, rc.roi_settings(settings.defaults.get(g.ROI_CLASS_SETTINGS)) if saved is None else saved["thresholds"], list(rows.values()))
    return _entry(roi, cell, stat, row)


def export_manual_labels(settings, exp: dict) -> dict:
    """Add the labels set by hand in this experiment to the labelled dataset of the lab (one file in the data root, for training a classifier later); returns {exported, records,
    experiments, path}. A record of the same ROI of the same Suite2p result is replaced; nothing is removed, so labels cleared here later stay in the dataset until set again.
    Raises ValueError when the classification belongs to an earlier Suite2p result."""
    source = exp["source_path"]
    mask, stat = experiment_rois(source)
    fingerprint = rc.stat_fingerprint(stat)
    saved = rc.load_classes(source, fingerprint)
    experiment = exp["canonical_id"]
    exported_at = datetime.now().isoformat(timespec="seconds")
    new = []
    for roi, row in ({} if saved is None else saved["rows"]).items():
        if row["manual"] and mask[roi]:
            features = {name: row[name] for name in rc.FEATURE_NAMES} if row.get("npix") is not None else rc.rounded(rc.roi_features(stat[roi]["ypix"], stat[roi]["xpix"]))
            new.append({
                "experiment": experiment, "fingerprint": fingerprint, "roi": roi, "cell": row["cell"], "label": row["manual"], "auto": row["auto"], "features": features,
                "ypix": [int(v) for v in stat[roi]["ypix"]], "xpix": [int(v) for v in stat[roi]["xpix"]], "thresholds": saved["thresholds"], "exported_at": exported_at,
            })
    path = path_for(PathKind.ROI_LABEL_DATASET, settings.source_data_root)
    records = json.loads(path.read_text(encoding="utf-8"))["records"] if path.exists() else []
    replaced = {(record["experiment"], record["fingerprint"], record["roi"]) for record in new}  # a record of the same ROI of the same Suite2p result is replaced, nothing is removed
    merged = [record for record in records if (record["experiment"], record["fingerprint"], record["roi"]) not in replaced] + new
    if new:  # one file for all experiments: each record has what a classifier needs (the label, the pixels, the features and the experiment and Suite2p result it is from)
        write_json(path, {"version": 1, "records": sorted(merged, key=lambda record: (record["experiment"], record["fingerprint"], record["roi"]))})
    return {"exported": len(new), "records": len(merged), "experiments": len({record["experiment"] for record in merged}), "path": str(path)}


def roi_fov_png(settings, exp: dict) -> bytes:
    """The field of view of the ROI types tab: the saved mean image of the neuron channel, in the pixels of the Suite2p ROIs (full resolution), as a PNG."""
    mc_dir = path_for(PathKind.MOTION_CORRECTED_DIR, exp["source_path"])
    image_path = path_for(PathKind.MC_NEURON_MEAN_IMAGE, mc_dir)
    try:
        key = (str(image_path), int(image_path.stat().st_mtime_ns))
    except FileNotFoundError as exc:
        raise FileNotFoundError("The mean image of the neuron channel is not saved yet (the motion correction step writes it): rerun motion correction.") from exc
    png = caches.ROI_FOV_PNG.get(key)
    if png is None:
        mean = np.asarray(get_mean_images(mc_dir, spatial_bin=1)[1], dtype=np.float32)
        low, high = np.percentile(mean, FOV_PERCENTILES)
        buffer = io.BytesIO()
        Image.fromarray((np.clip((mean - low) / max(float(high - low), 1e-9), 0, 1) * 255).astype(np.uint8)).save(buffer, format="PNG")
        png = buffer.getvalue()
        caches.ROI_FOV_PNG.put(key, png)
    return png
