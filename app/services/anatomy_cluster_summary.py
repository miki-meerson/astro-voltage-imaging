"""Experiment-level view of the significant anatomy-configuration clusters.

Selection (latest completed STA run of the experiment):
  1. the cell has at least one significant cluster in the anatomy configuration (not the full-FOV one),
  2. the cell is not motion-suspected (motion_sta.npz flag is False). Cells without motion diagnostics are left out and counted.

For every significant anatomy cluster of the selected cells it returns the cluster's mean trace (dF/F, -50..+50 frames from the
spike; the same trace as on the cell page) and the offset of the cluster centre from the centre of the cell's ROI, in full-resolution pixels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.services import polarity
from app.services.run_views import latest_completed_run
from app.services.spikes import suite2p_stat_cells
from app.services.visualizations import _load_cluster_result, _sta_dir

CACHE_NAME = "anatomy_cluster_summary.json"
CACHE_VERSION = 1


def _roi_center_fullres(stat_cells, cell_id: int) -> tuple[float, float]:
    roi = stat_cells[int(cell_id)]
    return float(np.mean(roi["ypix"])), float(np.mean(roi["xpix"]))


def _cluster_center_fullres(mask2d: np.ndarray, crop, spatial_bin: int) -> tuple[float, float]:
    ys, xs = np.nonzero(mask2d)
    y0, x0 = (int(crop[0]), int(crop[2])) if crop is not None else (0, 0)
    b = int(spatial_bin)
    # bin units -> full-resolution pixel coordinates (pixel centres)
    return (y0 + float(ys.mean()) + 0.5) * b - 0.5, (x0 + float(xs.mean()) + 0.5) * b - 0.5


def _signature(sta_dir: Path) -> list:
    csv = sta_dir / "sta_maps_batch_summary.csv"
    motion = sorted(sta_dir.glob("cell_*/motion_sta.npz"))
    return [CACHE_VERSION, csv.stat().st_mtime_ns if csv.exists() else 0, len(motion), motion[0].stat().st_mtime_ns if motion else 0]


def _summary(settings, conn, experiment_id: int) -> dict:
    import pandas as pd

    run = latest_completed_run(conn, int(experiment_id), "sta")
    if run is None:
        return {"available": False, "message": "No completed STA run for this experiment."}
    sta_dir = _sta_dir(settings, run)
    csv_path = sta_dir / "sta_maps_batch_summary.csv"
    if not csv_path.exists():
        return {"available": False, "message": f"STA summary was not found for run {int(run['id'])}."}

    cache_path = sta_dir / CACHE_NAME
    sig_now = _signature(sta_dir)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("signature") == sig_now:
                return cached["payload"]
        except Exception:
            pass

    df = pd.read_csv(csv_path)
    anat = df[(df["spike_group"].astype(str) == "anatomy") & (df["n_significant_clusters"].fillna(0) > 0)]
    cell_ids = sorted(int(c) for c in anat["cell_id"].unique())

    exp = conn.execute("SELECT source_path FROM experiments WHERE id=?", (int(experiment_id),)).fetchone()
    stat_cells = suite2p_stat_cells(exp["source_path"])
    db_ids = {int(r["pipeline_neuron_id"]): int(r["id"]) for r in conn.execute("SELECT id, pipeline_neuron_id FROM neurons WHERE experiment_id=?", (int(experiment_id),))}

    flagged, no_motion, used = [], [], []
    clusters = []
    for cid in cell_ids:
        mpath = sta_dir / f"cell_{cid}" / "motion_sta.npz"
        if not mpath.exists():
            no_motion.append(cid)
            continue
        with np.load(mpath, allow_pickle=True) as m:
            if bool(m["flag"]):
                flagged.append(cid)
                continue
        npz = sta_dir / f"cell_{cid}" / "astro_sta_cluster_permutation_anatomy.npz"
        if not npz.exists() or cid >= len(stat_cells):
            continue
        _, _, sig, meta = _load_cluster_result(npz)
        ry, rx = _roi_center_fullres(stat_cells, cid)
        used_cell = False
        for cluster, mask in sig:
            wide = cluster.get("wide_trace")
            if wide is None:
                continue
            t, trace = wide
            cy, cx = _cluster_center_fullres(np.asarray(mask, dtype=bool), meta.get("anatomy_crop_bounds"), int(meta["spatial_bin"]))
            m3 = np.asarray(cluster["mask3d"], dtype=bool)
            frames = np.flatnonzero(m3.any(axis=(1, 2)))
            aw = meta["tested_window"]
            clusters.append({
                "cell": cid, "neuron_id": db_ids.get(cid),
                "sign": polarity.polarity_from_sign(str(cluster.get("sign", ""))),
                "p": float(cluster.get("p_value", float("nan"))), "mass": float(cluster["mass"]),
                "n_px": int(np.asarray(mask).sum()), "bin": int(meta["spatial_bin"]),
                "first_rel": int(aw[0] + frames.min()), "last_rel": int(aw[0] + frames.max()),
                "dx": float(cx - rx), "dy": float(cy - ry), "dist": float(np.hypot(cx - rx, cy - ry)),
                "t": [int(v) for v in t], "trace": [round(float(v), 6) for v in trace],
            })
            used_cell = True
        if used_cell:
            used.append(cid)

    payload = {
        "available": True, "run_id": int(run["id"]), "cells": used, "clusters": clusters,
        "counts": {
            "cells_with_sig_anatomy_cluster": len(cell_ids), "motion_flagged": len(flagged), "no_motion_diagnostics": len(no_motion),
            "cells_shown": len(used), "clusters_shown": len(clusters),
        },
        "excluded": {"motion_flagged": flagged, "no_motion_diagnostics": no_motion},
    }
    try:
        cache_path.write_text(json.dumps({"signature": sig_now, "payload": payload}), encoding="utf-8")
    except Exception:
        pass
    return payload


def anatomy_cluster_summary(settings, conn, experiment_id: int) -> dict:
    """The summary (cached) plus, when a cluster job has produced it, the real-vs-fake group test of the same run."""
    payload = dict(_summary(settings, conn, experiment_id))
    if payload.get("available"):
        run = latest_completed_run(conn, int(experiment_id), "sta")
        path = _sta_dir(settings, run) / "anatomy_group_test.json"
        if path.exists():
            try:
                payload["group_test"] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    return payload
