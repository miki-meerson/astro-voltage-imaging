from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import shutil
import sqlite3

import numpy as np

from app.db.core import tx
from app.settings import Settings
from utils.thorsync_utils import find_ts_h5_files, list_h5_datasets

DATE_BEHAVIOR_RE = re.compile(r"^(?P<date>\d{2}-\d{2}-\d{4})[_-](?P<behavior>.+)$")
Z_RE = re.compile(r"^z(?P<z>-?[^-\\\/]+)-x(?P<zoom>.+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedExperiment:
    cage: str; mouse_name: str; recording_date: str; behavior: str; fov: str; z_plane: str; zoom: str; canonical_id: str


def parse_experiment_path(path: str | Path) -> ParsedExperiment:
    p = Path(path)
    if len(p.parts) < 6:
        raise ValueError("Path is too short for expected experiment hierarchy.")
    zm = Z_RE.match(p.name)
    dm = DATE_BEHAVIOR_RE.match(p.parts[-3])
    if not zm:
        raise ValueError("Final folder must look like z<zplane>-x<zoom>, for example z18-x2 or z-18-x2.")
    if not dm:
        raise ValueError("Date/behavior folder must look like DD-MM-YYYY_behavior or DD-MM-YYYY-behavior.")
    cage, mouse, fov = p.parts[-5], p.parts[-4], p.parts[-2]
    date, behavior, z, zoom = dm.group("date"), dm.group("behavior"), zm.group("z"), zm.group("zoom")
    canonical = f"{cage}__{mouse}__{date}_{behavior}__{fov}__z{z}__x{zoom}"
    return ParsedExperiment(cage, mouse, date, behavior, fov, z, zoom, canonical)


def validate_experiment(path: str | Path, settings: Settings):
    p = Path(path)
    parsed = parse_experiment_path(p)
    required = {
        "experiment_folder": p, "raw_movie": p / "Image_scan_1_region_0_0.tif", "pipeline_results": p / "pipeline_results",
        "experiment_xml": p / "Experiment.xml", "suite2p_iscell": p / "pipeline_results/for_suite2p/suite2p/plane0/iscell.npy",
        "suite2p_stat": p / "pipeline_results/for_suite2p/suite2p/plane0/stat.npy", "suite2p_F": p / "pipeline_results/for_suite2p/suite2p/plane0/F.npy",
        "suite2p_ops": p / "pipeline_results/for_suite2p/suite2p/plane0/ops.npy", "pb_astro_movie": p / "pipeline_results/pb_corrected/astro_photobleaching_corrected.tif",
    }
    checks, missing = {}, []
    for key, rp in required.items():
        ok = rp.exists()
        checks[key] = {"path": str(rp), "exists": ok}
        if not ok:
            missing.append(key)
    behavior = sorted(p.glob("imaging_*.csv"))
    ts_h5 = find_ts_h5_files(p)
    checks["behavior_csv"] = {"count": len(behavior), "paths": [str(x) for x in behavior]}
    checks["behavior_ts"] = {"count": len(ts_h5), "paths": [str(x) for x in ts_h5], "preferred": bool(ts_h5)}
    if ts_h5:
        try:
            checks["behavior_ts"]["datasets"] = list_h5_datasets(ts_h5[0])
        except Exception as exc:
            checks["behavior_ts"]["datasets_error"] = str(exc)
    if len(ts_h5) > 1:
        missing.append("exactly_one_ts_h5")
    if not ts_h5 and len(behavior) != 1:
        missing.append("behavior_csv_or_ts_h5")
    if missing:
        raise ValueError("Missing or invalid required inputs: " + ", ".join(missing))
    return parsed, {"state": "valid", "checks": checks}


def discover_neurons(path: str | Path) -> list[dict]:
    suite2p = Path(path) / "pipeline_results/for_suite2p/suite2p/plane0"
    iscell = np.load(suite2p / "iscell.npy", allow_pickle=True)[:, 0].astype(bool)
    stat = np.load(suite2p / "stat.npy", allow_pickle=True)
    out = []
    for idx, roi in enumerate(stat[iscell]):
        ypix, xpix = np.asarray(roi.get("ypix", [])), np.asarray(roi.get("xpix", []))
        out.append({"pipeline_neuron_id": str(idx), "roi_source": str(suite2p), "roi_summary": {"n_pixels": int(len(ypix)), "mean_y": float(np.mean(ypix)) if len(ypix) else None, "mean_x": float(np.mean(xpix)) if len(xpix) else None}})
    return out


def add_experiment(path: str | Path, settings: Settings) -> int:
    p = Path(path)
    parsed, report = validate_experiment(path, settings)
    with tx(settings) as conn:
        removed = conn.execute(
            "SELECT 1 FROM removed_experiments WHERE source_path=? OR canonical_id=? LIMIT 1",
            (str(p), parsed.canonical_id),
        ).fetchone()
        if removed is not None:
            raise ValueError("This experiment was removed from the registry and is blocked from being re-added.")
        existing = conn.execute(
            "SELECT id FROM experiments WHERE source_path=? OR canonical_id=?",
            (str(p), parsed.canonical_id),
        ).fetchone()
        if existing is not None:
            return int(existing["id"])
    neurons = discover_neurons(path)
    with tx(settings) as conn:
        try:
            cur = conn.execute("""INSERT INTO experiments(canonical_id,source_path,cage,mouse_name,recording_date,behavior,fov,z_plane,zoom,validation_report) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                               (parsed.canonical_id, str(p), parsed.cage, parsed.mouse_name, parsed.recording_date, parsed.behavior, parsed.fov, parsed.z_plane, parsed.zoom, json.dumps(report)))
        except sqlite3.IntegrityError:
            existing = conn.execute(
                "SELECT id FROM experiments WHERE source_path=? OR canonical_id=?",
                (str(p), parsed.canonical_id),
            ).fetchone()
            if existing is None:
                raise
            return int(existing["id"])
        exp_id = int(cur.lastrowid)
        for n in neurons:
            conn.execute("INSERT INTO neurons(experiment_id,pipeline_neuron_id,roi_source,roi_summary) VALUES(?,?,?,?)", (exp_id, n["pipeline_neuron_id"], n["roi_source"], json.dumps(n["roi_summary"])))
    return exp_id


def scan_parent(parent: str | Path, settings: Settings) -> dict:
    from datetime import datetime
    parent = Path(parent)
    summary = {"added": [], "existing": [], "skipped_removed": [], "invalid": []}
    started = datetime.now().isoformat(timespec="seconds")
    with tx(settings) as conn:
        scan_id = conn.execute("INSERT INTO scan_history(parent_path,started_at) VALUES(?,?)", (str(parent), started)).lastrowid
        existing = {r["source_path"] for r in conn.execute("SELECT source_path FROM experiments")}
        removed_paths = {r["source_path"] for r in conn.execute("SELECT source_path FROM removed_experiments")}
        removed_ids = {r["canonical_id"] for r in conn.execute("SELECT canonical_id FROM removed_experiments")}
    for candidate in parent.rglob("z*-x*"):
        if candidate.is_dir():
            if str(candidate) in existing:
                summary["existing"].append(str(candidate))
            elif str(candidate) in removed_paths:
                summary["skipped_removed"].append(str(candidate))
            else:
                try:
                    parsed = parse_experiment_path(candidate)
                    if parsed.canonical_id in removed_ids:
                        summary["skipped_removed"].append(str(candidate))
                    else:
                        summary["added"].append({"path": str(candidate), "experiment_id": add_experiment(candidate, settings)})
                except Exception as exc:
                    summary["invalid"].append({"path": str(candidate), "reason": str(exc)})
    completed = datetime.now().isoformat(timespec="seconds")
    with tx(settings) as conn:
        conn.execute("UPDATE scan_history SET completed_at=?,summary_json=? WHERE id=?", (completed, json.dumps(summary), scan_id))
        conn.execute("INSERT OR REPLACE INTO app_settings(key,value) VALUES('last_scan_time',?)", (completed,))
    _launch_default_steps_for_added(summary, settings)
    return summary


def _launch_default_steps_for_added(summary: dict, settings: Settings) -> None:
    from app.services.runs import create_run, launch_run

    for item in summary.get("added", []):
        exp_id = item.get("experiment_id")
        if exp_id is None:
            continue
        for step, params in settings.raw.get("defaults", {}).items():
            if step in {"spike", "behavior", "anatomy"}:
                rid = create_run(settings, int(exp_id), step, params, backend="local")
                launch_run(settings, rid)


def _runtime_owned(path: Path, settings: Settings) -> bool:
    try:
        target = path.resolve()
    except OSError:
        target = path.absolute()
    roots = [
        settings.output_root.resolve(),
        settings.test_output_root.resolve(),
        (settings.project_root / "runtime" / "scientific_database").resolve(),
    ]
    return any(target == root or root in target.parents for root in roots)


def _delete_or_quarantine(path: Path, settings: Settings) -> str:
    if not path.exists():
        return "missing"
    if not _runtime_owned(path, settings):
        raise ValueError(f"Refusing to remove non-runtime path: {path}")
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return "removed"
    except PermissionError:
        quarantine = path.with_name(f"_locked_removed_{path.name}")
        suffix = 2
        while quarantine.exists():
            quarantine = path.with_name(f"_locked_removed_{path.name}_{suffix}")
            suffix += 1
        path.rename(quarantine)
        return f"locked; moved to {quarantine}"


def remove_experiment_from_registry(experiment_id: int, settings: Settings) -> dict:
    with tx(settings) as conn:
        exp = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if exp is None:
            raise ValueError("Experiment not found.")
        run_rows = list(conn.execute("SELECT * FROM analysis_runs WHERE experiment_id=?", (experiment_id,)))
        canonical_id = exp["canonical_id"]
        source_path = exp["source_path"]
        conn.execute(
            "INSERT OR REPLACE INTO removed_experiments(canonical_id,source_path,note) VALUES(?,?,?)",
            (canonical_id, source_path, "Removed from GUI experiment list."),
        )
        conn.execute("DELETE FROM experiments WHERE id=?", (experiment_id,))

    targets: list[Path] = []
    for run in run_rows:
        if run["output_dir"]:
            targets.append(Path(run["output_dir"]))
        elif run["stdout_path"]:
            targets.append(Path(run["stdout_path"]).parent)
    targets.append(settings.project_root / "runtime" / "scientific_database" / canonical_id)

    seen = set()
    cleaned = []
    for target in targets:
        key = str(target)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"path": key, "result": _delete_or_quarantine(target, settings)})
    return {"canonical_id": canonical_id, "source_path": source_path, "cleaned": cleaned}
