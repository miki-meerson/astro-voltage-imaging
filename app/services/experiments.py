from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import shutil
import sqlite3

import numpy as np

from app.db.core import tx
from app.services.spikes import suite2p_plane0, suite2p_stat_cells
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


def validate_experiment(path: str | Path):
    p = Path(path)
    parsed = parse_experiment_path(p)
    raw_movie = _raw_movie_path(p)
    required = {"experiment_folder": p, "raw_movie": raw_movie, "experiment_xml": p / "Experiment.xml"}
    analysis_required = {
        "pipeline_results": p / "pipeline_results",
        "suite2p_iscell": p / "pipeline_results/for_suite2p/suite2p/plane0/iscell.npy",
        "suite2p_stat": p / "pipeline_results/for_suite2p/suite2p/plane0/stat.npy",
        "suite2p_F": p / "pipeline_results/for_suite2p/suite2p/plane0/F.npy",
        "suite2p_ops": p / "pipeline_results/for_suite2p/suite2p/plane0/ops.npy",
        "pb_astro_movie": p / "pipeline_results/pb_corrected/astro_photobleaching_corrected.tif",
        "pb_neuron_movie": p / "pipeline_results/pb_corrected/neuron_photobleaching_corrected.tif",
    }
    checks, missing = {}, []
    for key, rp in {**required, **analysis_required}.items():
        ok = rp.exists()
        checks[key] = {"path": str(rp), "exists": ok}
        if key in required and not ok:
            missing.append(key)
    behavior = sorted(p.glob("imaging_*.csv"))
    ts_h5 = find_ts_h5_files(p)
    ts_dirs = sorted([x for x in p.iterdir() if x.is_dir() and x.name.lower().startswith("ts")])
    ts_xml = sorted([x / "ThorRealTimeDataSettings.xml" for x in ts_dirs if (x / "ThorRealTimeDataSettings.xml").exists()])
    checks["behavior_csv"] = {"count": len(behavior), "paths": [str(x) for x in behavior]}
    checks["behavior_ts"] = {
        "count": len(ts_h5),
        "paths": [str(x) for x in ts_h5],
        "settings_xml_count": len(ts_xml),
        "settings_xml_paths": [str(x) for x in ts_xml],
        "preferred": bool(ts_h5),
    }
    if ts_h5:
        try:
            checks["behavior_ts"]["datasets"] = list_h5_datasets(ts_h5[0])
        except Exception as exc:
            checks["behavior_ts"]["datasets_error"] = str(exc)
    if len(ts_h5) > 1:
        missing.append("exactly_one_ts_h5")
    if missing:
        raise ValueError("Missing or invalid required inputs: " + ", ".join(missing))
    behavior_ready = bool(ts_h5) or len(behavior) == 1
    analysis_ready = behavior_ready and all(checks[key]["exists"] for key in analysis_required)
    if not behavior_ready:
        checks["behavior_source"] = {
            "exists": False,
            "reason": "No imaging CSV or TS H5 behavior data was found. TS settings XML alone is enough for upload/preprocessing, but not for behavior analysis.",
        }
    state = "valid" if analysis_ready else "needs_preprocessing"
    return parsed, {"state": state, "checks": checks, "analysis_ready": analysis_ready}


def _raw_movie_path(path: Path) -> Path:
    for name in ("Image_scan_1_region_0_0.tif", "Image_scan_1_region_0_0.raw", "raw_video.tif"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raw_candidates = sorted(path.glob("*.raw")) + sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
    return raw_candidates[0] if raw_candidates else path / "Image_scan_1_region_0_0.tif"


def discover_neurons(path: str | Path) -> list[dict]:
    out = []
    for idx, roi in enumerate(suite2p_stat_cells(path)):
        ypix, xpix = np.asarray(roi.get("ypix", [])), np.asarray(roi.get("xpix", []))
        out.append({"pipeline_neuron_id": str(idx), "roi_source": str(suite2p_plane0(path)), "roi_summary": {"n_pixels": int(len(ypix)), "mean_y": float(np.mean(ypix)) if len(ypix) else None, "mean_x": float(np.mean(xpix)) if len(xpix) else None}})
    return out


def _insert_neurons(conn, experiment_id: int, neurons: list[dict]) -> None:
    for n in neurons:
        conn.execute(
            "INSERT INTO neurons(experiment_id,pipeline_neuron_id,roi_source,roi_summary) VALUES(?,?,?,?)",
            (experiment_id, n["pipeline_neuron_id"], n["roi_source"], json.dumps(n["roi_summary"])),
        )


def add_experiment(path: str | Path, settings: Settings) -> int:
    p = Path(path)
    parsed, report = validate_experiment(path)
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
    analysis_ready = bool(report.get("analysis_ready"))
    neurons = discover_neurons(path) if analysis_ready else []
    workflow_stage = "analysis_ready" if analysis_ready else "preprocessing_pending"
    preprocessing_review_state = "confirmed" if analysis_ready else "unreviewed"
    suite2p_review_state = "confirmed" if analysis_ready else "unreviewed"
    with tx(settings) as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO experiments(
                  canonical_id,source_path,cage,mouse_name,recording_date,behavior,fov,z_plane,zoom,
                  validation_state,validation_report,workflow_stage,preprocessing_review_state,suite2p_review_state
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    parsed.canonical_id,
                    str(p),
                    parsed.cage,
                    parsed.mouse_name,
                    parsed.recording_date,
                    parsed.behavior,
                    parsed.fov,
                    parsed.z_plane,
                    parsed.zoom,
                    report["state"],
                    json.dumps(report),
                    workflow_stage,
                    preprocessing_review_state,
                    suite2p_review_state,
                ),
            )
        except sqlite3.IntegrityError:
            existing = conn.execute(
                "SELECT id FROM experiments WHERE source_path=? OR canonical_id=?",
                (str(p), parsed.canonical_id),
            ).fetchone()
            if existing is None:
                raise
            return int(existing["id"])
        exp_id = int(cur.lastrowid)
        _insert_neurons(conn, exp_id, neurons)
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
    for item in summary["added"]:
        launch_default_steps(settings, int(item["experiment_id"]))
    return summary


def launch_default_steps(settings: Settings, experiment_id: int) -> None:
    """Start what a newly registered experiment needs: cluster preprocessing for a raw one, spike / behavior / anatomy for a processed one."""
    from app.services.runs import create_run, launch_run

    with tx(settings) as conn:
        exp = conn.execute("SELECT workflow_stage FROM experiments WHERE id=?", (experiment_id,)).fetchone()
    if exp is not None and exp["workflow_stage"] == "preprocessing_pending":
        params = settings.raw.get("defaults", {}).get("preprocessing", {})
        launch_run(settings, create_run(settings, experiment_id, "preprocessing", params, backend="cluster"))
    elif exp is not None and exp["workflow_stage"] == "analysis_ready":
        launch_analysis_defaults(settings, experiment_id)


def launch_analysis_defaults(settings: Settings, experiment_id: int) -> list[int]:
    from app.services.runs import create_run, launch_run

    created = []
    for step, params in settings.raw.get("defaults", {}).items():
        if step in {"spike", "behavior", "anatomy"}:
            rid = create_run(settings, experiment_id, step, params, backend="local")
            launch_run(settings, rid)
            created.append(rid)
    return created


def confirm_preprocessing(settings: Settings, experiment_id: int) -> None:
    with tx(settings) as conn:
        conn.execute(
            """
            UPDATE experiments
            SET preprocessing_review_state='confirmed',
                workflow_stage='awaiting_suite2p',
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (experiment_id,),
        )


def confirm_suite2p(settings: Settings, experiment_id: int) -> int:
    with tx(settings) as conn:
        exp = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if exp is None:
            raise ValueError("Experiment not found.")
    parsed, report = validate_experiment(exp["source_path"])
    if not report.get("analysis_ready"):
        raise ValueError("Suite2p/manual outputs are not complete yet.")
    neurons = discover_neurons(exp["source_path"])
    with tx(settings) as conn:
        conn.execute(
            """
            UPDATE analysis_runs
            SET status='cancelled',
                ended_at=CURRENT_TIMESTAMP,
                review_note='Cancelled because Suite2p cells were refreshed.'
            WHERE experiment_id=?
              AND step IN ('spike','behavior','anatomy','sta')
              AND status IN ('pending','queued','running','completed','failed','blocked')
            """,
            (experiment_id,),
        )
        conn.execute("DELETE FROM neurons WHERE experiment_id=?", (experiment_id,))
        _insert_neurons(conn, experiment_id, neurons)
        conn.execute(
            """
            UPDATE experiments
            SET validation_state='valid',
                validation_report=?,
                suite2p_review_state='confirmed',
                workflow_stage='analysis_ready',
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (json.dumps(report), experiment_id),
        )
    return len(neurons)


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


def _remove_run_outputs(run_rows, canonical_id: str, settings: Settings) -> list[dict]:
    """Delete (or quarantine, when locked) the output folder of every run and the confirmed-cell folder of the experiment."""
    targets: list[Path] = []
    for run in run_rows:
        if run["output_dir"]:
            targets.append(Path(run["output_dir"]))
        elif run["stdout_path"]:
            targets.append(Path(run["stdout_path"]).parent)
    targets.append(settings.project_root / "runtime" / "scientific_database" / canonical_id)
    cleaned, seen = [], set()
    for target in targets:
        if str(target) not in seen:
            seen.add(str(target))
            cleaned.append({"path": str(target), "result": _delete_or_quarantine(target, settings)})
    return cleaned


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

    cleaned = _remove_run_outputs(run_rows, canonical_id, settings)
    return {"canonical_id": canonical_id, "source_path": source_path, "cleaned": cleaned}


def reset_experiment_app_data(experiment_id: int, settings: Settings) -> dict:
    with tx(settings) as conn:
        exp = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if exp is None:
            raise ValueError("Experiment not found.")
        run_rows = list(conn.execute("SELECT * FROM analysis_runs WHERE experiment_id=?", (experiment_id,)))
        canonical_id = exp["canonical_id"]
        source_path = exp["source_path"]
        try:
            _, report = validate_experiment(source_path)
        except Exception as exc:
            report = {"state": "invalid", "checks": {}, "analysis_ready": False, "reset_validation_error": str(exc)}
        analysis_ready = bool(report.get("analysis_ready"))
        workflow_stage = "awaiting_suite2p" if analysis_ready else "preprocessing_pending"
        preprocessing_review_state = "confirmed" if analysis_ready else "unreviewed"

        counts = {
            "runs": conn.execute("SELECT COUNT(*) AS n FROM analysis_runs WHERE experiment_id=?", (experiment_id,)).fetchone()["n"],
            "neurons": conn.execute("SELECT COUNT(*) AS n FROM neurons WHERE experiment_id=?", (experiment_id,)).fetchone()["n"],
            "confirmed_neurons": conn.execute("SELECT COUNT(*) AS n FROM confirmed_neurons WHERE experiment_id=?", (experiment_id,)).fetchone()["n"],
            "metadata_overrides": conn.execute("SELECT COUNT(*) AS n FROM metadata_overrides WHERE experiment_id=?", (experiment_id,)).fetchone()["n"],
        }
        conn.execute("DELETE FROM confirmed_neurons WHERE experiment_id=?", (experiment_id,))
        conn.execute("DELETE FROM metadata_overrides WHERE experiment_id=?", (experiment_id,))
        conn.execute("DELETE FROM analysis_runs WHERE experiment_id=?", (experiment_id,))
        conn.execute("DELETE FROM neurons WHERE experiment_id=?", (experiment_id,))
        conn.execute(
            """
            UPDATE experiments
            SET validation_state=?,
                validation_report=?,
                workflow_stage=?,
                preprocessing_review_state=?,
                suite2p_review_state='unreviewed',
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (
                report["state"],
                json.dumps(report),
                workflow_stage,
                preprocessing_review_state,
                experiment_id,
            ),
        )

    cleaned = _remove_run_outputs(run_rows, canonical_id, settings)
    return {
        "canonical_id": canonical_id,
        "source_path": source_path,
        "counts": counts,
        "cleaned": cleaned,
        "workflow_stage": workflow_stage,
    }
