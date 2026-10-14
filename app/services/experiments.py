from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import shutil

from app.db import experiment_repository as repo, run_repository
from app.services.runs import ACTIVE_STATUSES, STEPS, create_run, feeding_steps, hypothesis_for, launch_run, sta_steps
from app.services.sta_results import STA_RESULT_TYPES
from app.services.suite2p_data import experiment_stat_cells, forget_suite2p
from app.settings import Settings
from utils.suite2p_utils import roi_center
from utils.thorsync_utils import find_ts_h5_files, list_h5_datasets
from utils.paths_utils import PathKind, path_for
from utils import globals_utils as g

PREPROCESSING_STEP = "preprocessing"
ANALYSIS_STEPS = tuple(step for step in STEPS if step != PREPROCESSING_STEP)  # the steps that need the confirmed Suite2p cells
FEEDING_STEP_ORDER = ("spike", "behavior", "anatomy")  # the steps start_analysis may create, in the order they are created
EXPECTED_ERRORS = (ValueError, OSError)  # a bad experiment folder: wrong name, missing or unreadable file

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
    required = {"experiment_folder": p, "raw_movie": path_for(PathKind.RAW_MOVIE, p), "experiment_xml": path_for(PathKind.EXPERIMENT_XML, p)}
    plane0 = path_for(PathKind.SUITE2P_PLANE0, p)
    analysis_required = {
        "pipeline_results": path_for(PathKind.PIPELINE_RESULTS, p),
        "suite2p_iscell": path_for(PathKind.SUITE2P_ISCELL, plane0),
        "suite2p_stat": path_for(PathKind.SUITE2P_STAT, plane0),
        "suite2p_F": path_for(PathKind.SUITE2P_F, plane0),
        "suite2p_ops": path_for(PathKind.SUITE2P_OPS, plane0),
        "pb_astro_movie": path_for(PathKind.ASTRO_PB_MOVIE, p),
        "pb_neuron_movie": path_for(PathKind.NEURON_PB_MOVIE, p),
    }
    checks, missing = {}, []
    for key, rp in {**required, **analysis_required}.items():
        ok = rp.exists()
        checks[key] = {"path": str(rp), "exists": ok}
        if key in required and not ok:
            missing.append(key)
    behavior = sorted(p.glob(g.IMAGING_CSV_GLOB))
    ts_h5 = find_ts_h5_files(p)
    ts_dirs = sorted([x for x in p.iterdir() if x.is_dir() and x.name.lower().startswith(g.TS_DIR_PREFIX)])
    ts_xml = sorted([x / g.THOR_SETTINGS_XML for x in ts_dirs if (x / g.THOR_SETTINGS_XML).exists()])
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
        except OSError as exc:  # h5py: the file is locked, truncated or not an HDF5 file
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


def discover_neurons(path: str | Path) -> list[dict]:
    out = []
    for idx, roi in enumerate(experiment_stat_cells(path)):
        mean_y, mean_x = roi_center(roi)
        out.append({"pipeline_neuron_id": str(idx), "roi_source": str(path_for(PathKind.SUITE2P_PLANE0, path)), "roi_summary": {"n_pixels": len(roi["ypix"]), "mean_y": mean_y, "mean_x": mean_x}})
    return out


def add_experiment(path: str | Path, settings: Settings) -> int:
    p = Path(path)
    parsed, report = validate_experiment(path)
    if repo.is_removed_from_registry(settings, str(p), parsed.canonical_id):
        raise ValueError("This experiment was removed from the registry and is blocked from being re-added.")
    existing = repo.registered_experiment_id(settings, str(p), parsed.canonical_id)
    if existing is not None:
        return existing
    analysis_ready = bool(report.get("analysis_ready"))
    neurons = discover_neurons(path) if analysis_ready else []
    review_state = repo.REVIEW_CONFIRMED if analysis_ready else repo.REVIEW_UNREVIEWED
    return repo.insert_experiment(
        settings,
        {"canonical_id": parsed.canonical_id, "source_path": str(p), "cage": parsed.cage, "mouse_name": parsed.mouse_name, "recording_date": parsed.recording_date,
         "behavior": parsed.behavior, "fov": parsed.fov, "z_plane": parsed.z_plane, "zoom": parsed.zoom},
        report,
        repo.STAGE_ANALYSIS_READY if analysis_ready else repo.STAGE_PREPROCESSING_PENDING,
        review_state,
        review_state,
        neurons,
    )


def scan_parent(parent: str | Path, settings: Settings) -> dict:
    parent = Path(parent)
    summary = {"added": [], "existing": [], "skipped_removed": [], "invalid": []}
    scan_id = repo.start_scan(settings, str(parent), datetime.now().isoformat(timespec="seconds"))
    existing, removed_paths, removed_ids = repo.registry_snapshot(settings)
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
                except EXPECTED_ERRORS as exc:
                    summary["invalid"].append({"path": str(candidate), "reason": str(exc)})
    repo.finish_scan(settings, scan_id, datetime.now().isoformat(timespec="seconds"), summary)
    for item in summary["added"]:
        launch_default_steps(settings, int(item["experiment_id"]))
    return summary


def launch_default_steps(settings: Settings, experiment_id: int) -> None:
    """Start what a newly registered experiment needs: cluster preprocessing for a raw one. A processed one starts nothing: every analysis
    after preprocessing starts when the user asks for it (start_analysis)."""
    if repo.workflow_stage_of(settings, experiment_id) == repo.STAGE_PREPROCESSING_PENDING:
        params = settings.raw.get("defaults", {}).get(PREPROCESSING_STEP, {})
        launch_run(settings, create_run(settings, experiment_id, PREPROCESSING_STEP, params, backend="cluster"))


def _marked_result_types(result_types) -> list[str]:
    """The marked result types, once each in the order of STA_RESULT_TYPES. The page only offers known types and cannot submit none (form_rules.js)."""
    marked = set(result_types)
    return [name for name, _ in STA_RESULT_TYPES if name in marked]


def _sta_run_params(settings: Settings, step: str, result_types: list[str], options: dict) -> dict:
    """The parameters of one hypothesis run: the STA defaults, the options of the request, the settings of the hypothesis, the marked result types."""
    params = dict(settings.raw.get("defaults", {}).get("sta", {}))
    params.setdefault("anatomy_spatial_bin", 1)
    return {**params, **options, **hypothesis_for(step).settings(), "result_types": result_types}


def start_analysis(settings: Settings, experiment_id: int, result_types, sta_options: dict | None = None) -> list[int]:
    """Start the analysis of ONE experiment for the marked result types (and only those); returns the ids of the runs it created.

    Needs the analysis-ready stage. Creates the feeding steps (spike, behavior, and anatomy when marked) that have no completed run and no run in
    progress yet: a completed run is never replaced (edited spike labels must stay). Then creates one STA run per hypothesis (not for a hypothesis
    that already has a run in progress), with the hypothesis settings and the marked result types: 'pending' and launched at once when every
    feeding step has completed, else 'waiting' (_launch_sta_if_ready in runs.py starts it when the last feeding step completes).
    sta_options: extra STA parameters of the request (for example the spatial bins); the hypothesis settings win over them."""
    stage = repo.workflow_stage_of(settings, experiment_id)
    if stage is None:
        raise ValueError("Experiment not found.")
    if stage != repo.STAGE_ANALYSIS_READY:
        raise ValueError("Analysis can run only after preprocessing and Suite2p have been confirmed.")
    types = _marked_result_types(result_types)
    needed = feeding_steps(types)
    completed = run_repository.steps_with_status(settings, experiment_id, ("completed",))
    in_progress = run_repository.steps_with_status(settings, experiment_id, ACTIVE_STATUSES)
    feeding = _create_feeding_runs(settings, experiment_id, needed - completed - in_progress)
    ready = needed <= completed
    sta = _create_sta_runs(settings, experiment_id, types, sta_options or {}, "pending" if ready else "waiting", in_progress)
    for run_id in feeding + (sta if ready else []):  # every STA run exists before a feeding run starts, so one that completes at once finds the waiting runs
        launch_run(settings, run_id)
    return feeding + sta


def _create_feeding_runs(settings: Settings, experiment_id: int, steps: set[str]) -> list[int]:
    defaults = settings.raw.get("defaults", {})
    return [create_run(settings, experiment_id, step, defaults.get(step, {}), backend="local") for step in FEEDING_STEP_ORDER if step in steps]


def _create_sta_runs(settings: Settings, experiment_id: int, result_types: list[str], options: dict, status: str, in_progress: set[str]) -> list[int]:
    """One run per hypothesis that has no run in progress yet."""
    return [create_run(settings, experiment_id, step, _sta_run_params(settings, step, result_types, options), backend="cluster", status=status)
            for step in sta_steps() if step not in in_progress]


def confirm_preprocessing(settings: Settings, experiment_id: int) -> None:
    repo.confirm_preprocessing(settings, experiment_id)


def confirm_suite2p(settings: Settings, experiment_id: int) -> int:
    exp = repo.experiment_row(settings, experiment_id)
    if exp is None:
        raise ValueError("Experiment not found.")
    parsed, report = validate_experiment(exp["source_path"])
    if not report.get("analysis_ready"):
        raise ValueError("Suite2p/manual outputs are not complete yet.")
    forget_suite2p(exp["source_path"])  # the user says the extraction is done: read the Suite2p files again
    neurons = discover_neurons(exp["source_path"])
    repo.confirm_suite2p(settings, experiment_id, neurons, report, ANALYSIS_STEPS)
    return len(neurons)


def _runtime_owned(path: Path, settings: Settings) -> bool:
    try:
        target = path.resolve()
    except OSError:
        target = path.absolute()
    roots = [
        settings.output_root.resolve(),
        settings.test_output_root.resolve(),
        path_for(PathKind.SCIENTIFIC_DATABASE_ROOT, settings.project_root).resolve(),
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
    targets.append(path_for(PathKind.SCIENTIFIC_DATABASE_EXPERIMENT, settings.project_root, canonical_id=canonical_id))
    cleaned, seen = [], set()
    for target in targets:
        if str(target) not in seen:
            seen.add(str(target))
            cleaned.append({"path": str(target), "result": _delete_or_quarantine(target, settings)})
    return cleaned


def remove_experiment_from_registry(experiment_id: int, settings: Settings) -> dict:
    exp, run_rows = repo.remove_experiment(settings, experiment_id, "Removed from GUI experiment list.")
    cleaned = _remove_run_outputs(run_rows, exp["canonical_id"], settings)
    return {"canonical_id": exp["canonical_id"], "source_path": exp["source_path"], "cleaned": cleaned}


def reset_experiment_app_data(experiment_id: int, settings: Settings) -> dict:
    exp = repo.experiment_row(settings, experiment_id)
    if exp is None:
        raise ValueError("Experiment not found.")
    try:
        _, report = validate_experiment(exp["source_path"])
    except EXPECTED_ERRORS as exc:
        report = {"state": "invalid", "checks": {}, "analysis_ready": False, "reset_validation_error": str(exc)}
    analysis_ready = bool(report.get("analysis_ready"))
    workflow_stage = repo.STAGE_AWAITING_SUITE2P if analysis_ready else repo.STAGE_PREPROCESSING_PENDING
    preprocessing_review_state = repo.REVIEW_CONFIRMED if analysis_ready else repo.REVIEW_UNREVIEWED
    counts, run_rows = repo.reset_experiment(settings, experiment_id, report, workflow_stage, preprocessing_review_state)
    cleaned = _remove_run_outputs(run_rows, exp["canonical_id"], settings)
    return {
        "canonical_id": exp["canonical_id"],
        "source_path": exp["source_path"],
        "counts": counts,
        "cleaned": cleaned,
        "workflow_stage": workflow_stage,
    }
