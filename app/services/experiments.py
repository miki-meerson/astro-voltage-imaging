from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import re
import shutil

from app.db import experiment_repository as repo, run_repository
from app.services.runs import ACTIVE_STATUSES, STA_RESULT_TYPES, STEPS, create_run, feeding_steps, hypothesis_for, launch_run, sta_steps
from app.services.suite2p_data import experiment_stat_cells, forget_suite2p
from app.settings import Settings
from utils.file_utils import path_within
from utils.hypotheses import hypothesis_name
from utils.suite2p_utils import roi_center
from utils.thorsync_utils import find_ts_h5_files, list_h5_datasets
from utils.paths_utils import PathKind, path_for
from utils import globals_utils as g

ANALYSIS_STEPS = tuple(step for step in STEPS if step != g.PREPROCESSING_STEP)  # the steps that need the confirmed Suite2p cells
STARTED_STATUSES = ("pending", "queued", "running")  # the statuses of a run that has a job or is about to start one
PAST_PREPROCESSING = (repo.STAGE_PREPROCESSING_REVIEW, repo.STAGE_AWAITING_SUITE2P, repo.STAGE_ANALYSIS_READY)  # the stages of an experiment whose preprocessing results exist
FEEDING_STEP_ORDER = (g.SPIKE_STEP, g.BEHAVIOR_STEP, g.ANATOMY_STEP)  # the steps start_analysis may create, in the order they are created
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


def behavior_ready(report: dict) -> bool:
    """Whether the experiment has behavior data (one imaging csv or a TS h5 file), from its validation report."""
    checks = report.get("checks", {})
    return bool(checks.get("behavior_ts", {}).get("count")) or checks.get("behavior_csv", {}).get("count") == 1


def experiment_steps(report: dict) -> tuple[str, ...]:
    """The steps an experiment must complete to count as COMPLETED: every step of the pipeline, except the behavior split of an experiment that has no behavior data (it
    could never complete)."""
    return tuple(step for step in STEPS if step != g.BEHAVIOR_STEP or behavior_ready(report))


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
    analysis_ready = all(checks[key]["exists"] for key in analysis_required)  # behavior data is not needed: only the behavioral result type reads it
    state = "valid" if analysis_ready else "needs_preprocessing"
    report = {"state": state, "checks": checks, "analysis_ready": analysis_ready}
    if not behavior_ready(report):
        checks["behavior_source"] = {
            "exists": False,
            "reason": "No imaging CSV or TS H5 behavior data was found. TS settings XML alone is enough for upload/preprocessing, but not for behavior analysis (the behavioral result type).",
        }
    return parsed, report


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


def launch_default_steps(settings: Settings, experiment_id: int, params: dict | None = None) -> None:
    """Start what a newly registered experiment needs: cluster preprocessing for a raw one (with the preprocessing defaults, changed by `params`); the behavior split for
    a processed one that has behavior data (as when Suite2p is confirmed: an experiment that arrives with Suite2p already confirmed never goes through that step). Every
    other analysis starts when the user asks for it (start_analysis)."""
    stage = repo.workflow_stage_of(settings, experiment_id)
    if stage == repo.STAGE_PREPROCESSING_PENDING:
        start_preprocessing(settings, experiment_id, params or {})
    elif stage == repo.STAGE_ANALYSIS_READY and behavior_ready(json.loads(repo.experiment_row(settings, experiment_id)["validation_report"] or "{}")):
        start_behavior_split(settings, experiment_id)


def start_analysis(settings: Settings, experiment_id: int, result_types, sta_options: dict | None = None, hypotheses=None) -> list[int]:
    """Start the analysis of ONE experiment for the marked result types (and only those); returns the ids of the runs it created.

    Needs the analysis-ready stage. Creates the feeding steps (spike, behavior, and anatomy when marked) that have no completed run and no run in
    progress yet: a completed run is never replaced (edited spike labels must stay). Then creates one STA run per hypothesis (not for a hypothesis
    that already has a run in progress), with the hypothesis settings and the marked result types: 'pending' and launched at once when every
    feeding step has completed, else 'waiting' (_launch_sta_if_ready in runs.py starts it when the last feeding step completes).
    sta_options: extra STA parameters of the request (for example the spatial bins); the hypothesis settings win over them.
    hypotheses: the names of the hypotheses to run (None: all of them); a rerun of one hypothesis passes its name."""
    stage = repo.workflow_stage_of(settings, experiment_id)
    defaults = settings.defaults

    if stage != repo.STAGE_ANALYSIS_READY:
        raise ValueError("Analysis can run only after preprocessing and Suite2p have been confirmed.")
    # the marked result types, once each in the order of STA_RESULT_TYPES (the page only offers known types and cannot submit none: form_rules.js)
    types = [name for name, _ in STA_RESULT_TYPES if name in set(result_types)]
    known = {hypothesis_name(step) for step in sta_steps()}
    if hypotheses is not None and not set(hypotheses) <= known:
        raise ValueError(f"Unknown hypothesis: {', '.join(sorted(set(hypotheses) - known))}.")
    needed = feeding_steps(types)
    completed = run_repository.steps_with_status(settings, experiment_id, ("completed",))
    in_progress = run_repository.steps_with_status(settings, experiment_id, ACTIVE_STATUSES)
    steps_to_create = needed - completed - in_progress
    feeding = [create_run(settings, experiment_id, step, defaults.get(step, {}), backend="local") for step in FEEDING_STEP_ORDER if step in steps_to_create]
    ready = needed <= completed
    # the STA defaults, the options of the request, the settings of the hypothesis, the marked result types (the hypothesis settings win)
    sta_defaults = {"anatomy_spatial_bin": 1, **defaults.get("sta", {}), **(sta_options or {})}
    sta = [create_run(settings, experiment_id, step, {**sta_defaults, **hypothesis_for(step).settings(), "result_types": types},
                      backend="cluster", status="pending" if ready else "waiting")
           for step in sta_steps() if step not in in_progress and (hypotheses is None or hypothesis_name(step) in hypotheses)]  # one run per chosen hypothesis that has no run in progress yet
    for run_id in feeding + (sta if ready else []):  # every STA run exists before a feeding run starts, so one that completes at once finds the waiting runs
        launch_run(settings, run_id)
    return feeding + sta


def start_spike_detection(settings: Settings, experiment_id: int) -> int:
    """Start the spike detection of the experiment's Suite2p cells as a new run; returns its id. A detection already in progress is not doubled."""
    if g.SPIKE_STEP in run_repository.steps_with_status(settings, experiment_id, ACTIVE_STATUSES):
        raise ValueError("Spike detection is already running for this experiment.")
    run_id = create_run(settings, experiment_id, g.SPIKE_STEP, settings.defaults.get(g.SPIKE_STEP, {}), backend="local")
    launch_run(settings, run_id)
    return run_id


def start_behavior_split(settings: Settings, experiment_id: int) -> int | None:
    """Start the behavior split of an experiment that has behavior data, unless one already completed or is in progress (its result does not depend on the cells, so a
    new Suite2p confirmation does not need a new one); returns the run id, or None when nothing was started."""
    if run_repository.steps_with_status(settings, experiment_id, ("completed", *ACTIVE_STATUSES)) & {g.BEHAVIOR_STEP}:
        return None
    run_id = create_run(settings, experiment_id, g.BEHAVIOR_STEP, settings.defaults.get(g.BEHAVIOR_STEP, {}), backend="local")
    launch_run(settings, run_id)
    return run_id


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
    start_spike_detection(settings, experiment_id)  # the cells are new (the old analysis runs were cancelled above): detect their spikes now; editing them is manual, afterwards
    if behavior_ready(report):
        start_behavior_split(settings, experiment_id)  # the behavior does not depend on the cells: split the movie by it once
    return len(neurons)


def _delete_or_quarantine(path: Path, settings: Settings) -> str:
    if not path.exists():
        return "missing"
    roots = [settings.output_root, settings.test_output_root, path_for(PathKind.SCIENTIFIC_DATABASE_ROOT, settings.project_root)]
    if not path_within(path, *roots):
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


def start_preprocessing(settings: Settings, experiment_id: int, params: dict) -> int:
    """Start a preprocessing run of an experiment (the preprocessing defaults changed by `params`) on the cluster; returns its id.

    For an experiment that is already past preprocessing (finished, or processed before: its results are in its folder) the new results replace the old ones, so everything
    that was made from them goes first: its runs, cells, confirmations and spike edits are deleted and it is put back to the start of preprocessing; Suite2p has to be
    run again by hand on the new movies. Raises ValueError while a preprocessing run is in progress (two runs write the same files and corrupt each other's results) and,
    for such an experiment, while any other run has started."""
    if g.PREPROCESSING_STEP in run_repository.steps_with_status(settings, experiment_id, ACTIVE_STATUSES):
        raise ValueError("Preprocessing is already running for this experiment: wait for it to finish (a second run would write the same files).")
    if repo.workflow_stage_of(settings, experiment_id) in PAST_PREPROCESSING:
        if run_repository.steps_with_status(settings, experiment_id, STARTED_STATUSES):  # a waiting run has no job yet: it is discarded with the rest of the app data
            raise ValueError("A run of this experiment is still in progress: wait for it, or cancel it, before processing the experiment again.")
        exp = repo.experiment_row(settings, experiment_id)
        report = {**validate_experiment(exp["source_path"])[1], "state": "needs_preprocessing", "analysis_ready": False}  # the folder as it is, not ready until processed again
        _, run_rows = repo.reset_experiment(settings, experiment_id, report, repo.STAGE_PREPROCESSING_PENDING, repo.REVIEW_UNREVIEWED)
        _remove_run_outputs(run_rows, exp["canonical_id"], settings)
    run_id = create_run(settings, experiment_id, g.PREPROCESSING_STEP, {**settings.defaults.get(g.PREPROCESSING_STEP, {}), **params}, backend="cluster")
    launch_run(settings, run_id)
    return run_id


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
