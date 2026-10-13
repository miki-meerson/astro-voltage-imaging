"""Every path of the experiment folder and of an STA run, built in ONE place.

Design: a path is a PathKind (which file or folder it is) hung on an anchor (the folder it is relative to) with a name template made from
utils/globals_utils.py. One function, path_for(kind, anchor_path, **fields), builds any of them; there is no function per file. To add a
file, add one member to PathKind and one line to _REGISTRY. Importable without Flask or the app (the cluster jobs use it).

Anchors: EXPERIMENT (the experiment folder), RUN (the folder of an app run), STA_RUN (the folder of an STA run) and STA_ANALYSIS
(<STA run>/sta_algorithm_home/sta-analysis).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from utils import globals_utils as g


class Anchor(Enum):
    EXPERIMENT = "experiment folder"
    RUN = "run folder"
    STA_RUN = "STA run folder"
    STA_ANALYSIS = "STA analysis folder"
    PROJECT = "project folder"
    OUTPUT_ROOT = "outputs root (runtime/outputs or the test outputs folder)"
    SUITE2P_PLANE0 = "Suite2p plane0 folder"
    MOTION_CORRECTED = "motion_corrected folder"


class PathKind(Enum):
    # anchored on the experiment folder
    PIPELINE_RESULTS = auto()
    QC_DIR = auto()
    PB_CORRECTED_DIR = auto()
    MOTION_CORRECTED_DIR = auto()
    ASTRO_PB_MOVIE = auto()
    NEURON_PB_MOVIE = auto()
    BAD_SEGMENTS = auto()
    SUITE2P_PLANE0 = auto()
    STA_MOVIE_TIF_CELL_DIR = auto()  # fields: cell
    STA_MOVIE_TIF = auto()  # fields: cell, group, bin
    # anchored on the folder of an app run
    RUN_STA_ANALYSIS = auto()  # <run>/sta-analysis (runs of the Anatomy Crop and Behavior Split steps keep files there)
    ANATOMY_CROP_PARAMS = auto()
    BEHAVIOR_SEGMENTS = auto()
    SPIKE_CELL_DIR = auto()  # fields: cell
    # anchored on the folder of an STA run
    STA_ALGORITHM_HOME = auto()
    STA_ANALYSIS = auto()
    # anchored on the STA analysis folder
    STA_SUMMARY_CSV = auto()
    ANATOMY_SUMMARY = auto()
    GROUP_TEST = auto()
    GROUP_TEST_RECORDS = auto()
    CELL_DIR = auto()  # fields: cell
    ANATOMY_RESULT = auto()  # fields: cell
    CLUSTER_RESULT = auto()  # fields: cell, result_type
    MOTION_STA = auto()  # fields: cell
    # preprocessing: QC files (experiment folder), raw movie names, run-folder files, runner scripts (project folder)
    PB_QC_METRICS = auto()  # anchor: experiment; fields: pb_kind
    PB_QC_TRACE = auto()  # anchor: experiment; fields: pb_kind
    PB_QC_PLOT = auto()  # anchor: experiment; fields: pb_kind
    PREPROCESSING_STATE = auto()  # anchor: run
    PREPROCESSING_PARAMS_DIR = auto()  # anchor: run
    PREPROCESSING_PARAMS = auto()  # anchor: run; fields: run_id
    CLUSTER_JOB_ID = auto()  # anchor: run
    PREPROCESSING_SOURCE = auto()  # anchor: project
    PREPROCESSING_RUNNER = auto()  # anchor: project; fields: script
    # area 5: Suite2p outputs, motion-correction outputs, STA job files
    SUITE2P_ISCELL = auto()  # anchor: Suite2p plane0 folder
    SUITE2P_STAT = auto()  # anchor: Suite2p plane0 folder
    SUITE2P_F = auto()  # anchor: Suite2p plane0 folder
    SUITE2P_FNEU = auto()  # anchor: Suite2p plane0 folder
    SUITE2P_OPS = auto()  # anchor: Suite2p plane0 folder
    MC_ASTRO_MEAN_IMAGE = auto()  # anchor: motion_corrected folder
    MC_NEURON_MEAN_IMAGE = auto()  # anchor: motion_corrected folder
    MC_SHIFTS = auto()  # anchor: experiment
    BINNED_ASTRO_MOVIE = auto()  # anchor: experiment; fields: bin
    RUN_BEHAVIORAL_BINNED_MOVIE = auto()  # anchor: run; fields: group, bin
    RUN_BEHAVIORAL_BINNED_FRAMES = auto()  # anchor: run; fields: group, bin
    STA_USED_PARAMETERS_JSON = auto()  # anchor: STA analysis folder
    STA_USED_PARAMETERS_NPZ = auto()  # anchor: STA analysis folder
    MOTION_SERIES = auto()  # anchor: STA analysis folder
    MEAN_ASTRO_BINNED = auto()  # anchor: STA analysis folder
    STA_SKIPPED_BEHAVIORAL_GROUPS = auto()  # anchor: STA analysis folder
    NEURON_STA = auto()  # anchor: STA analysis folder; fields: cell, suffix
    SPIKE_PEAK_PARAMS = auto()  # anchor: run (spike run); fields: cell
    STA_MOVIE_NPY = auto()  # anchor: STA analysis folder; fields: cell, name, bin
    # experiment registry: files the validation looks for, confirmed-cell data folders (project folder)
    RAW_MOVIE = auto()  # anchor: experiment
    EXPERIMENT_XML = auto()  # anchor: experiment
    SCIENTIFIC_DATABASE_ROOT = auto()  # anchor: project
    SCIENTIFIC_DATABASE_EXPERIMENT = auto()  # anchor: project; fields: canonical_id
    # area 4: spike label files of a spike run (anchor: run; fields: cell) and the confirmed-cell folders (anchor: project)
    SPIKE_LABELS = auto()
    SPIKE_PEAKS = auto()
    SPIKE_MANUAL_REVIEW = auto()
    CONFIRMED_CELL_DIR = auto()  # fields: canonical_id, cell
    CONFIRMED_SPIKES_DIR = auto()  # fields: canonical_id, cell
    CONFIRMED_STA_DIR = auto()  # fields: canonical_id, cell
    CONFIRMED_CLUSTER_MASKS = auto()  # fields: canonical_id, cell, name

    # area 3 (runs / monitoring / adapters)
    RUN_STDOUT_LOG = auto()  # anchor: run
    RUN_STDERR_LOG = auto()  # anchor: run
    CLUSTER_STDOUT_LOG = auto()  # anchor: run
    CLUSTER_STDERR_LOG = auto()  # anchor: run
    RUN_METRICS_JSON = auto()  # anchor: run
    BEHAVIOR_SPEED_PNG = auto()  # anchor: run
    PREPROCESSING_STEP_LOG = auto()  # anchor: run; fields: step
    SPIKE_REVIEW_PNG = auto()  # anchor: run; fields: cell
    ANATOMY_CROP_PREVIEW_PNG = auto()  # anchor: run
    STA_JOB_PARAMS_EFFECTIVE = auto()  # anchor: run
    GROUP_TEST_STDOUT_LOG = auto()  # anchor: run
    GROUP_TEST_JOB_ID = auto()  # anchor: run
    CLUSTER_CODE = auto()  # anchor: project
    STA_JOB_SCRIPT = auto()  # anchor: project
    STA_ALGORITHM_SCRIPT = auto()  # anchor: project
    GROUP_TEST_JOB_SCRIPT = auto()  # anchor: project
    RUN_DIR = auto()  # anchor: outputs root; fields: canonical_id, cell, step, run_id
    RUN_DIR_ATTEMPT = auto()  # anchor: outputs root; fields: canonical_id, cell, step, run_id, attempt

@dataclass(frozen=True)
class _Spec:
    anchor: Anchor
    parts: tuple[str, ...]  # path under the anchor; the {fields} are filled by path_for


_E, _R, _S, _A = Anchor.EXPERIMENT, Anchor.RUN, Anchor.STA_RUN, Anchor.STA_ANALYSIS
_P0, _MC = Anchor.SUITE2P_PLANE0, Anchor.MOTION_CORRECTED
_CELL = g.CELL_DIR_TEMPLATE

_REGISTRY: dict[PathKind, _Spec] = {
    PathKind.PIPELINE_RESULTS: _Spec(_E, (g.PIPELINE_RESULTS_DIR,)),
    PathKind.QC_DIR: _Spec(_E, (g.PIPELINE_RESULTS_DIR, g.QC_DIR)),
    PathKind.PB_CORRECTED_DIR: _Spec(_E, (g.PIPELINE_RESULTS_DIR, g.PB_CORRECTED_DIR)),
    PathKind.MOTION_CORRECTED_DIR: _Spec(_E, (g.PIPELINE_RESULTS_DIR, g.MOTION_CORRECTED_DIR)),
    PathKind.ASTRO_PB_MOVIE: _Spec(_E, (g.PIPELINE_RESULTS_DIR, g.PB_CORRECTED_DIR, g.ASTRO_PB_MOVIE)),
    PathKind.NEURON_PB_MOVIE: _Spec(_E, (g.PIPELINE_RESULTS_DIR, g.PB_CORRECTED_DIR, g.NEURON_PB_MOVIE)),
    PathKind.BAD_SEGMENTS: _Spec(_E, (g.PIPELINE_RESULTS_DIR, g.QC_DIR, g.BAD_SEGMENTS)),
    PathKind.SUITE2P_PLANE0: _Spec(_E, (g.PIPELINE_RESULTS_DIR, g.FOR_SUITE2P_DIR, g.SUITE2P_DIR, g.PLANE0_DIR)),
    PathKind.STA_MOVIE_TIF_CELL_DIR: _Spec(_E, (g.STA_MOVIES_TIF_DIR, _CELL)),
    PathKind.STA_MOVIE_TIF: _Spec(_E, (g.STA_MOVIES_TIF_DIR, _CELL, g.STA_MOVIE_TIF_TEMPLATE)),
    PathKind.RUN_STA_ANALYSIS: _Spec(_R, (g.STA_ANALYSIS_DIR,)),
    PathKind.ANATOMY_CROP_PARAMS: _Spec(_R, (g.STA_ANALYSIS_DIR, g.ANATOMY_CROP_PARAMS)),
    PathKind.BEHAVIOR_SEGMENTS: _Spec(_R, (g.BEHAVIOR_SEGMENTS,)),
    PathKind.SPIKE_CELL_DIR: _Spec(_R, (_CELL,)),
    PathKind.STA_ALGORITHM_HOME: _Spec(_S, (g.STA_ALGORITHM_HOME_DIR,)),
    PathKind.STA_ANALYSIS: _Spec(_S, (g.STA_ALGORITHM_HOME_DIR, g.STA_ANALYSIS_DIR)),
    PathKind.STA_SUMMARY_CSV: _Spec(_A, (g.STA_SUMMARY_CSV,)),
    PathKind.ANATOMY_SUMMARY: _Spec(_A, (g.ANATOMY_SUMMARY,)),
    PathKind.GROUP_TEST: _Spec(_A, (g.GROUP_TEST,)),
    PathKind.GROUP_TEST_RECORDS: _Spec(_A, (g.GROUP_TEST_RECORDS,)),
    PathKind.CELL_DIR: _Spec(_A, (_CELL,)),
    PathKind.ANATOMY_RESULT: _Spec(_A, (_CELL, g.ANATOMY_RESULT)),
    PathKind.CLUSTER_RESULT: _Spec(_A, (_CELL, g.TYPED_RESULT_TEMPLATE)),
    PathKind.MOTION_STA: _Spec(_A, (_CELL, g.MOTION_STA)),
    PathKind.PB_QC_METRICS: _Spec(_E, (g.PIPELINE_RESULTS_DIR, g.QC_DIR, g.PB_QC_METRICS_TEMPLATE)),
    PathKind.PB_QC_TRACE: _Spec(_E, (g.PIPELINE_RESULTS_DIR, g.QC_DIR, g.PB_QC_TRACE_TEMPLATE)),
    PathKind.PB_QC_PLOT: _Spec(_E, (g.PIPELINE_RESULTS_DIR, g.QC_DIR, g.PB_QC_PLOT_TEMPLATE)),
    PathKind.PREPROCESSING_STATE: _Spec(_R, (g.PREPROCESSING_STATE_FILE,)),
    PathKind.PREPROCESSING_PARAMS_DIR: _Spec(_R, (g.PREPROCESSING_PARAMS_DIR,)),
    PathKind.PREPROCESSING_PARAMS: _Spec(_R, (g.PREPROCESSING_PARAMS_DIR, g.PREPROCESSING_PARAMS_TEMPLATE)),
    PathKind.CLUSTER_JOB_ID: _Spec(_R, (g.CLUSTER_JOB_ID_FILE,)),
    PathKind.PREPROCESSING_SOURCE: _Spec(Anchor.PROJECT, g.PREPROCESSING_SOURCE_DIR),
    PathKind.PREPROCESSING_RUNNER: _Spec(Anchor.PROJECT, (*g.PREPROCESSING_SOURCE_DIR, g.PREPROCESSING_RUNNER_DIR, "{script}")),
    PathKind.SUITE2P_ISCELL: _Spec(_P0, (g.SUITE2P_ISCELL_FILE,)),
    PathKind.SUITE2P_STAT: _Spec(_P0, (g.SUITE2P_STAT_FILE,)),
    PathKind.SUITE2P_F: _Spec(_P0, (g.SUITE2P_F_FILE,)),
    PathKind.SUITE2P_FNEU: _Spec(_P0, (g.SUITE2P_FNEU_FILE,)),
    PathKind.SUITE2P_OPS: _Spec(_P0, (g.SUITE2P_OPS_FILE,)),
    PathKind.MC_ASTRO_MEAN_IMAGE: _Spec(_MC, (g.ASTRO_MEAN_IMAGE_FILE,)),
    PathKind.MC_NEURON_MEAN_IMAGE: _Spec(_MC, (g.NEURON_MEAN_IMAGE_FILE,)),
    PathKind.MC_SHIFTS: _Spec(_E, (g.PIPELINE_RESULTS_DIR, g.MOTION_CORRECTED_DIR, g.MOTION_SHIFTS_DIR, g.MOTION_SHIFTS_FILE)),
    PathKind.BINNED_ASTRO_MOVIE: _Spec(_E, (g.PIPELINE_RESULTS_DIR, g.PB_CORRECTED_DIR, g.BINNED_ASTRO_MOVIE_TEMPLATE)),
    PathKind.RUN_BEHAVIORAL_BINNED_MOVIE: _Spec(_R, (g.STA_ANALYSIS_DIR, g.BEHAVIORAL_BINNED_MOVIE_TEMPLATE)),
    PathKind.RUN_BEHAVIORAL_BINNED_FRAMES: _Spec(_R, (g.STA_ANALYSIS_DIR, g.BEHAVIORAL_BINNED_FRAMES_TEMPLATE)),
    PathKind.STA_USED_PARAMETERS_JSON: _Spec(_A, (g.STA_USED_PARAMETERS_JSON,)),
    PathKind.STA_USED_PARAMETERS_NPZ: _Spec(_A, (g.STA_USED_PARAMETERS_NPZ,)),
    PathKind.MOTION_SERIES: _Spec(_A, (g.MOTION_SERIES_FILE,)),
    PathKind.MEAN_ASTRO_BINNED: _Spec(_A, (g.MEAN_ASTRO_BINNED_FILE,)),
    PathKind.STA_SKIPPED_BEHAVIORAL_GROUPS: _Spec(_A, (g.STA_SKIPPED_BEHAVIORAL_GROUPS,)),
    PathKind.NEURON_STA: _Spec(_A, (_CELL, g.NEURON_STA_TEMPLATE)),
    PathKind.SPIKE_PEAK_PARAMS: _Spec(_R, (_CELL, g.PEAK_DETECTION_PARAMS_FILE)),
    PathKind.STA_MOVIE_NPY: _Spec(_A, (_CELL, g.STA_MOVIE_NPY_TEMPLATE)),
    PathKind.RAW_MOVIE: _Spec(_E, (g.RAW_MOVIE,)),
    PathKind.EXPERIMENT_XML: _Spec(_E, (g.EXPERIMENT_XML,)),
    PathKind.SCIENTIFIC_DATABASE_ROOT: _Spec(Anchor.PROJECT, (g.RUNTIME_DIR, g.SCIENTIFIC_DATABASE_DIR)),
    PathKind.SCIENTIFIC_DATABASE_EXPERIMENT: _Spec(Anchor.PROJECT, (g.RUNTIME_DIR, g.SCIENTIFIC_DATABASE_DIR, "{canonical_id}")),
    PathKind.SPIKE_LABELS: _Spec(_R, (_CELL, g.SPIKE_LABELS)),
    PathKind.SPIKE_PEAKS: _Spec(_R, (_CELL, g.SPIKE_PEAKS)),
    PathKind.SPIKE_MANUAL_REVIEW: _Spec(_R, (_CELL, g.MANUAL_SPIKE_REVIEW)),
    PathKind.CONFIRMED_CELL_DIR: _Spec(Anchor.PROJECT, (g.RUNTIME_DIR, g.SCIENTIFIC_DATABASE_DIR, "{canonical_id}", _CELL)),
    PathKind.CONFIRMED_SPIKES_DIR: _Spec(Anchor.PROJECT, (g.RUNTIME_DIR, g.SCIENTIFIC_DATABASE_DIR, "{canonical_id}", _CELL, g.CONFIRMED_SPIKES_SUBDIR)),
    PathKind.CONFIRMED_STA_DIR: _Spec(Anchor.PROJECT, (g.RUNTIME_DIR, g.SCIENTIFIC_DATABASE_DIR, "{canonical_id}", _CELL, g.CONFIRMED_STA_SUBDIR)),
    PathKind.CONFIRMED_CLUSTER_MASKS: _Spec(Anchor.PROJECT, (g.RUNTIME_DIR, g.SCIENTIFIC_DATABASE_DIR, "{canonical_id}", _CELL, g.CONFIRMED_STA_SUBDIR, g.CONFIRMED_CLUSTER_MASKS_TEMPLATE)),
    PathKind.RUN_STDOUT_LOG: _Spec(_R, (g.RUN_STDOUT_LOG,)),
    PathKind.RUN_STDERR_LOG: _Spec(_R, (g.RUN_STDERR_LOG,)),
    PathKind.CLUSTER_STDOUT_LOG: _Spec(_R, (g.CLUSTER_STDOUT_LOG,)),
    PathKind.CLUSTER_STDERR_LOG: _Spec(_R, (g.CLUSTER_STDERR_LOG,)),
    PathKind.RUN_METRICS_JSON: _Spec(_R, (g.RUN_METRICS_JSON,)),
    PathKind.BEHAVIOR_SPEED_PNG: _Spec(_R, (g.BEHAVIOR_SPEED_PNG,)),
    PathKind.PREPROCESSING_STEP_LOG: _Spec(_R, (g.PREPROCESSING_STEP_LOG_TEMPLATE,)),
    PathKind.SPIKE_REVIEW_PNG: _Spec(_R, (_CELL, g.SPIKE_REVIEW_PNG)),
    PathKind.ANATOMY_CROP_PREVIEW_PNG: _Spec(_R, (g.ANATOMY_CROP_PREVIEW_PNG,)),
    PathKind.STA_JOB_PARAMS_EFFECTIVE: _Spec(_R, (g.STA_JOB_PARAMS_EFFECTIVE,)),
    PathKind.GROUP_TEST_STDOUT_LOG: _Spec(_R, (g.GROUP_TEST_STDOUT_LOG,)),
    PathKind.GROUP_TEST_JOB_ID: _Spec(_R, (g.GROUP_TEST_JOB_ID_FILE,)),
    PathKind.CLUSTER_CODE: _Spec(Anchor.PROJECT, (g.CLUSTER_CODE_DIR,)),
    PathKind.STA_JOB_SCRIPT: _Spec(Anchor.PROJECT, (g.CLUSTER_CODE_DIR, g.STA_JOB_SCRIPT)),
    PathKind.STA_ALGORITHM_SCRIPT: _Spec(Anchor.PROJECT, (g.CLUSTER_CODE_DIR, g.STA_ALGORITHM_SCRIPT)),
    PathKind.GROUP_TEST_JOB_SCRIPT: _Spec(Anchor.PROJECT, (g.CLUSTER_CODE_DIR, g.GROUP_TEST_JOB_SCRIPT)),
    PathKind.RUN_DIR: _Spec(Anchor.OUTPUT_ROOT, (g.RUN_DIR_TEMPLATE,)),
    PathKind.RUN_DIR_ATTEMPT: _Spec(Anchor.OUTPUT_ROOT, (g.RUN_DIR_ATTEMPT_TEMPLATE,)),
}


def anchor_of(kind: PathKind) -> Anchor:
    """Which folder a kind is relative to."""
    return _REGISTRY[kind].anchor


def path_for(kind: PathKind, anchor_path: str | Path, **fields) -> Path:
    """The path of `kind` under `anchor_path` (the folder anchor_of(kind) names). `fields` fill the placeholders of the name
    (cell=, group=, bin=, result_type=); a missing field is an error."""
    spec = _REGISTRY[kind]
    try:
        parts = [part.format(**fields) for part in spec.parts]
    except KeyError as exc:
        raise ValueError(f"{kind.name} needs the field {exc.args[0]!r}") from None
    return Path(anchor_path).joinpath(*parts)


def result_file_name(result_type: str) -> str:
    """File name of the cluster result of a result type ('all', 'simple', 'complex', 'anatomy', ...)."""
    return g.ALL_RESULT if result_type == g.ALL_GROUP else g.TYPED_RESULT_TEMPLATE.format(result_type=result_type)


def sta_dir_of_run(sta_run_dir: str | Path) -> Path:
    """<STA run folder> -> its sta-analysis folder."""
    return path_for(PathKind.STA_ANALYSIS, sta_run_dir)


def run_dir_of_sta_dir(sta_dir: str | Path) -> Path:
    """<STA run folder>/sta_algorithm_home/sta-analysis -> <STA run folder>."""
    return Path(sta_dir).parents[1]


def cell_dirs(parent: str | Path) -> dict[int, Path]:
    """{cell number: folder} of the cell_<N> folders inside an STA analysis folder or a spike run folder, ordered by number."""
    found = {}
    for path in Path(parent).glob(g.CELL_DIR_GLOB):
        suffix = path.name[len(g.CELL_DIR_PREFIX):]
        if path.is_dir() and suffix.isdigit():
            found[int(suffix)] = path
    return dict(sorted(found.items()))


@dataclass(frozen=True)
class STAPaths:
    home_dir: Path
    sta_dir: Path
    pb_dir: Path
    mc_dir: Path
    astro_movie_path: Path
    suite2p_path: Path
    raw_movie_path: Path
    bad_segments_path: Path


def resolve_sta_paths(home_dir: str | Path, sta_home_dir: str | Path | None = None) -> STAPaths:
    home_dir = Path(home_dir)
    sta_home = Path(sta_home_dir) if sta_home_dir is not None else home_dir
    if path_for(PathKind.STA_ALGORITHM_HOME, sta_home).exists():
        sta_home = path_for(PathKind.STA_ALGORITHM_HOME, sta_home)
    return STAPaths(
        home_dir=home_dir,
        sta_dir=sta_home / g.STA_ANALYSIS_DIR,
        pb_dir=path_for(PathKind.PB_CORRECTED_DIR, home_dir),
        mc_dir=path_for(PathKind.MOTION_CORRECTED_DIR, home_dir),
        astro_movie_path=path_for(PathKind.ASTRO_PB_MOVIE, home_dir),
        suite2p_path=path_for(PathKind.SUITE2P_PLANE0, home_dir),
        raw_movie_path=home_dir / g.RAW_MOVIE,
        bad_segments_path=path_for(PathKind.BAD_SEGMENTS, home_dir),
    )
