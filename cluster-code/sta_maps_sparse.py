from __future__ import annotations

import argparse
import csv
import gc
import multiprocessing as mp
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.ndimage import find_objects, generate_binary_structure, label
from scipy.sparse import csr_matrix, vstack

from utils import globals_utils as g
from utils.cluster_map_summary import save_cluster_map_summary
from utils.clusters_utils import _merge_similar_clusters
from utils.motion_diagnostics import (
    build_motion_series,
    cuts_from_bad_segments,
    motion_sta_stats,
    registration_quality,
)
from utils.border import INTERPOLATION_PX, analysed_shifts, border_margins_px, read_bad_segments, valid_mask
from utils.global_trends import compute_global_trends
from utils.hypotheses import frame_windows, run_settings, seconds_to_frames, time_bin_frames
from utils.movie_utils import read_astro_movie, read_frame_rate, spatial_bin_movie
from utils.sta_stats_utils import (
    NULL_STANDARDIZATION_MODES,
    f0_for_valid_centers,
    filter_valid_peaks,
    isolated_spikes,
    max_mass_by_sign,
    min_footprint_in_analysis_px,
    null_sd_map,
    passes_extent,
    per_sign_p_values,
    per_sign_thresholds,
    silence_pool,
    smooth_spatial,
    standardize,
)
from utils.paths_utils import PathKind, cell_dirs, path_for, resolve_sta_paths, result_file_name
from utils.file_utils import read_json, write_json
from utils.sta_trace_utils import crop_bounds
from utils.qc_utils import read_trimmed_frames
from utils.tif_export import sta_note, write_imagej_stack
from utils.suite2p_utils import load_suite2p_context, neuron_centroid_binned

HOME_DIR = Path(r"Z:\Adam-Lab-Shared\Data\Miki_Meerson\Imaging\HYP-JEDI\Hyp3\L\02-02-2026-awake\FOV1\z-18-x2")
SOURCE_HOME_DIR: Path | None = None
# the input runs and the run name, given by the app in the job config (no folder layout is assumed around home_dir)
SPIKE_RUN_DIR: Path | None = None
BEHAVIOR_SEGMENTS_FILE: Path | None = None
ANATOMY_CROP_PARAMS_FILE: Path | None = None
RUN_NAME = ""
LINUX_SHARED_ROOT = Path("/ems/elsc-labs/adam-y/Adam-Lab-Shared")
WINDOWS_SHARED_ROOT = Path(r"Z:\Adam-Lab-Shared")
SCRIPT_DIR = Path(__file__).resolve().parent
JOB_PARAMS_FILE = "sta_maps_job_params.json"

HYPOTHESIS: dict | None = None  # the hypothesis settings (seconds) of this run, from the job config (see utils/hypotheses.py)

SPATIAL_BIN = 8
ANATOMY_SPATIAL_BIN = 1
N_PERMUTATIONS = 200
MIN_SPIKES_FOR_CLUSTER = 3
N_WORKERS = max(1, min(4, (mp.cpu_count() or 2) - 1))

VOXEL_ALPHA = 0.02
CLUSTER_ALPHA = 0.1
CLUSTER_CONNECTIVITY = 2
MIN_CLUSTER_SIZE = 3
MIN_CLUSTER_MASS = 0.0
MERGE_SIMILAR_CLUSTERS = False
MERGE_MAX_DISTANCE_PX = 3.0
MERGE_MIN_TRACE_CORR = 0.9
RNG_SEED = 0

# Statistic / cluster-definition options. Defaults reproduce the original (legacy) behaviour; the app's
# defaults.sta in config.example.yml turns the recommended settings on. See docs/statistics.md.
NULL_STANDARDIZATION = "none"  # "none" | "per_pixel_z": divide effect maps by the per-pixel permutation-null sd
NULL_SD_FLOOR_QUANTILE = 0.05
PER_SIGN_NULL = False  # opt-in: per-sign voxel thresholds + per-sign max-mass nulls, p = min(1, 2 * p_sign) (docs/statistics.md)
SMOOTH_SIGMA_FULLRES_PX = 0.0  # spatial gaussian (full-resolution px; converted by spatial bin, off below 0.25 analysis px)
MIN_CLUSTER_FOOTPRINT_FULLRES_PX = 0.0  # min spatial footprint, in full-resolution px^2 (converted by spatial bin)
MIN_CLUSTER_FRAMES = 0  # min number of frames a cluster must span
EXCLUDE_SPIKES_NEAR_CUTS = False  # drop spikes whose analysis window contains a splice where a bad segment was removed (the random-frame chance copies are drawn from the same splice-free frames)
MOTION_DIAGNOSTICS = True  # diagnostic only: spike-triggered motion / registration-quality test per cell
MOTION_FLAG_ALPHA = 0.05
MASK_BORDER = True  # leave the border that the registration shifts expose out of the test (utils/border.py)
BORDER_EXTRA_PX = INTERPOLATION_PX  # pixels added to the largest shift toward each side
MOTION_PERMUTATIONS = 2000  # random-frame copies of the motion test (the series are 1-D, so many copies are cheap; the maps use N_PERMUTATIONS)
MOTION_REG_QUALITY_BIN = 0  # spatial bin of the movie used for the registration-quality series; 0 = auto

SAVE_BINNED_STA_MOVIES = True
SAVE_FIJI_TIFS = False  # save each binned STA movie as an ImageJ .tif in <experiment>/sta_movies_tif/<hypothesis>/cell_<N>/ INSTEAD of the .npy in the run folder (needs SAVE_BINNED_STA_MOVIES)
OVERWRITE = True
CELL_IDS: list[int] | str = g.ALL_GROUP
RESULT_TYPES = (g.ALL_GROUP,)
BEHAVIORAL_GROUPS = ("active", "inactive")
VALID_RESULT_TYPES = {g.ALL_GROUP, g.ANATOMY_GROUP, "behavioral"}
SUMMARY_CSV: str | None = None

TIMING_PHASES = ("spikes+pool", "observed STA", "surrogate events", "surrogate STA", "baseline map", "clusters", "saving")  # seconds per cell, one log line

_MOVIE: np.ndarray | None = None
_BASELINE_MOVIE: np.ndarray | None = None  # the run's baseline-averaged movie (see baseline_average_movie); None: built per cell for the crop
_MOVIE_SHAPE: tuple[int, int, int] | None = None
_VALID_CENTERS: np.ndarray | None = None
_F0_BY_POS: np.ndarray | None = None
_SUITE2P_CTX: dict | None = None
_ANATOMY_RADIUS: int | None = None
RUN_CONFIG: dict | None = None  # settings dict of this run, handed to pool workers (see init_worker)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse STA map analysis from automated spike labels.")
    parser.add_argument(
        "config",
        nargs="?",
        type=str,
        default=None,
        help=f"Path to {JOB_PARAMS_FILE}. Defaults to cluster-code/{JOB_PARAMS_FILE} when present.",
    )
    return parser.parse_args()


def load_config(path_arg: str | None) -> dict:
    config_path = None if path_arg in (None, "", "None") else Path(path_arg)
    if config_path is None:
        default_path = SCRIPT_DIR / JOB_PARAMS_FILE
        config_path = default_path if default_path.exists() else None
    if config_path is None:
        return {}
    config = read_json(config_path, "The job settings file")
    print(f"Loaded settings from {config_path}")
    return config


def apply_config(config: dict) -> None:
    global HOME_DIR, SOURCE_HOME_DIR, SPIKE_RUN_DIR, BEHAVIOR_SEGMENTS_FILE, ANATOMY_CROP_PARAMS_FILE, RUN_NAME, SPATIAL_BIN, ANATOMY_SPATIAL_BIN, N_PERMUTATIONS, N_WORKERS, VOXEL_ALPHA, CLUSTER_ALPHA
    global CLUSTER_CONNECTIVITY, MIN_CLUSTER_SIZE, MIN_CLUSTER_MASS, MERGE_SIMILAR_CLUSTERS
    global MERGE_MAX_DISTANCE_PX, MERGE_MIN_TRACE_CORR, RNG_SEED, SAVE_BINNED_STA_MOVIES, SAVE_FIJI_TIFS
    global OVERWRITE, CELL_IDS, RESULT_TYPES, SUMMARY_CSV, HYPOTHESIS
    global MIN_SPIKES_FOR_CLUSTER, NULL_STANDARDIZATION, NULL_SD_FLOOR_QUANTILE, SMOOTH_SIGMA_FULLRES_PX, PER_SIGN_NULL
    global MIN_CLUSTER_FOOTPRINT_FULLRES_PX, MIN_CLUSTER_FRAMES, MOTION_DIAGNOSTICS, MOTION_FLAG_ALPHA, MOTION_PERMUTATIONS, MOTION_REG_QUALITY_BIN, MASK_BORDER, BORDER_EXTRA_PX
    global EXCLUDE_SPIKES_NEAR_CUTS

    if "home_dir" in config:
        HOME_DIR = Path(config["home_dir"])
    SOURCE_HOME_DIR = Path(config["source_home_dir"]) if config.get("source_home_dir") else None
    SPIKE_RUN_DIR = Path(config["spike_run_dir"]) if config.get("spike_run_dir") else None
    BEHAVIOR_SEGMENTS_FILE = Path(config["behavior_segments_path"]) if config.get("behavior_segments_path") else None
    ANATOMY_CROP_PARAMS_FILE = Path(config["anatomy_crop_params_path"]) if config.get("anatomy_crop_params_path") else None
    RUN_NAME = str(config.get("run_name", ""))
    CELL_IDS = config.get("cell_ids", CELL_IDS)
    SPATIAL_BIN = int(config.get("spatial_bin", SPATIAL_BIN))
    ANATOMY_SPATIAL_BIN = int(config.get("anatomy_spatial_bin", ANATOMY_SPATIAL_BIN))
    N_PERMUTATIONS = int(config.get("n_permutations", N_PERMUTATIONS))
    N_WORKERS = int(config.get("n_workers", config.get("workers", N_WORKERS)))
    VOXEL_ALPHA = float(config.get("voxel_alpha", VOXEL_ALPHA))
    CLUSTER_ALPHA = float(config.get("cluster_alpha", CLUSTER_ALPHA))
    CLUSTER_CONNECTIVITY = int(config.get("cluster_connectivity", CLUSTER_CONNECTIVITY))
    MIN_CLUSTER_SIZE = int(config.get("min_cluster_size", MIN_CLUSTER_SIZE))
    MIN_CLUSTER_MASS = float(config.get("min_cluster_mass", MIN_CLUSTER_MASS))
    MERGE_SIMILAR_CLUSTERS = bool(config.get("merge_similar_clusters", MERGE_SIMILAR_CLUSTERS))
    MERGE_MAX_DISTANCE_PX = float(config.get("merge_max_distance_px", MERGE_MAX_DISTANCE_PX))
    MERGE_MIN_TRACE_CORR = float(config.get("merge_min_trace_corr", MERGE_MIN_TRACE_CORR))
    RNG_SEED = int(config.get("rng_seed", RNG_SEED))
    MIN_SPIKES_FOR_CLUSTER = int(config.get("min_spikes_for_cluster", MIN_SPIKES_FOR_CLUSTER))
    NULL_STANDARDIZATION = str(config.get("null_standardization", NULL_STANDARDIZATION))
    if NULL_STANDARDIZATION not in NULL_STANDARDIZATION_MODES:
        raise ValueError(f"null_standardization must be one of {NULL_STANDARDIZATION_MODES}, got {NULL_STANDARDIZATION!r}")
    NULL_SD_FLOOR_QUANTILE = float(config.get("null_sd_floor_quantile", NULL_SD_FLOOR_QUANTILE))
    PER_SIGN_NULL = bool(config.get("per_sign_null", PER_SIGN_NULL))
    SMOOTH_SIGMA_FULLRES_PX = float(config.get("smooth_sigma_fullres_px", SMOOTH_SIGMA_FULLRES_PX))
    MIN_CLUSTER_FOOTPRINT_FULLRES_PX = float(config.get("min_cluster_footprint_fullres_px", MIN_CLUSTER_FOOTPRINT_FULLRES_PX))
    MIN_CLUSTER_FRAMES = int(config.get("min_cluster_frames", MIN_CLUSTER_FRAMES))
    EXCLUDE_SPIKES_NEAR_CUTS = bool(config.get("exclude_spikes_near_cuts", EXCLUDE_SPIKES_NEAR_CUTS))
    MOTION_DIAGNOSTICS = bool(config.get("motion_diagnostics", MOTION_DIAGNOSTICS))
    MOTION_FLAG_ALPHA = float(config.get("motion_flag_alpha", MOTION_FLAG_ALPHA))
    MOTION_PERMUTATIONS = int(config.get("motion_permutations", MOTION_PERMUTATIONS))
    MASK_BORDER = bool(config.get("mask_border", MASK_BORDER))
    BORDER_EXTRA_PX = int(config.get("border_extra_px", BORDER_EXTRA_PX))
    MOTION_REG_QUALITY_BIN = int(config.get("motion_reg_quality_bin", MOTION_REG_QUALITY_BIN))
    SAVE_BINNED_STA_MOVIES = bool(config.get("save_binned_sta_movies", SAVE_BINNED_STA_MOVIES))
    SAVE_FIJI_TIFS = bool(config.get("save_fiji_tifs", SAVE_FIJI_TIFS))
    OVERWRITE = bool(config.get("overwrite", OVERWRITE))
    RESULT_TYPES = config.get("result_types", RESULT_TYPES)
    RESULT_TYPES = (RESULT_TYPES,) if isinstance(RESULT_TYPES, str) else tuple(RESULT_TYPES)
    unknown = sorted(set(RESULT_TYPES) - VALID_RESULT_TYPES)
    if unknown:
        raise ValueError(f"Unknown result type(s): {unknown}")
    SUMMARY_CSV = config.get("summary_csv", SUMMARY_CSV)
    HYPOTHESIS = run_settings(config)


def _resolve_shared_path(path: Path) -> Path:
    """`path` itself when it exists, else its Windows (Z:) / Linux (/ems/...) counterpart when that exists, else `path`."""
    if path.exists():
        return path
    raw = str(path)
    windows_prefix = r"Z:\Adam-Lab-Shared"
    linux_prefix = LINUX_SHARED_ROOT.as_posix()
    if raw.startswith(windows_prefix):
        candidate = LINUX_SHARED_ROOT / raw[len(windows_prefix):].lstrip(r"\/").replace("\\", "/")
        if candidate.exists():
            return candidate
    if raw.replace("\\", "/").startswith(linux_prefix):
        candidate = WINDOWS_SHARED_ROOT / raw.replace("\\", "/")[len(linux_prefix):].lstrip("/")
        if candidate.exists():
            return candidate
    return path


def home_dir() -> Path:
    return _resolve_shared_path(HOME_DIR)


def source_home_dir() -> Path:
    return home_dir() if SOURCE_HOME_DIR is None else _resolve_shared_path(SOURCE_HOME_DIR)


def spike_analysis_dir() -> Path:
    """The spike run folder (cell_<N>/ label files) the STA reads."""
    if SPIKE_RUN_DIR is None:
        raise ValueError("The STA job config has no spike_run_dir (the app writes it from the experiment's completed spike run).")
    return _resolve_shared_path(SPIKE_RUN_DIR)


def behavior_segments_path() -> Path:
    if BEHAVIOR_SEGMENTS_FILE is None:
        raise ValueError("The STA job config has no behavior_segments_path (the behavioral result type needs the Behavior Split run).")
    return _resolve_shared_path(BEHAVIOR_SEGMENTS_FILE)


def binned_movie_path(paths, spatial_bin: int | None = None) -> Path:
    bin_factor = SPATIAL_BIN if spatial_bin is None else int(spatial_bin)
    return path_for(PathKind.BINNED_ASTRO_MOVIE, paths.home_dir, bin=bin_factor)


def movie_input_path(paths, spatial_bin: int | None = None) -> Path:
    bin_factor = SPATIAL_BIN if spatial_bin is None else int(spatial_bin)
    return paths.astro_movie_path if bin_factor <= 1 else binned_movie_path(paths, bin_factor)


def behavioral_binned_movie_path(paths, group: str, spatial_bin: int | None = None) -> Path:
    bin_factor = SPATIAL_BIN if spatial_bin is None else int(spatial_bin)
    return path_for(PathKind.RUN_BEHAVIORAL_BINNED_MOVIE, paths.home_dir, group=group, bin=bin_factor)


def behavioral_binned_frames_path(paths, group: str, spatial_bin: int | None = None) -> Path:
    bin_factor = SPATIAL_BIN if spatial_bin is None else int(spatial_bin)
    return path_for(PathKind.RUN_BEHAVIORAL_BINNED_FRAMES, paths.home_dir, group=group, bin=bin_factor)


def anatomy_crop_params_path() -> Path:
    if ANATOMY_CROP_PARAMS_FILE is None:
        raise ValueError("The STA job config has no anatomy_crop_params_path (the anatomy result type needs the Anatomy Crop run).")
    return _resolve_shared_path(ANATOMY_CROP_PARAMS_FILE)


def load_anatomy_radius(target_spatial_bin: int) -> dict:
    path = anatomy_crop_params_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the app Anatomy Crop step before requesting result_type='anatomy'."
        )
    with np.load(path, allow_pickle=True) as d:
        source_radius = int(np.asarray(d["anatomy_radius_binned"]).item())
        source_spatial_bin = int(np.asarray(d["spatial_bin"]).item())
    fullres_radius = int(source_radius) * max(1, int(source_spatial_bin))
    target_radius = int(np.ceil(fullres_radius / max(1, int(target_spatial_bin))))
    return {
        "anatomy_spatial_bin": int(target_spatial_bin),
        "anatomy_radius_binned": target_radius,
        "anatomy_source_radius_binned": source_radius,
        "anatomy_source_spatial_bin": source_spatial_bin,
        "anatomy_fullres_radius_px": fullres_radius,
    }


def ensure_binned_movie(paths, spatial_bin: int | None = None) -> np.memmap:
    bin_factor = SPATIAL_BIN if spatial_bin is None else int(spatial_bin)
    if bin_factor <= 1:
        print(f"Using full-resolution astro movie without writing a bin1 cache: {paths.astro_movie_path}")
        return read_astro_movie(str(paths.astro_movie_path), as_memmap=True)
    cache_path = binned_movie_path(paths, bin_factor)
    stale = cache_path.exists() and cache_path.stat().st_mtime < Path(paths.astro_movie_path).stat().st_mtime
    if stale:
        print(f"Rebuilding the binned astro movie cache: it is older than the movie it was made from ({cache_path.name}); the movie was processed again.")
    if cache_path.exists() and not stale:
        movie = np.load(cache_path, mmap_mode="r")
        frame_means = np.asarray(movie.mean(axis=(1, 2)), dtype=np.float32)
        zero_frames = int(np.sum(frame_means == 0))
        if zero_frames == 0:
            print(f"Using binned astro movie: {cache_path}")
            return movie
        print(f"Rebuilding invalid binned astro movie cache: {zero_frames}/{movie.shape[0]} frames are zero.")
        del movie

    print(f"Building binned astro movie once: {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    astro = read_astro_movie(str(paths.astro_movie_path), as_memmap=True)
    sample = spatial_bin_movie(np.asarray(astro[:1], dtype=np.float32), bin_factor)
    tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp.npy")
    out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(astro.shape[0], *sample.shape[1:]))
    for start in range(0, astro.shape[0], 512):
        stop = min(start + 512, astro.shape[0])
        out[start:stop] = spatial_bin_movie(np.asarray(astro[start:stop], dtype=np.float32), bin_factor)
        print(f"  cached frames {start}-{stop}")
    out.flush()
    del out
    tmp_path.replace(cache_path)
    return np.load(cache_path, mmap_mode="r")


def load_behavioral_frame_groups() -> dict[str, np.ndarray]:
    path = behavior_segments_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Create it from the behavior notebook before running result_type='behavioral'."
        )
    with np.load(path, allow_pickle=True) as d:
        frame_groups = {
            "active": np.asarray(d["active_frames"], dtype=np.int32),
            "inactive": np.asarray(d["inactive_frames"], dtype=np.int32),
        }
    for group, frames in frame_groups.items():
        if frames.size > 1 and np.any(np.diff(frames) <= 0):
            raise ValueError(f"{path}: {group}_frames must be strictly increasing.")
    return frame_groups


def ensure_behavioral_binned_movie(paths, group: str, source_frames: np.ndarray, spatial_bin: int | None = None) -> np.memmap:
    bin_factor = SPATIAL_BIN if spatial_bin is None else int(spatial_bin)
    cache_path = behavioral_binned_movie_path(paths, group, bin_factor)
    frames_path = behavioral_binned_frames_path(paths, group, bin_factor)
    source_frames = np.asarray(source_frames, dtype=np.int32)
    if cache_path.exists():
        movie = np.load(cache_path, mmap_mode="r")
        saved_frames = np.load(frames_path) if frames_path.exists() else None
        frames_match = saved_frames is not None and np.array_equal(np.asarray(saved_frames, dtype=np.int32), source_frames)
        if movie.shape[0] == source_frames.size and frames_match:
            print(f"Using behavioral {group} binned astro movie: {cache_path}")
            return movie
        print(
            f"Rebuilding behavioral {group} movie cache: "
            f"{cache_path} has {movie.shape[0]} frames, expected {source_frames.size}."
        )
        del movie

    full_movie = ensure_binned_movie(paths, bin_factor)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp.npy")
    out = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.float32,
        shape=(source_frames.size, *full_movie.shape[1:]),
    )
    for start in range(0, source_frames.size, 512):
        stop = min(start + 512, source_frames.size)
        out[start:stop] = np.asarray(full_movie[source_frames[start:stop]], dtype=np.float32)
        print(f"  cached behavioral {group} frames {start}-{stop}")
    out.flush()
    del out
    np.save(frames_path, source_frames)
    tmp_path.replace(cache_path)
    return np.load(cache_path, mmap_mode="r")


def relative_indices(window: tuple[int, int], pre: int, post: int) -> np.ndarray:
    rel = np.arange(-pre, post + 1)
    return np.flatnonzero((rel >= window[0]) & (rel <= window[1]))


def compute_effect_movie(
    sta_movie: np.ndarray,
    pre_frames: int,
    post_frames: int,
    pre_window: tuple[int, int] = (-20, -1),
    post_window: tuple[int, int] = (0, 20),
    analysis_window: tuple[int, int] | None = None,
    baseline_window: tuple[int, int] | None = None,
    time_bin: int = 1,
) -> dict:
    """The effect movie: the STA frames of the analysis window minus the baseline map. With
    time_bin > 1 the frames are first averaged in consecutive bins of that many frames, starting at the window start (a last part shorter
    than a bin is dropped)."""
    if analysis_window is None:
        analysis_window = (pre_window[0], post_window[1])
    if baseline_window is None:
        baseline_window = pre_window

    pre_idx = relative_indices(pre_window, pre_frames, post_frames)
    post_idx = relative_indices(post_window, pre_frames, post_frames)
    analysis_idx = relative_indices(analysis_window, pre_frames, post_frames)
    baseline_idx = relative_indices(baseline_window, pre_frames, post_frames)
    if baseline_idx.size == 0:
        raise ValueError("baseline_window selects no frames.")
    rel = np.arange(-pre_frames, post_frames + 1)
    baseline_map = sta_movie[baseline_idx].mean(axis=0)
    frames = sta_movie[analysis_idx]
    if time_bin > 1:
        n_bins = frames.shape[0] // time_bin
        frames = frames[:n_bins * time_bin].reshape(n_bins, time_bin, *frames.shape[1:]).mean(axis=1)
    effect_movie = frames - baseline_map[None]
    post_pre_map = sta_movie[post_idx].mean(axis=0) - baseline_map
    return {
        "effect_movie": effect_movie.astype(np.float32),
        "post_pre_map": post_pre_map.astype(np.float32),
        "baseline_map": baseline_map.astype(np.float32),
        "analysis_relative_frames": rel[analysis_idx],
        "pre_idx": pre_idx,
        "post_idx": post_idx,
        "analysis_idx": analysis_idx,
        "baseline_idx": baseline_idx,
    }


_EXPORT_FPS: float | None = None


def export_fiji_sta(cell_id: int, group: str, observed_sta: np.ndarray, pre: int, n_spikes: int) -> bool:
    """The binned STA movie as an ImageJ .tif in the experiment folder (sta_movies_tif/<hypothesis>/cell_<N>/sta_<group>_bin<B>.tif).

    Returns whether it was written. Never fails the STA: an export problem is only reported (the caller then keeps the .npy)."""
    global _EXPORT_FPS
    try:
        source = source_home_dir()
        if _EXPORT_FPS is None:
            _EXPORT_FPS = read_frame_rate(source)
        write_imagej_stack(
            path_for(PathKind.STA_MOVIE_TIF, source, hypothesis=HYPOTHESIS["hypothesis"], cell=cell_id, group=group, bin=SPATIAL_BIN),
            observed_sta,
            _EXPORT_FPS,
            pre,
            sta_note(cell_id, n_spikes, f"{group} spikes, spatial bin {SPATIAL_BIN}", run=RUN_NAME),
        )
        return True
    except OSError as exc:  # the share could not be written: the .npy is kept; a frame rate that cannot be read is not caught
        print(f"WARNING: Fiji tif for cell {cell_id} {group} was not written: {exc}")
        return False


def valid_centers(n_frames: int, pre: int, post: int, cuts=None) -> np.ndarray:
    """Spike frames whose window [c - pre, c + post] fits in the movie and, when `cuts` is given, contains no splice.

    A cut k means frames k-1 and k are not adjacent in time (a bad segment was removed between them). The splice lies in the
    window when c - pre <= k - 1 and k <= c + post. The same centres are where the random-frame chance copies are drawn, so the null is matched.
    """
    centers = np.arange(pre + 1, n_frames - post, dtype=np.int32)
    k = np.asarray(sorted({int(c) for c in cuts or []}), dtype=np.int64)
    if k.size and centers.size:
        first = np.searchsorted(k, centers.astype(np.int64) - pre + 1, side="left")  # first cut at or after c - pre + 1
        hit = (first < k.size) & (k[np.minimum(first, k.size - 1)] <= centers + post)
        centers = centers[~hit]
    return centers


def load_cut_frames(paths, n_frames: int) -> list[int]:
    """Positions (analysed frames) where removed bad segments left a splice.

    Bad segments are removed outside the app; qc/bad_segments.json ([start, end) frame ranges in the numbering before the removal) records where. Position in the cleaned movie = start - frames removed before it - photobleaching `trimmed`."""
    experiment_dir = Path(paths.mc_dir).parents[1]  # <experiment>/pipeline_results/motion_corrected
    cuts = cuts_from_bad_segments(read_bad_segments(experiment_dir), read_trimmed_frames(experiment_dir))
    return [c for c in cuts if 0 < c < int(n_frames)]


def run_cut_frames(paths, n_frames: int) -> list[int]:
    if not EXCLUDE_SPIKES_NEAR_CUTS:
        return []
    try:
        cuts = load_cut_frames(paths, n_frames)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"exclude_spikes_near_cuts is on but the cut positions could not be read: {exc}") from exc
    print(f"Bad-segment splices at analysed frames {cuts or 'none'}: spikes whose window [-pre, +post] contains one are excluded")
    return cuts


def crop_f0_for_valid_centers(crop: tuple[int, int, int, int], centers: np.ndarray, baseline_window: tuple[int, int]) -> np.ndarray:
    if _MOVIE is None:
        raise RuntimeError("Worker was not initialized.")
    y0, y1, x0, x1 = crop
    return f0_for_valid_centers(_MOVIE[:, y0:y1, x0:x1], centers, baseline_window)


def init_worker(
    movie_path: str,
    centers: np.ndarray,
    f0_by_pos: np.ndarray,
    suite2p_path: str,
    anatomy_radius: int | None,
    spatial_bin: int,
    config: dict | None = None,
    baseline_path: str | None = None,
) -> None:
    # Workers started with the "spawn" method (Windows/macOS) re-import this module and would otherwise run with
    # module defaults; apply the run config explicitly so every start method behaves the same.
    if config is not None:
        apply_config(config)
    global _MOVIE, _BASELINE_MOVIE, _MOVIE_SHAPE, _VALID_CENTERS, _F0_BY_POS, _SUITE2P_CTX, _ANATOMY_RADIUS, SPATIAL_BIN
    movie_path_obj = Path(movie_path)
    movie = np.load(movie_path_obj, mmap_mode="r") if movie_path_obj.suffix == ".npy" else read_astro_movie(str(movie_path_obj), as_memmap=True)
    _MOVIE = movie
    _BASELINE_MOVIE = np.load(baseline_path, mmap_mode="r") if baseline_path else None
    _MOVIE_SHAPE = tuple(int(x) for x in movie.shape)
    _VALID_CENTERS = np.asarray(centers, dtype=np.int32)
    _F0_BY_POS = np.asarray(f0_by_pos, dtype=np.float32)
    _ANATOMY_RADIUS = anatomy_radius
    _SUITE2P_CTX = load_suite2p_context(suite2p_path) if anatomy_radius is not None else None
    SPATIAL_BIN = int(spatial_bin)


def list_cells() -> list[int]:
    found = []
    for cell, path in cell_dirs(spike_analysis_dir()).items():
        if (path / g.SPIKE_LABELS).exists():
            found.append(cell)
    if CELL_IDS == g.ALL_GROUP:
        return found
    wanted = {int(x) for x in CELL_IDS}
    return [cell_id for cell_id in found if cell_id in wanted]


def load_kept_peaks(cell_id: int) -> np.ndarray:
    """The kept spikes of a cell, in the numbering of the analysed movie: every spike of its label file except the ones the user labelled
    "exclude". Excluded spikes must never enter any STA (all / anatomy / behavioral). For g.POOLED_CELL_ID: the kept spikes of all the cells of the run
    together (a frame with spikes of several cells is one event)."""
    if cell_id == g.POOLED_CELL_ID:
        return np.unique(np.concatenate([load_kept_peaks(cell) for cell in list_cells()]))
    with np.load(path_for(PathKind.SPIKE_CELL_DIR, spike_analysis_dir(), cell=cell_id) / g.SPIKE_LABELS, allow_pickle=True) as d:
        peaks = np.asarray(d["peaks_all"], dtype=np.int32)
        labels = np.char.lower(np.asarray(d["labels"]).astype("U8"))
    return peaks[labels != "exclude"]


def load_spikes(
    cell_id: int, n_frames: int, pre: int, post: int, centers: np.ndarray | None = None, silence: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, int]:
    """(spikes used, number of valid spikes before the silence rule) of a cell.

    `centers` (the run's valid centres, splice-free when exclude_spikes_near_cuts is on) also drops spikes near splices. `silence` = (frames
    before, frames after): a spike is used only if the cell has no other kept spike (also outside the valid centres) that close."""
    peaks = load_kept_peaks(cell_id)
    valid = filter_valid_peaks(peaks, n_frames, pre, post).astype(np.int32)
    if centers is not None:
        valid = valid[np.isin(valid, centers)]
    return isolated_spikes(valid, peaks, *silence), int(valid.size)


def load_behavioral_spikes(
    cell_id: int,
    source_frames: np.ndarray,
    n_frames: int,
    pre: int,
    post: int,
    silence: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, int]:
    """(spikes used in the behavioral movie, number of valid ones before the silence rule). The silence rule is applied to the original
    frame numbers, then the spikes are remapped into the behavioral movie."""
    source_frames = np.asarray(source_frames, dtype=np.int32)
    peaks = load_kept_peaks(cell_id)

    def remap(p: np.ndarray) -> np.ndarray:
        frame_pos = np.searchsorted(source_frames, p)
        in_bounds = frame_pos < source_frames.size
        matched = np.zeros(p.size, dtype=bool)
        matched[in_bounds] = source_frames[frame_pos[in_bounds]] == p[in_bounds]
        return filter_valid_peaks(frame_pos[matched].astype(np.int32), n_frames, pre, post).astype(np.int32)

    return remap(isolated_spikes(peaks, peaks, *silence)), int(remap(peaks).size)


def cell_surrogate_pool(cell_id: int, frames: np.ndarray, silence: tuple[int, int]) -> np.ndarray:
    """Indices into `frames` (the original frame numbers of the run's valid centres) where a chance copy of a spike of this cell may be drawn:
    no kept spike of the cell within the silence window. Built once per cell."""
    peaks = load_kept_peaks(cell_id)
    return silence_pool(frames, peaks, *silence)


def event_matrix(peaks: np.ndarray, positions: np.ndarray | None = None, f0_by_pos: np.ndarray | None = None) -> csr_matrix:
    """Row-normalised event weights over the valid centres. positions=None: the observed spike train (1 row). Otherwise `positions`
    (n_surrogates, n_spikes) are indices into the valid centres: one surrogate spike train per row."""
    if _VALID_CENTERS is None or _F0_BY_POS is None:
        raise RuntimeError("Worker was not initialized.")
    f0 = _F0_BY_POS if f0_by_pos is None else np.asarray(f0_by_pos, dtype=np.float32)
    peaks = np.asarray(peaks, dtype=np.int32)
    peak_pos = np.searchsorted(_VALID_CENTERS, peaks)
    if peaks.size == 0:
        n_rows = 1 if positions is None else len(positions)
        return csr_matrix((n_rows, _VALID_CENTERS.size), dtype=np.float32)

    if np.any(_VALID_CENTERS[peak_pos] != peaks):
        raise ValueError("All peaks must be valid centers.")

    if positions is None:
        rows = np.zeros(peaks.size, dtype=np.int32)
        cols = peak_pos.astype(np.int32)
        data = (1.0 / (peaks.size * f0[cols])).astype(np.float32)
        return csr_matrix((data, (rows, cols)), shape=(1, _VALID_CENTERS.size), dtype=np.float32)

    rows = np.repeat(np.arange(len(positions), dtype=np.int32), peaks.size)
    cols = np.asarray(positions).reshape(-1).astype(np.int32)
    data = (1.0 / (peaks.size * f0[cols])).astype(np.float32)
    return csr_matrix((data, (rows, cols)), shape=(len(positions), _VALID_CENTERS.size), dtype=np.float32)


@contextmanager
def timed(timing: dict, phase: str):
    """Adds the seconds spent in the block to timing[phase] (a phase of TIMING_PHASES)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timing[phase] += time.perf_counter() - start


def timing_line(cell_id: int, timing: dict) -> str:
    return f"cell {cell_id} seconds: " + ", ".join(f"{phase} {timing[phase]:.1f}" for phase in TIMING_PHASES)


def frame_shape(crop: tuple[int, int, int, int] | None) -> tuple[int, int]:
    """(height, width) of the analysed image: the whole movie frame, or the crop."""
    if crop is None:
        return _MOVIE_SHAPE[1], _MOVIE_SHAPE[2]
    return crop[1] - crop[0], crop[3] - crop[2]


def movie_frames(movie: np.ndarray, index, crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """The frames `index` (an index array or a slice) of `movie` as float32 rows of pixels, cut to `crop` (y0, y1, x0, x1) when given."""
    frames = movie[index] if crop is None else movie[index, crop[0]:crop[1], crop[2]:crop[3]]
    frames = np.asarray(frames, dtype=np.float32)
    return frames.reshape(frames.shape[0], int(np.prod(frames.shape[1:])))


def baseline_rels(pre: int, post: int, windows: tuple) -> np.ndarray:
    """The relative frames of the baseline window (consecutive)."""
    rels = np.arange(-pre, post + 1, dtype=np.int32)[relative_indices(windows[3], pre, post)]
    if rels.size == 0:
        raise ValueError("baseline_window selects no frames.")
    return rels


def baseline_average_movie(
    movie: np.ndarray, rels: np.ndarray, crop: tuple[int, int, int, int] | None = None, path: Path | None = None, chunk: int = 512,
) -> np.ndarray:
    """The moving mean of the movie over the consecutive relative frames `rels`: row k is the mean of movie[k + rels[0]] ... movie[k + rels[-1]]
    shifted so that the baseline average of centre c is row c + rels[0] (float32, pixels flattened). Written to the .npy `path` when given.

    The STA is linear in the frames, so the baseline map of any spike train is its event weights times this one movie: one gather per
    train instead of one per baseline frame. Sums are done in float64 over chunks of time, so memory stays bounded."""
    length = int(rels.size)
    n_out = int(movie.shape[0]) - length + 1
    shape = (n_out, movie_frames(movie, slice(0, 1), crop).shape[1])
    out = np.empty(shape, dtype=np.float32) if path is None else np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    for start in range(0, n_out, chunk):
        stop = min(start + chunk, n_out)
        frames = movie_frames(movie, slice(start, stop + length - 1), crop).astype(np.float64)
        running = np.concatenate([np.zeros((1, frames.shape[1])), np.cumsum(frames, axis=0)])
        out[start:stop] = (running[length:] - running[:stop - start]) / length
    return out


def event_columns(events: csr_matrix) -> tuple[np.ndarray, csr_matrix]:
    """The valid-centre columns an event matrix uses (sorted: memmap locality) and the matrix cut to them. Only these frames are gathered."""
    cols = np.unique(events.indices)
    return cols, events[:, cols]


def sta_for_rels(events: csr_matrix, rels: np.ndarray, crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    if _MOVIE is None or _VALID_CENTERS is None or _MOVIE_SHAPE is None:
        raise RuntimeError("Worker was not initialized.")
    rels = np.asarray(rels, dtype=np.int32)
    height, width = frame_shape(crop)
    cols, used = event_columns(events)
    centers = _VALID_CENTERS[cols]
    out = np.empty((events.shape[0], rels.size, height * width), dtype=np.float32)
    for i, rel in enumerate(rels):
        # Astro movies are read in GEVI-flipped polarity, so their baseline is
        # negative while the normalization denominator is abs(F0). This computes
        # (frame - F0) / abs(F0), which is frame / abs(F0) + 1 for flipped astro.
        out[:, i, :] = np.asarray(used @ movie_frames(_MOVIE, centers + int(rel), crop), dtype=np.float32) + 1.0
    return out.reshape(events.shape[0], rels.size, height, width)


def permutation_effects(
    events_list: list[csr_matrix], pre: int, post: int, windows: tuple, crop: tuple[int, int, int, int] | None, timing: dict, time_bin: int = 1,
) -> list[np.ndarray]:
    """The effect maps (STA of each frame of the analysis window minus the baseline map) of every surrogate train; one array per matrix of
    `events_list`. With time_bin > 1 each step is the STA of a bin of that many frames, read from the moving mean of the movie over the bin
    (built here, for `crop`); the steps are the same bins as compute_effect_movie.

    The matrices The matrices (the result types of one cell) are stacked: every frame is gathered once, over the columns any of them uses.

    The baseline map of a train is its event weights times the baseline-averaged movie (the run's, else built here for `crop`). The "+ 1.0"
    of sta_for_rels is in both terms of the difference, so it is left out."""
    if _MOVIE is None or _VALID_CENTERS is None or _MOVIE_SHAPE is None:
        raise RuntimeError("Worker was not initialized.")
    base_rels = baseline_rels(pre, post, windows)
    all_rels = np.arange(-pre, post + 1, dtype=np.int32)
    analysis_rels = all_rels[relative_indices(windows[2], pre, post)]
    step_rels = analysis_rels[:(analysis_rels.size // time_bin) * time_bin:time_bin]  # the first frame of each step
    bin_movie = baseline_average_movie(_MOVIE, np.arange(time_bin, dtype=np.int32), crop) if time_bin > 1 else None
    height, width = frame_shape(crop)
    events = vstack(events_list, format="csr")
    cols, used = event_columns(events)
    centers = _VALID_CENTERS[cols]
    with timed(timing, "baseline map"):
        baseline_movie = _BASELINE_MOVIE if crop is None and _BASELINE_MOVIE is not None else baseline_average_movie(_MOVIE, base_rels, crop)
        baseline_map = np.asarray(used @ np.asarray(baseline_movie[centers + int(base_rels[0])], dtype=np.float32), dtype=np.float32)
    bounds = np.cumsum([0] + [e.shape[0] for e in events_list])
    effects = [np.empty((e.shape[0], step_rels.size, height * width), dtype=np.float32) for e in events_list]
    with timed(timing, "surrogate STA"):
        for i, rel in enumerate(step_rels):
            frames = movie_frames(_MOVIE, centers + int(rel), crop) if bin_movie is None else bin_movie[centers + int(rel)]
            values = np.asarray(used @ frames, dtype=np.float32) - baseline_map
            for effect, first, last in zip(effects, bounds[:-1], bounds[1:]):
                effect[:, i] = values[first:last]
    return [effect.reshape(effect.shape[0], effect.shape[1], height, width) for effect in effects]


def effective_smooth_sigma_px() -> float:
    """Smoothing sigma in analysis (binned) pixels; sub-0.25 px smoothing is a no-op and is skipped."""
    sigma = SMOOTH_SIGMA_FULLRES_PX / max(1, int(SPATIAL_BIN))
    return float(sigma) if sigma >= 0.25 else 0.0


def effective_min_footprint_px() -> int:
    return min_footprint_in_analysis_px(MIN_CLUSTER_FOOTPRINT_FULLRES_PX, SPATIAL_BIN)


def cluster_records(effect: np.ndarray, threshold: float, neg_threshold: float | None = None) -> list[dict]:
    """neg_threshold (positive magnitude) is given only under per_sign_null; None = one symmetric threshold (legacy)."""
    structure = generate_binary_structure(3, CLUSTER_CONNECTIVITY)
    min_footprint = effective_min_footprint_px()
    records = []
    neg_thr = threshold if neg_threshold is None else neg_threshold
    for sign, mask in (("positive", effect >= threshold), ("negative", effect <= -neg_thr)):
        labels, _ = label(mask, structure=structure)
        for label_id, slc in enumerate(find_objects(labels), start=1):
            if slc is None:
                continue
            local = labels[slc] == label_id
            size = int(local.sum())
            if size < MIN_CLUSTER_SIZE:
                continue
            if (min_footprint or MIN_CLUSTER_FRAMES) and not passes_extent(local, min_footprint, MIN_CLUSTER_FRAMES):
                continue
            values = effect[slc][local]
            mass = float(np.abs(values).sum())
            if mass < MIN_CLUSTER_MASS:
                continue
            full = np.zeros_like(effect, dtype=bool)
            full[slc] = local
            records.append({"sign": sign, "mass": mass, "size": size, "peak_value": float(values[np.argmax(np.abs(values))]), "mask": full})
    if MERGE_SIMILAR_CLUSTERS and len(records) > 1:
        records = _merge_similar_clusters(records, effect, MERGE_MAX_DISTANCE_PX, MERGE_MIN_TRACE_CORR)
    return records


def summarize_clusters(
    effect: np.ndarray,
    threshold: float,
    null_max: np.ndarray,
    neg_threshold: float | None = None,
    null_max_neg: np.ndarray | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Legacy: one threshold, one pooled null. per_sign_null (neg_threshold and null_max_neg given): `threshold` / `null_max`
    are the positive-sign ones; p_value = min(1, 2 * p_value_sign) with p_value_sign against the cluster's own-sign null."""
    per_sign = neg_threshold is not None
    records = sorted(cluster_records(effect, threshold, neg_threshold), key=lambda c: c["mass"], reverse=True)
    sig_mask = np.zeros_like(effect, dtype=bool)
    summary = []
    for cluster_id, record in enumerate(records, start=1):
        if per_sign:
            p_sign, p_value = per_sign_p_values(record["mass"], record["sign"], null_max, null_max_neg)
        else:
            p_value = float((1 + np.sum(null_max >= record["mass"])) / (len(null_max) + 1))
        is_sig = bool(p_value <= CLUSTER_ALPHA)
        if is_sig:
            sig_mask |= record["mask"]
        row = {
            "cluster_id": cluster_id,
            "sign": record["sign"],
            "mass": float(record["mass"]),
            "size": int(record["size"]),
            "peak_value": float(record["peak_value"]),
            "p_value": p_value,
            "is_significant": is_sig,
        }
        if per_sign:
            row["p_value_sign"] = p_sign
        summary.append(row)
    return summary, sig_mask


def save_cluster_result(path: Path, result: dict) -> None:
    cluster_array = np.asarray([
        (c["cluster_id"], c["sign"], c["mass"], c["size"], c["peak_value"], c["p_value"], c["is_significant"])
        for c in result["cluster_summary"]
    ], dtype=object)
    payload = dict(
        cell_id=np.asarray(result["cell_id"], dtype=np.int32),
        spike_group=np.asarray(result["spike_group"]),
        peaks=np.asarray(result["peaks"], dtype=np.int32),
        observed_sta_movie=np.asarray(result["observed_sta_movie"], dtype=np.float32),
        observed_effect_movie=np.asarray(result["observed_effect_movie"], dtype=np.float32),
        observed_post_pre_map=np.asarray(result["observed_post_pre_map"], dtype=np.float32),
        significant_cluster_mask=np.asarray(result["significant_cluster_mask"], dtype=bool),
        perm_effects=np.asarray(result["perm_effects"], dtype=np.float32),
        null_max_cluster_mass=np.asarray(result["null_max_cluster_mass"], dtype=np.float32),
        cluster_summary=cluster_array,
        cluster_threshold=np.asarray(result["cluster_threshold"], dtype=np.float32),
        permutation_positions=np.asarray(result["permutation_positions"], dtype=np.int32),
        pre_frames=np.asarray(result["pre_frames"], dtype=np.int32),
        post_frames=np.asarray(result["post_frames"], dtype=np.int32),
        pre_window=np.asarray(result["pre_window"], dtype=np.int32),
        post_window=np.asarray(result["post_window"], dtype=np.int32),
        analysis_window=np.asarray(result["analysis_window"], dtype=np.int32),
        baseline_window=np.asarray(result["baseline_window"], dtype=np.int32),
        time_bin_frames=np.asarray(result["time_bin_frames"], dtype=np.int32),
        spatial_bin=np.asarray(result.get("spatial_bin", SPATIAL_BIN), dtype=np.int32),
        n_permutations=np.asarray(N_PERMUTATIONS, dtype=np.int32),
        min_spikes_for_cluster=np.asarray(MIN_SPIKES_FOR_CLUSTER, dtype=np.int32),
        rng_seed=np.asarray(RNG_SEED, dtype=np.int32),
        silence_frames=np.asarray(result["silence_frames"], dtype=np.int32),
        n_peaks_before_silence=np.asarray(result["n_peaks_before_silence"], dtype=np.int32),
        n_surrogate_frames=np.asarray(result["n_surrogate_frames"], dtype=np.int32),
        result_types=np.asarray(RESULT_TYPES),
        movie_shape=np.asarray(result.get("movie_shape", ()), dtype=np.int32),
        n_valid_centers=np.asarray(result.get("n_valid_centers", -1), dtype=np.int32),
        voxel_alpha=np.asarray(VOXEL_ALPHA, dtype=np.float32),
        cluster_alpha=np.asarray(CLUSTER_ALPHA, dtype=np.float32),
        cluster_connectivity=np.asarray(CLUSTER_CONNECTIVITY, dtype=np.int32),
        min_cluster_size=np.asarray(MIN_CLUSTER_SIZE, dtype=np.int32),
        min_cluster_mass=np.asarray(MIN_CLUSTER_MASS, dtype=np.float32),
        merge_similar_clusters=np.asarray(MERGE_SIMILAR_CLUSTERS),
        merge_max_distance_px=np.asarray(MERGE_MAX_DISTANCE_PX, dtype=np.float32),
        merge_min_trace_corr=np.asarray(MERGE_MIN_TRACE_CORR, dtype=np.float32),
        null_standardization=np.asarray(NULL_STANDARDIZATION),
        smooth_sigma_px=np.asarray(effective_smooth_sigma_px(), dtype=np.float32),
        smooth_sigma_fullres_px=np.asarray(SMOOTH_SIGMA_FULLRES_PX, dtype=np.float32),
        min_cluster_footprint_px=np.asarray(effective_min_footprint_px(), dtype=np.int32),
        min_cluster_footprint_fullres_px=np.asarray(MIN_CLUSTER_FOOTPRINT_FULLRES_PX, dtype=np.float32),
        min_cluster_frames=np.asarray(MIN_CLUSTER_FRAMES, dtype=np.int32),
        significance_statistic=np.asarray(result.get("significance_statistic", "effect")),
    )
    if result.get("valid_mask") is not None:  # the analysed pixels: the border outside them was left out of the test
        payload["valid_mask"] = np.asarray(result["valid_mask"], dtype=bool)
    payload.update({key: np.asarray(value) for key, value in HYPOTHESIS.items()})  # the hypothesis name and its settings (seconds)
    payload["per_sign_null"] = np.asarray(PER_SIGN_NULL)
    if PER_SIGN_NULL:
        # cluster_threshold above is the positive threshold; null_max_cluster_mass the elementwise max of the two nulls.
        # cluster_summary's p_value column is the Bonferroni-corrected p; the uncorrected own-sign p is saved aligned to its rows.
        payload["cluster_threshold_pos"] = np.asarray(result["cluster_threshold_pos"], dtype=np.float32)
        payload["cluster_threshold_neg"] = np.asarray(result["cluster_threshold_neg"], dtype=np.float32)
        payload["null_max_cluster_mass_pos"] = np.asarray(result["null_max_cluster_mass_pos"], dtype=np.float32)
        payload["null_max_cluster_mass_neg"] = np.asarray(result["null_max_cluster_mass_neg"], dtype=np.float32)
        payload["cluster_p_value_sign"] = np.asarray([c["p_value_sign"] for c in result["cluster_summary"]], dtype=np.float64)
    if result.get("null_sd_map") is not None:
        payload["null_sd_map"] = np.asarray(result["null_sd_map"], dtype=np.float32)
    if result.get("significance_statistic", "effect") != "effect" or effective_smooth_sigma_px() > 0:
        payload["observed_effect_raw_movie"] = np.asarray(result["observed_effect_raw_movie"], dtype=np.float32)
    for key in (
        "anatomy_radius_binned",
        "anatomy_center_yx_binned",
        "anatomy_crop_bounds",
        "anatomy_spatial_bin",
        "anatomy_source_radius_binned",
        "anatomy_source_spatial_bin",
        "anatomy_fullres_radius_px",
    ):
        if key in result:
            payload[key] = np.asarray(result[key], dtype=np.int32)
    for key in ("behavioral_group", "source_behavioral_segments"):
        if key in result:
            payload[key] = np.asarray(result[key])
    np.savez(path, **payload)


def prepare_test_statistic(raw_effect: np.ndarray, perm_effects: np.ndarray, valid: np.ndarray | None = None):
    """Map raw effect maps to the statistic that is thresholded and clustered.

    `valid` ((H, W) boolean, None = every pixel): the border outside it is set to 0 before and after the smoothing, and left out of the sd floor,
    so nothing in it can enter a cluster or a threshold.

    Optional spatial smoothing, then optional per-pixel standardization by the permutation-null sd
    (removes the brightness-dependence of the pooled voxel threshold). Returns (observed, permuted, sd_map|None).
    With default settings both inputs are returned unchanged (legacy behaviour).
    """
    observed, permuted = raw_effect, perm_effects
    if valid is not None:
        observed, permuted = observed * valid, permuted * valid
    sigma = effective_smooth_sigma_px()
    if sigma > 0:
        observed = smooth_spatial(observed, sigma)
        permuted = smooth_spatial(permuted, sigma)
        if valid is not None:
            observed, permuted = observed * valid, permuted * valid
    sd_map = None
    if NULL_STANDARDIZATION == "per_pixel_z":
        sd_map = null_sd_map(permuted, NULL_SD_FLOOR_QUANTILE, valid)
        observed = standardize(observed, sd_map)
        permuted = standardize(permuted, sd_map)
    return observed, permuted, sd_map


def find_significant_clusters(obs_stat: np.ndarray, perm_stat: np.ndarray, valid: np.ndarray | None = None):
    """Voxel threshold(s), max-mass null(s) and cluster p-values of the tested statistic. The voxel thresholds are taken from the pixels of `valid`
    only (None = all), so the masked border does not lower them.

    Returns (threshold, null_max, clusters, sig_mask, extra). Legacy (PER_SIGN_NULL off): one threshold from |perm|,
    one pooled null, extra = {}. PER_SIGN_NULL on: see sta_stats_utils.per_sign_thresholds / per_sign_p_values;
    threshold is the positive one, null_max the pooled max of the two nulls, extra holds the per-sign thresholds and nulls.
    """
    voxels = perm_stat if valid is None else perm_stat[..., valid]
    if not PER_SIGN_NULL:
        threshold = float(np.quantile(np.abs(voxels), 1.0 - VOXEL_ALPHA))
        null_max = np.asarray([max((c["mass"] for c in cluster_records(effect, threshold)), default=0.0) for effect in perm_stat], dtype=np.float32)
        clusters, sig_mask = summarize_clusters(obs_stat, threshold, null_max)
        return threshold, null_max, clusters, sig_mask, {}
    # opt-in: separate voxel thresholds and separate max-mass nulls for depolarizing (+) and hyperpolarizing (-) clusters
    threshold, neg_threshold = per_sign_thresholds(voxels, VOXEL_ALPHA)
    masses = np.asarray([max_mass_by_sign(cluster_records(effect, threshold, neg_threshold)) for effect in perm_stat], dtype=np.float32)
    null_pos, null_neg = masses[:, 0].copy(), masses[:, 1].copy()
    null_max = np.maximum(null_pos, null_neg)  # pooled max, kept in the legacy field for readers that only know it
    clusters, sig_mask = summarize_clusters(obs_stat, threshold, null_pos, neg_threshold, null_neg)
    extra = {
        "cluster_threshold_pos": threshold,
        "cluster_threshold_neg": neg_threshold,
        "null_max_cluster_mass_pos": null_pos,
        "null_max_cluster_mass_neg": null_neg,
    }
    return threshold, null_max, clusters, sig_mask, extra


def save_observed_sta(cell_id: int, group: str, observed_sta: np.ndarray, pre: int, n_spikes: int, out_dir: Path) -> None:
    if not SAVE_BINNED_STA_MOVIES:
        return
    # one saved copy: the .tif in the experiment folder; the .npy in the run folder only when the .tif could not be written
    if SAVE_FIJI_TIFS and export_fiji_sta(cell_id, group, observed_sta, pre, n_spikes):
        return
    sta_name = g.STA_MOVIE_NPY_PREFIX if group == g.ALL_GROUP else f"{g.STA_MOVIE_NPY_PREFIX}_{group}"
    np.save(path_for(PathKind.STA_MOVIE_NPY, out_dir.parent, cell=cell_id, name=sta_name, bin=SPATIAL_BIN), observed_sta.astype(np.float32))


def cluster_skip_reason(cell_id: int, group: str, peaks: np.ndarray, pool: np.ndarray, silence: tuple[int, int]) -> str:
    """Why a group with enough spikes gets no cluster test (fewer random frames than spikes), else ''. Prints the warning."""
    if peaks.size < MIN_SPIKES_FOR_CLUSTER or pool.size >= peaks.size:
        return ""
    reason = (
        f"{peaks.size} spikes but only {pool.size} frames without another spike within the silence window "
        f"({silence[0]} frames before, {silence[1]} after): no random frames to compare with"
    )
    print(f"WARNING: cell {cell_id} {group}: cluster test skipped: {reason}")
    return reason


def tested_group_result(
    cell_id: int,
    group: str,
    peaks: np.ndarray,
    observed_sta: np.ndarray,
    perm_effects: dict,
    positions: np.ndarray,
    run: tuple,
    pool: np.ndarray,
    n_before: int,
    timing: dict,
    time_bin: int = 1,
    valid: np.ndarray | None = None,
) -> tuple[dict, list[dict]]:
    """(result to save, cluster summary) of one group. `perm_effects` is {group index: surrogate effect maps}; this group's entry is
    taken out so its memory is freed as soon as the test statistic exists. `run` = (pre, post, windows, silence, group index)."""
    pre, post, windows, silence, index = run
    with timed(timing, "clusters"):
        observed_effect = compute_effect_movie(observed_sta, pre, post, *windows, time_bin=time_bin)
        obs_stat, perm_stat, sd_map = prepare_test_statistic(observed_effect["effect_movie"], perm_effects.pop(index), valid)
        threshold, null_max, clusters, sig_mask, threshold_extra = find_significant_clusters(obs_stat, perm_stat, valid)
    result = {
        **threshold_extra,
        "cell_id": cell_id,
        "spike_group": group,
        "peaks": peaks,
        "observed_sta_movie": observed_sta,
        # observed_effect_movie / perm_effects hold the TESTED statistic (effect, or per-pixel z when
        # null_standardization == "per_pixel_z"); downstream code re-derives clusters from them + cluster_threshold.
        "observed_effect_movie": obs_stat,
        "observed_effect_raw_movie": observed_effect["effect_movie"],
        "observed_post_pre_map": observed_effect["post_pre_map"],
        "significant_cluster_mask": sig_mask,
        "perm_effects": perm_stat,
        "null_sd_map": sd_map,
        "significance_statistic": "z" if sd_map is not None else "effect",
        "null_max_cluster_mass": null_max,
        "cluster_summary": clusters,
        "cluster_threshold": threshold,
        "permutation_positions": positions,
        "pre_frames": pre,
        "post_frames": post,
        "pre_window": windows[0],
        "post_window": windows[1],
        "analysis_window": windows[2],
        "baseline_window": windows[3],
        "time_bin_frames": time_bin,
        "spatial_bin": SPATIAL_BIN,
        "movie_shape": _MOVIE_SHAPE,
        "n_valid_centers": _VALID_CENTERS.size if _VALID_CENTERS is not None else -1,
        "silence_frames": silence,
        "n_peaks_before_silence": n_before,
        "n_surrogate_frames": pool.size,
        "valid_mask": valid,
    }
    return result, clusters


def group_row(cell_id: int, group: str, peaks: np.ndarray, n_before: int, pool: np.ndarray, clusters: list[dict], cluster_path: Path | str, skip_reason: str) -> dict:
    row = {
        "cell_id": cell_id,
        "spike_group": group,
        "hypothesis": HYPOTHESIS["hypothesis"],
        "n_peaks": int(peaks.size),
        "n_peaks_before_silence": int(n_before),
        "n_surrogate_frames": int(pool.size),
        "n_clusters": int(len(clusters)),
        "n_significant_clusters": int(sum(c["is_significant"] for c in clusters)),
        "max_cluster_mass": float(max((c["mass"] for c in clusters), default=0.0)),
        "cluster_npz_path": str(cluster_path),
    }
    if skip_reason:
        row.update({"status": "skipped", "skip_reason": skip_reason})
    return row


def analyze_groups(
    cell_id: int,
    groups: list[tuple[str, np.ndarray, int]],
    pre: int,
    post: int,
    windows: tuple,
    out_dir: Path,
    silence: tuple[int, int],
    pool: np.ndarray,
    timing: dict,
    crop: tuple[int, int, int, int] | None = None,
    f0_by_pos: np.ndarray | None = None,
    extra_result: dict | None = None,
    time_bin: int = 1,
    valid: np.ndarray | None = None,
) -> list[dict]:
    """The observed STA, the random-frame test and the saved files of one or more spike groups of one cell, in one pass over the movie.

    `groups` = [(result type, peaks after the silence rule, number of spikes before it)]; they share the movie, `pool` (indices into the valid
    centres where the chance copies may be drawn: the cell's silence pool), `crop` and `f0_by_pos`, so the frames are gathered once for all of them. `time_bin`: frames per averaged step of the tested window. `valid`: (H, W) mask of the analysed pixels (None = all).
    A group with fewer pool frames than spikes gets no cluster test. `timing`: seconds per phase (TIMING_PHASES), added to."""
    for group, peaks, n_before in groups:
        print(f"cell {cell_id} {group}: {peaks.size} peaks ({n_before} before the silence rule), {pool.size} random frames")
    rels = np.arange(-pre, post + 1, dtype=np.int32)
    with timed(timing, "observed STA"):
        observed = sta_for_rels(vstack([event_matrix(peaks, f0_by_pos=f0_by_pos) for _, peaks, _ in groups], format="csr"), rels, crop)
    with timed(timing, "saving"):
        for (group, peaks, _), observed_sta in zip(groups, observed):
            save_observed_sta(cell_id, group, observed_sta, pre, int(peaks.size), out_dir)
    skips = [cluster_skip_reason(cell_id, group, peaks, pool, silence) for group, peaks, _ in groups]
    tested = [k for k, (_, peaks, _) in enumerate(groups) if peaks.size >= MIN_SPIKES_FOR_CLUSTER and not skips[k]]
    with timed(timing, "surrogate events"):
        positions = {k: surrogate_positions(cell_id, int(groups[k][1].size), pool) for k in tested}
        perm_events = [event_matrix(groups[k][1], positions[k], f0_by_pos=f0_by_pos) for k in tested]
    perm_effects = dict(zip(tested, permutation_effects(perm_events, pre, post, windows, crop, timing, time_bin))) if tested else {}
    rows = []
    for k, (group, peaks, n_before) in enumerate(groups):
        cluster_path, clusters = out_dir / result_file_name(group), []
        if k in tested:
            result, clusters = tested_group_result(
                cell_id, group, peaks, observed[k], perm_effects, positions[k], (pre, post, windows, silence, k), pool, n_before, timing, time_bin, valid,
            )
            result.update(extra_result or {})
            with timed(timing, "saving"):
                if OVERWRITE or not cluster_path.exists():
                    save_cluster_result(cluster_path, result)
            del result
        elif OVERWRITE and cluster_path.exists():
            cluster_path.unlink()
            cluster_path = ""
        else:
            cluster_path = ""
        rows.append(group_row(cell_id, group, peaks, n_before, pool, clusters, cluster_path, skips[k]))
    return rows


def anatomy_analysis_kwargs(cell_id: int, baseline_window: tuple[int, int], anatomy_settings: dict | None = None, margins: tuple[int, int, int, int] | None = None) -> dict:
    if _MOVIE_SHAPE is None or _VALID_CENTERS is None or _SUITE2P_CTX is None or _ANATOMY_RADIUS is None:
        raise RuntimeError("Anatomy result requested but anatomy crop settings were not initialized.")
    center = neuron_centroid_binned(cell_id, SPATIAL_BIN, _MOVIE_SHAPE[1:], _SUITE2P_CTX)
    crop = crop_bounds(center, _ANATOMY_RADIUS, _MOVIE_SHAPE[1:])
    extra = {
        "anatomy_radius_binned": _ANATOMY_RADIUS,
        "anatomy_center_yx_binned": center,
        "anatomy_crop_bounds": crop,
    }
    if anatomy_settings:
        extra.update(anatomy_settings)
    valid = analysis_valid(margins)
    return {
        "crop": crop,
        "f0_by_pos": crop_f0_for_valid_centers(crop, _VALID_CENTERS, baseline_window),
        "extra_result": extra,
        "valid": None if valid is None else valid[crop[0]:crop[1], crop[2]:crop[3]],
    }


def analysis_cfg(pre: int, post: int, windows: tuple, n_frames: int, time_bin: int = 1, border_margins_px: tuple | None = None, **extra) -> dict:
    return {
        "time_bin": int(time_bin),
        "border_margins_px": border_margins_px,
        "pre": pre,
        "post": post,
        "windows": tuple(windows),
        "n_frames": int(n_frames),
        **extra,
    }


def sta_run_parameters(
    fps: int,
    pre: int,
    post: int,
    windows: tuple,
    standard_result_types: tuple[str, ...],
    anatomy_settings: dict | None,
    silence: tuple[int, int],
    border_margins: tuple[int, int, int, int] | None = None,
) -> dict:
    return {
        "mask_border": bool(MASK_BORDER),
        "border_margins_px": [int(v) for v in border_margins] if border_margins else [],
        "home_dir": str(home_dir()),
        "source_home_dir": str(source_home_dir()),
        "spatial_bin": int(SPATIAL_BIN),
        "anatomy_spatial_bin": int(ANATOMY_SPATIAL_BIN),
        "cell_ids": CELL_IDS,
        "result_types": list(RESULT_TYPES),
        "standard_result_types": list(standard_result_types),
        "behavioral_groups": list(BEHAVIORAL_GROUPS) if "behavioral" in RESULT_TYPES else [],
        "n_permutations": int(N_PERMUTATIONS),
        "min_spikes_for_cluster": int(MIN_SPIKES_FOR_CLUSTER),
        "n_workers": int(N_WORKERS),
        "voxel_alpha": float(VOXEL_ALPHA),
        "cluster_alpha": float(CLUSTER_ALPHA),
        "cluster_connectivity": int(CLUSTER_CONNECTIVITY),
        "min_cluster_size": int(MIN_CLUSTER_SIZE),
        "min_cluster_mass": float(MIN_CLUSTER_MASS),
        "null_standardization": NULL_STANDARDIZATION,
        "null_sd_floor_quantile": float(NULL_SD_FLOOR_QUANTILE),
        "per_sign_null": bool(PER_SIGN_NULL),
        "exclude_spikes_near_cuts": bool(EXCLUDE_SPIKES_NEAR_CUTS),
        "smooth_sigma_fullres_px": float(SMOOTH_SIGMA_FULLRES_PX),
        "min_cluster_footprint_fullres_px": float(MIN_CLUSTER_FOOTPRINT_FULLRES_PX),
        "min_cluster_frames": int(MIN_CLUSTER_FRAMES),
        "motion_diagnostics": bool(MOTION_DIAGNOSTICS),
        "motion_flag_alpha": float(MOTION_FLAG_ALPHA),
        "motion_permutations": int(MOTION_PERMUTATIONS),
        "motion_reg_quality_bin": int(MOTION_REG_QUALITY_BIN),
        "merge_similar_clusters": bool(MERGE_SIMILAR_CLUSTERS),
        "merge_max_distance_px": float(MERGE_MAX_DISTANCE_PX),
        "merge_min_trace_corr": float(MERGE_MIN_TRACE_CORR),
        "rng_seed": int(RNG_SEED),
        "save_binned_sta_movies": bool(SAVE_BINNED_STA_MOVIES),
        "save_fiji_tifs": bool(SAVE_FIJI_TIFS),
        "overwrite": bool(OVERWRITE),
        "summary_csv": SUMMARY_CSV,
        "fps": float(fps),
        **HYPOTHESIS,
        "silence_frames": [int(v) for v in silence],
        "pre_frames": int(pre),
        "post_frames": int(post),
        "pre_window": [int(v) for v in windows[0]],
        "post_window": [int(v) for v in windows[1]],
        "analysis_window": [int(v) for v in windows[2]],
        "baseline_window": [int(v) for v in windows[3]],
        "anatomy_settings": anatomy_settings or {},
    }


def save_sta_run_parameters(sta_dir: Path, params: dict) -> None:
    sta_dir.mkdir(parents=True, exist_ok=True)
    json_path = path_for(PathKind.STA_USED_PARAMETERS_JSON, sta_dir)
    write_json(json_path, params)
    npz_payload = {}
    for key, value in params.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                npz_payload[f"{key}_{subkey}"] = np.asarray(subvalue)
        else:
            npz_payload[key] = np.asarray(value if value is not None else "")
    np.savez(path_for(PathKind.STA_USED_PARAMETERS_NPZ, sta_dir), **npz_payload)


def cell_output_dir(sta_dir_s: str, cell_id: int) -> Path:
    out_dir = path_for(PathKind.CELL_DIR, sta_dir_s, cell=cell_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def surrogate_positions(cell_id: int, n_spikes: int, pool: np.ndarray, n_copies: int | None = None) -> np.ndarray:
    """The chance copies of one spike train: (n_copies, n_spikes; default N_PERMUTATIONS) positions (indices into the valid centres), each row n_spikes distinct
    random frames drawn from `pool`, the cell's silence pool (the valid spike frames with no real spike of the cell in the silence window;
    all valid centres when the silence rule is off, which gives the same draws as before the rule existed). Same seed for the same
    (rng_seed, cell, number of spikes), so the null of a cell does not change between runs."""
    if n_spikes > pool.size:
        raise ValueError(f"Cell {cell_id} has {n_spikes} spikes but only {pool.size} valid spike frames exist: no random surrogate can be drawn.")
    rng = np.random.default_rng([RNG_SEED, int(cell_id) if cell_id != g.POOLED_CELL_ID else 10 ** 6, int(n_spikes)])  # a seed is not negative
    draws = np.stack([rng.choice(int(pool.size), size=int(n_spikes), replace=False) for _ in range(N_PERMUTATIONS if n_copies is None else n_copies)])
    return np.asarray(pool)[draws].astype(np.int32)


def save_neuron_sta_metadata(out_dir: Path, cell_id: int, pre: int, post: int, suffix: str = "", **extra) -> None:
    payload = {
        "cell_id": np.asarray(cell_id, dtype=np.int32),
        "pre_frames": np.asarray(pre, dtype=np.int32),
        "post_frames": np.asarray(post, dtype=np.int32),
        "source_spike_labels": str(path_for(PathKind.SPIKE_CELL_DIR, spike_analysis_dir(), cell=cell_id) / g.SPIKE_LABELS),
        **extra,
    }
    np.savez(path_for(PathKind.NEURON_STA, out_dir.parent, cell=cell_id, suffix=suffix), **payload)


def run_pool(worker_fn, cfg: dict, cells: list[int], sta_dir: Path, init_args: tuple, movie: np.ndarray | None = None) -> list[dict]:
    """Runs `worker_fn` for every cell. `movie`: the pool's full-frame movie; its baseline-averaged copy is written once to a temporary folder
    inside the STA folder (deleted at the end) that all workers memory-map. None (the anatomy pool): each worker builds it per cell for its crop."""
    rows = []
    with tempfile.TemporaryDirectory(dir=sta_dir, prefix="baseline_") as tmp:
        baseline_path = None
        if movie is not None:
            baseline_path = path_for(PathKind.BASELINE_AVERAGE_MOVIE, tmp)
            baseline_average_movie(movie, baseline_rels(cfg["pre"], cfg["post"], cfg["windows"]), path=baseline_path).flush()
        with mp.Pool(N_WORKERS, initializer=init_worker, initargs=(*init_args, str(baseline_path) if baseline_path else None)) as pool:
            for cell_rows in pool.imap_unordered(worker_fn, [(cell_id, cfg, str(sta_dir)) for cell_id in cells]):
                rows.extend(cell_rows)
    return rows


def run_motion_diagnostics(cell_id: int, peaks: np.ndarray, pool: np.ndarray, cfg: dict, out_dir: Path) -> dict:
    """Spike-triggered motion test for one cell; writes cell_<id>/motion_sta.npz. Returns the cell's motion status for the summary csv
    (the one place that decides it). Never raises (diagnostic only)."""
    series = cfg.get("motion_series")
    not_tested = {g.MOTION_STATUS_COLUMN: g.MOTION_NOT_TESTED}
    if series is None or not MOTION_DIAGNOSTICS:
        return not_tested
    try:
        windows = cfg["windows"]
        surrogate_frames = _VALID_CENTERS[surrogate_positions(cell_id, int(peaks.size), pool, MOTION_PERMUTATIONS)]
        stats = motion_sta_stats(
            series, peaks, surrogate_frames, cfg["pre"], cfg["post"],
            baseline_window=tuple(int(v) for v in windows[3]),
            analysis_window=tuple(int(v) for v in windows[2]),
            flag_alpha=MOTION_FLAG_ALPHA,
        )
        np.savez(path_for(PathKind.MOTION_STA, out_dir.parent, cell=cell_id), cell_id=np.asarray(cell_id, dtype=np.int32), **stats)
        return {g.MOTION_STATUS_COLUMN: g.MOTION_SUSPECTED if bool(stats["flag"]) else g.MOTION_OK, "motion_p": float(stats["p_family"])}
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print(f"WARNING: motion diagnostics failed for cell {cell_id}: {exc}")
        return not_tested


def effective_motion_reg_bin() -> int:
    """Bin of the movie used for registration quality: the configured one, else the run's own spatial bin
    (its cache is already in use), else 8. Never 1: the full-resolution movie is too large for a per-frame correlation."""
    if MOTION_REG_QUALITY_BIN > 1:
        return int(MOTION_REG_QUALITY_BIN)
    return int(SPATIAL_BIN) if int(SPATIAL_BIN) > 1 else 8


def build_run_motion_series(paths, n_frames: int) -> dict | None:
    """Shifts (+ registration quality from the bin-8 astro movie) aligned to the analysed (kept) frames."""
    if not MOTION_DIAGNOSTICS:
        return None
    try:
        shifts = analysed_shifts(Path(paths.mc_dir).parents[1], int(n_frames))
    except (OSError, ValueError) as exc:
        print(f"WARNING: motion diagnostics skipped: {exc}")
        return None
    reg = None
    reg_bin = effective_motion_reg_bin()
    try:
        print(f"Registration quality from the bin-{reg_bin} astro movie")
        reg = registration_quality(ensure_binned_movie(paths, reg_bin))
        if len(reg) != int(n_frames):
            print(f"WARNING: registration-quality series has {len(reg)} frames vs {n_frames}; dropping it")
            reg = None
    except (OSError, ValueError) as exc:
        print(f"WARNING: registration-quality series unavailable: {exc}")
    series = build_motion_series(shifts, reg)
    paths.sta_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path_for(PathKind.MOTION_SERIES, paths.sta_dir), reg_quality_bin=np.asarray(reg_bin if reg is not None else 0), **series)
    return series


def run_border_margins(paths) -> tuple[int, int, int, int] | None:
    """(top, bottom, left, right) border margins in full-resolution pixels from the applied shifts of the analysed frames, None when mask_border is off.
    Raises ValueError saying what is missing when the shifts cannot be read or matched: without them the border cannot be left out."""
    if not MASK_BORDER:
        return None
    try:
        n_frames = int(read_astro_movie(str(paths.astro_movie_path), as_memmap=True).shape[0])
        margins = border_margins_px(analysed_shifts(Path(paths.mc_dir).parents[1], n_frames, with_patches=True), BORDER_EXTRA_PX)
    except (OSError, ValueError) as exc:
        raise ValueError(f"The border mask needs the applied shifts of the analysed frames: {exc} Fix that, or set mask_border: false to run without the mask.") from exc
    print(f"Border left out of the test (top, bottom, left, right): {margins} px", flush=True)
    return margins


def analysis_valid(margins: tuple[int, int, int, int] | None) -> np.ndarray | None:
    """The mask of the analysed pixels of this pool's movie (None = no mask)."""
    return None if margins is None else valid_mask(_MOVIE_SHAPE[1:], margins, SPATIAL_BIN)


def save_global_trends(paths, movie: np.ndarray, motion_series: dict | None, fps: float, cuts: list[int], margins: tuple[int, int, int, int] | None) -> None:
    """global_trends.npz of the run: the traces and the amplitude maps of the signals the whole field shares (utils/global_trends.py), from the movie
    the standard result types analyse. A diagnostic like the motion test: a problem is reported and does not fail the run."""
    print("Global trends of the field (frame mean and motion series; amplitude maps) ...", flush=True)
    try:
        trends = compute_global_trends(movie, motion_series)
        paths.sta_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            path_for(PathKind.GLOBAL_TRENDS, paths.sta_dir), **trends, fps=np.asarray(fps), spatial_bin=np.asarray(SPATIAL_BIN), cuts=np.asarray(cuts, dtype=np.int32),
            border_margins_px=np.asarray(margins or (0, 0, 0, 0), dtype=np.int32),
        )
    except (OSError, ValueError, MemoryError) as exc:
        print(f"WARNING: the global trends were not saved ({exc}). The Hypotheses tab needs them: rerun STA Maps.", flush=True)


def process_cell(args: tuple[int, dict, str]) -> list[dict]:
    cell_id, cfg, sta_dir_s = args
    pre, post = cfg["pre"], cfg["post"]
    windows = cfg["windows"]
    out_dir = cell_output_dir(sta_dir_s, cell_id)

    timing = dict.fromkeys(TIMING_PHASES, 0.0)
    silence = cfg["silence"]
    with timed(timing, "spikes+pool"):
        spikes, n_before = load_spikes(cell_id, cfg["n_frames"], pre, post, _VALID_CENTERS if EXCLUDE_SPIKES_NEAR_CUTS else None, silence)
        pool = cell_surrogate_pool(cell_id, _VALID_CENTERS, silence)
    save_neuron_sta_metadata(out_dir, cell_id, pre, post)
    motion_row = run_motion_diagnostics(cell_id, spikes, pool, cfg, out_dir)

    result_types = cfg.get("result_types", RESULT_TYPES)
    rows = analyze_groups(cell_id, [(g.ALL_GROUP, spikes, n_before)], pre, post, windows, out_dir, silence, pool, timing, time_bin=cfg["time_bin"], valid=analysis_valid(cfg["border_margins_px"])) if g.ALL_GROUP in result_types else []
    if g.ANATOMY_GROUP in result_types:  # its own crop of the movie: a pass of its own
        kwargs = anatomy_analysis_kwargs(cell_id, windows[3], cfg.get("anatomy_settings"), cfg["border_margins_px"])
        anatomy = [(g.ANATOMY_GROUP, spikes, n_before)]
        rows.extend(analyze_groups(cell_id, anatomy, pre, post, windows, out_dir, silence, pool, timing, time_bin=cfg["time_bin"], **kwargs))
    for row in rows:
        row.update(motion_row)
    print(timing_line(cell_id, timing), flush=True)
    return rows


def process_cell_behavioral(args: tuple[int, dict, str]) -> list[dict]:
    cell_id, cfg, sta_dir_s = args
    pre, post = cfg["pre"], cfg["post"]
    windows = cfg["windows"]
    behavioral_group = cfg["behavioral_group"]
    source_frames = np.asarray(cfg["source_frames"], dtype=np.int32)
    out_dir = cell_output_dir(sta_dir_s, cell_id)

    timing = dict.fromkeys(TIMING_PHASES, 0.0)
    silence = cfg["silence"]
    with timed(timing, "spikes+pool"):
        peaks, n_before = load_behavioral_spikes(cell_id, source_frames, cfg["n_frames"], pre, post, silence)
        pool = cell_surrogate_pool(cell_id, source_frames[_VALID_CENTERS], silence)  # the silence is measured in the original frame numbers
    group = f"behavioral_{behavioral_group}"
    save_neuron_sta_metadata(
        out_dir,
        cell_id,
        pre,
        post,
        suffix=f"_{group}",
        source_behavioral_segments=str(cfg["source_behavioral_segments"]),
        behavioral_group=np.asarray(behavioral_group),
    )

    rows = analyze_groups(
        cell_id,
        [(group, peaks, n_before)],
        pre,
        post,
        windows,
        out_dir,
        silence,
        pool,
        timing,
        time_bin=cfg["time_bin"],
        valid=analysis_valid(cfg["border_margins_px"]),
        extra_result={
            "behavioral_group": np.asarray(behavioral_group),
            "source_behavioral_segments": np.asarray(str(cfg["source_behavioral_segments"])),
        },
    )
    print(timing_line(cell_id, timing), flush=True)
    return rows


def skipped_behavioral_rows(cells: list[int], behavioral_group: str, source_frames: np.ndarray, reason: str) -> list[dict]:
    group = f"behavioral_{behavioral_group}"
    return [
        {
            "cell_id": int(cell_id),
            "spike_group": group,
            "hypothesis": HYPOTHESIS["hypothesis"],
            "n_peaks": 0,
            "n_clusters": 0,
            "n_significant_clusters": 0,
            "max_cluster_mass": 0.0,
            "cluster_npz_path": "",
            "status": "skipped",
            "skip_reason": reason,
            "source_frame_count": int(source_frames.size),
        }
        for cell_id in cells
    ]


def write_summary(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cell_id",
        "spike_group",
        "status",
        "skip_reason",
        "source_frame_count",
        "hypothesis",
        "n_peaks",
        "n_peaks_before_silence",
        "n_surrogate_frames",
        "n_clusters",
        "n_significant_clusters",
        "max_cluster_mass",
        "cluster_npz_path",
    ]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["cell_id"], r["spike_group"])))


def main() -> None:
    global RUN_CONFIG, RESULT_TYPES
    RUN_CONFIG = load_config(parse_args().config)
    apply_config(RUN_CONFIG)
    pooled = bool(HYPOTHESIS["pool_cells"])
    if pooled:  # one result for the whole experiment: its movie is the whole field, there is no cell to crop around
        if g.ALL_GROUP not in RESULT_TYPES:
            raise ValueError(f"The pooled hypothesis {HYPOTHESIS['hypothesis']} makes only the '{g.ALL_GROUP}' result type: mark it for the run.")
        RESULT_TYPES = (g.ALL_GROUP,)
    resolved_home = home_dir()
    resolved_source_home = source_home_dir()
    paths = resolve_sta_paths(resolved_source_home, resolved_home)
    fps = read_frame_rate(resolved_source_home)
    pre, post, *windows = frame_windows(HYPOTHESIS, fps)
    silence = (seconds_to_frames(HYPOTHESIS["silence_before_seconds"], fps), seconds_to_frames(HYPOTHESIS["silence_after_seconds"], fps))
    time_bin = time_bin_frames(HYPOTHESIS, fps)
    standard_result_types = tuple(result_type for result_type in RESULT_TYPES if result_type not in {"behavioral", g.ANATOMY_GROUP})
    anatomy_settings = load_anatomy_radius(ANATOMY_SPATIAL_BIN) if g.ANATOMY_GROUP in RESULT_TYPES else None
    border = run_border_margins(paths)
    used_params = sta_run_parameters(fps, pre, post, windows, standard_result_types, anatomy_settings, silence, border)
    save_sta_run_parameters(paths.sta_dir, used_params)
    cells = list_cells()
    if not cells:
        raise RuntimeError(f"No automated spike labels found in {spike_analysis_dir()}")
    if pooled:
        print(f"Pooled hypothesis: the spikes of {len(cells)} cells are tested together as one list of events")
        cells = [g.POOLED_CELL_ID]

    print(f"Running sparse STA maps for {len(cells)} {'pooled result' if pooled else 'cells'} with {N_WORKERS} workers")
    print(f"Home dir: {resolved_home}")
    print(f"Result types: {', '.join(RESULT_TYPES)}")
    print(f"STA spatial bin: {SPATIAL_BIN}")
    print(f"Hypothesis {HYPOTHESIS['hypothesis']} at {fps:g} fps: pre {pre}, post {post}, windows {windows}, silence {silence} frames (before, after), time bin {time_bin} frames")
    print(
        f"Statistic: null_standardization={NULL_STANDARDIZATION}, per_sign_null={PER_SIGN_NULL},smooth_sigma_fullres_px={SMOOTH_SIGMA_FULLRES_PX}, "
        f"min_cluster_size={MIN_CLUSTER_SIZE}, min_cluster_footprint_fullres_px={MIN_CLUSTER_FOOTPRINT_FULLRES_PX}, "
        f"min_cluster_frames={MIN_CLUSTER_FRAMES}, cluster_alpha={CLUSTER_ALPHA}, min_spikes={MIN_SPIKES_FOR_CLUSTER}"
    )
    if anatomy_settings is not None:
        print(
            "Anatomy STA spatial bin: "
            f"{ANATOMY_SPATIAL_BIN}; crop radius: {anatomy_settings['anatomy_radius_binned']} binned pixels "
            f"({anatomy_settings['anatomy_fullres_radius_px']} full-res px)"
        )
    print(f"Saving to {paths.sta_dir}")

    rows = []
    if standard_result_types:
        movie = ensure_binned_movie(paths, SPATIAL_BIN)
        cuts = run_cut_frames(paths, int(movie.shape[0]))
        centers = valid_centers(int(movie.shape[0]), pre, post, cuts)
        f0_by_pos = f0_for_valid_centers(movie, centers, windows[3])
        cfg = analysis_cfg(
            pre,
            post,
            windows,
            movie.shape[0],
            time_bin,
            border,
            result_types=standard_result_types,
            silence=silence,
        )
        cfg["result_types"] = standard_result_types
        cfg["motion_series"] = build_run_motion_series(paths, int(movie.shape[0]))
        save_global_trends(paths, movie, cfg["motion_series"], fps, cuts, border)
        init_args = (str(movie_input_path(paths, SPATIAL_BIN)), centers, f0_by_pos, str(paths.suite2p_path), None, SPATIAL_BIN, RUN_CONFIG)
        rows.extend(run_pool(process_cell, cfg, cells, paths.sta_dir, init_args, movie))
        del movie, centers, f0_by_pos, cfg, init_args  # the next result type has its own movie
        gc.collect()

    if g.ANATOMY_GROUP in RESULT_TYPES:
        anatomy_movie = ensure_binned_movie(paths, ANATOMY_SPATIAL_BIN)
        anatomy_centers = valid_centers(int(anatomy_movie.shape[0]), pre, post, run_cut_frames(paths, int(anatomy_movie.shape[0])))
        anatomy_f0 = f0_for_valid_centers(anatomy_movie, anatomy_centers, windows[3])
        anatomy_cfg = analysis_cfg(
            pre,
            post,
            windows,
            anatomy_movie.shape[0],
            time_bin,
            border,
            result_types=(g.ANATOMY_GROUP,),
            anatomy_settings=anatomy_settings,
            silence=silence,
        )
        if not standard_result_types:  # motion diagnostics are per cell, so run them once (with the first pool)
            anatomy_cfg["motion_series"] = build_run_motion_series(paths, int(anatomy_movie.shape[0]))
        init_args = (
            str(movie_input_path(paths, ANATOMY_SPATIAL_BIN)),
            anatomy_centers,
            anatomy_f0,
            str(paths.suite2p_path),
            anatomy_settings["anatomy_radius_binned"],
            ANATOMY_SPATIAL_BIN,
            RUN_CONFIG,
        )
        rows.extend(run_pool(process_cell, anatomy_cfg, cells, paths.sta_dir, init_args))
        del anatomy_movie, anatomy_centers, anatomy_f0, anatomy_cfg, init_args
        gc.collect()

    if "behavioral" in RESULT_TYPES:
        frame_groups = load_behavioral_frame_groups()
        source_behavioral_segments = behavior_segments_path()
        skipped_groups = []
        for behavioral_group in BEHAVIORAL_GROUPS:
            source_frames = frame_groups[behavioral_group]
            if source_frames.size == 0:
                reason = f"Skipped behavioral_{behavioral_group}: no {behavioral_group} frames are available."
                print(f"WARNING: {reason}")
                skipped_groups.append(
                    {
                        "behavioral_group": behavioral_group,
                        "source_frame_count": int(source_frames.size),
                        "reason": reason,
                    }
                )
                rows.extend(skipped_behavioral_rows(cells, behavioral_group, source_frames, reason))
                continue
            behavioral_movie = ensure_behavioral_binned_movie(paths, behavioral_group, source_frames, SPATIAL_BIN)
            behavioral_centers = valid_centers(int(behavioral_movie.shape[0]), pre, post)
            if behavioral_centers.size == 0:
                reason = (
                    f"Behavioral {behavioral_group} movie has {behavioral_movie.shape[0]} frames, "
                    f"which is too short for pre={pre} and post={post}."
                )
                print(f"WARNING: Skipped behavioral_{behavioral_group}: {reason}")
                skipped_groups.append(
                    {
                        "behavioral_group": behavioral_group,
                        "source_frame_count": int(source_frames.size),
                        "reason": reason,
                    }
                )
                rows.extend(skipped_behavioral_rows(cells, behavioral_group, source_frames, reason))
                continue
            behavioral_f0 = f0_for_valid_centers(behavioral_movie, behavioral_centers, windows[3])
            behavioral_cfg = analysis_cfg(
                pre,
                post,
                windows,
                behavioral_movie.shape[0],
                time_bin,
                border,
                silence=silence,
                behavioral_group=behavioral_group,
                source_frames=frame_groups[behavioral_group],
                source_behavioral_segments=str(source_behavioral_segments),
            )
            init_args = (
                str(behavioral_binned_movie_path(paths, behavioral_group, SPATIAL_BIN)),
                behavioral_centers,
                behavioral_f0,
                str(paths.suite2p_path),
                None,
                SPATIAL_BIN,
                RUN_CONFIG,
            )
            rows.extend(run_pool(process_cell_behavioral, behavioral_cfg, cells, paths.sta_dir, init_args, behavioral_movie))
            del behavioral_movie, behavioral_centers, behavioral_f0, behavioral_cfg, init_args
            gc.collect()
        if skipped_groups:
            skipped_path = path_for(PathKind.STA_SKIPPED_BEHAVIORAL_GROUPS, paths.sta_dir)
            write_json(skipped_path, skipped_groups)
            print(f"Behavioral subgroup skip report saved to {skipped_path}")

    summary_path = Path(SUMMARY_CSV) if SUMMARY_CSV else path_for(PathKind.STA_SUMMARY_CSV, paths.sta_dir)
    if not summary_path.is_absolute():
        summary_path = resolved_home / summary_path
    # the motion test runs once per cell (with the first pool); every row of the cell, whichever pool made it, carries that one answer
    motion_of_cell = {row["cell_id"]: {key: row[key] for key in (g.MOTION_STATUS_COLUMN, "motion_p") if key in row}
                      for row in rows if row.get(g.MOTION_STATUS_COLUMN, g.MOTION_NOT_TESTED) != g.MOTION_NOT_TESTED}
    for row in rows:
        row.update(motion_of_cell.get(row["cell_id"], {g.MOTION_STATUS_COLUMN: g.MOTION_NOT_TESTED}))
    write_summary(rows, summary_path)
    print(f"Done. Summary saved to {summary_path}")

    if not HYPOTHESIS["pool_cells"]:  # the cluster map and the cluster traces panels read this file (a pooled hypothesis has one result for the whole experiment, no map of cells)
        print("Saving the cluster map summary for the experiment page ...", flush=True)
        try:
            print(f"Cluster map summary saved to {save_cluster_map_summary(paths.sta_dir, resolved_source_home, summary_csv=summary_path, log=lambda m: print(m, flush=True))}")
        except (OSError, ValueError, KeyError) as exc:  # all STA results are already saved: a problem here must not fail the run
            print(f"WARNING: the cluster map summary was not written ({exc}). The cluster map and cluster traces panels need it: rerun STA Maps.", flush=True)


if __name__ == "__main__":
    main()
