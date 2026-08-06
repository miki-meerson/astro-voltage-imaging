from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import find_objects, generate_binary_structure, label
from scipy.sparse import csr_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sta_cluster_permutation import compute_effect_movie, filter_valid_peaks
from utils.clusters_utils import _merge_similar_clusters
from utils.movie_utils import get_movie_parameters, read_astro_movie, spatial_bin_movie
from utils.paths_utils import resolve_sta_paths
from utils.suite2p_utils import load_suite2p_context, neuron_centroid_binned

HOME_DIR = Path(r"Z:\Adam-Lab-Shared\Data\Miki_Meerson\Imaging\HYP-JEDI\Hyp3\L\02-02-2026-awake\FOV1\z-18-x2")
LINUX_SHARED_ROOT = Path("/ems/elsc-labs/adam-y/Adam-Lab-Shared")
WINDOWS_SHARED_ROOT = Path(r"Z:\Adam-Lab-Shared")
SPIKE_ANALYSIS_FOLDERS = ("automated_spike_analysis", "automated-spike-analysis")
LABEL_FILE = "simple_complex_spike_labels.npz"
BEHAVIORAL_SEGMENTS_FILE = "behavioral_segements.npz"

PRE_SECONDS = 2.0
POST_SECONDS = 3.0
BASELINE_SECONDS = 1.0
ANALYSIS_HALF_WINDOW_SECONDS = 1.0
ANALYSIS_HALF_WINDOW_FRAMES: int | None = None

SPATIAL_BIN = 8
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

SAVE_BINNED_STA_MOVIES = True
OVERWRITE = True
CELL_IDS: list[int] | str = "all"
RESULT_TYPES = ("all", "simple", "complex")
BEHAVIORAL_GROUPS = ("active", "inactive")
VALID_RESULT_TYPES = {"all", "simple", "complex", "anatomy", "behavioral"}
SUMMARY_CSV: str | None = None

ANATOMY_CROP_PARAMS = "anatomy_crop_params.npz"

_MOVIE: np.ndarray | None = None
_MOVIE_FLAT: np.ndarray | None = None
_MOVIE_SHAPE: tuple[int, int, int] | None = None
_VALID_CENTERS: np.ndarray | None = None
_F0_BY_POS: np.ndarray | None = None
_SUITE2P_CTX: dict | None = None
_ANATOMY_RADIUS: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse STA map analysis from automated spike labels.")
    parser.add_argument(
        "config",
        nargs="?",
        type=str,
        default=None,
        help="Path to sta_maps_job_params.json. Defaults to cluster-code/sta_maps_job_params.json when present.",
    )
    return parser.parse_args()


def load_config(path_arg: str | None) -> dict:
    config_path = None if path_arg in (None, "", "None") else Path(path_arg)
    if config_path is None:
        default_path = SCRIPT_DIR / "sta_maps_job_params.json"
        config_path = default_path if default_path.exists() else None
    if config_path is None:
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"Loaded settings from {config_path}")
    return config


def apply_config(config: dict) -> None:
    global HOME_DIR, SPATIAL_BIN, N_PERMUTATIONS, N_WORKERS, VOXEL_ALPHA, CLUSTER_ALPHA
    global CLUSTER_CONNECTIVITY, MIN_CLUSTER_SIZE, MIN_CLUSTER_MASS, MERGE_SIMILAR_CLUSTERS
    global MERGE_MAX_DISTANCE_PX, MERGE_MIN_TRACE_CORR, RNG_SEED, SAVE_BINNED_STA_MOVIES
    global OVERWRITE, CELL_IDS, RESULT_TYPES, SUMMARY_CSV, PRE_SECONDS, POST_SECONDS
    global BASELINE_SECONDS, ANALYSIS_HALF_WINDOW_SECONDS, ANALYSIS_HALF_WINDOW_FRAMES

    if "home_dir" in config:
        HOME_DIR = Path(config["home_dir"])
    CELL_IDS = config.get("cell_ids", CELL_IDS)
    SPATIAL_BIN = int(config.get("spatial_bin", SPATIAL_BIN))
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
    SAVE_BINNED_STA_MOVIES = bool(config.get("save_binned_sta_movies", SAVE_BINNED_STA_MOVIES))
    OVERWRITE = bool(config.get("overwrite", OVERWRITE))
    RESULT_TYPES = config.get("result_types", RESULT_TYPES)
    RESULT_TYPES = (RESULT_TYPES,) if isinstance(RESULT_TYPES, str) else tuple(RESULT_TYPES)
    unknown = sorted(set(RESULT_TYPES) - VALID_RESULT_TYPES)
    if unknown:
        raise ValueError(f"Unknown result type(s): {unknown}")
    SUMMARY_CSV = config.get("summary_csv", SUMMARY_CSV)
    PRE_SECONDS = float(config.get("pre_seconds", PRE_SECONDS))
    POST_SECONDS = float(config.get("post_seconds", POST_SECONDS))
    BASELINE_SECONDS = float(config.get("baseline_seconds", BASELINE_SECONDS))
    ANALYSIS_HALF_WINDOW_SECONDS = float(config.get("analysis_half_window_seconds", ANALYSIS_HALF_WINDOW_SECONDS))
    if "analysis_half_window" in config:
        ANALYSIS_HALF_WINDOW_FRAMES = int(config["analysis_half_window"])
    if "analysis_half_window_frames" in config:
        ANALYSIS_HALF_WINDOW_FRAMES = int(config["analysis_half_window_frames"])


def home_dir() -> Path:
    if HOME_DIR.exists():
        return HOME_DIR
    raw = str(HOME_DIR)
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
    return HOME_DIR


def spike_analysis_dir() -> Path:
    for name in SPIKE_ANALYSIS_FOLDERS:
        path = home_dir() / name
        if path.exists():
            return path
    return home_dir() / SPIKE_ANALYSIS_FOLDERS[0]


def binned_movie_path(paths) -> Path:
    return paths.pb_dir / f"astro_photobleaching_corrected_gevi_flipped_bin{SPATIAL_BIN}.npy"


def behavioral_binned_movie_path(paths, group: str) -> Path:
    return paths.sta_dir / f"astro_photobleaching_corrected_gevi_flipped_behavioral_{group}_bin{SPATIAL_BIN}.npy"


def anatomy_crop_params_path(paths) -> Path:
    return paths.sta_dir / ANATOMY_CROP_PARAMS


def load_anatomy_radius(paths) -> int:
    path = anatomy_crop_params_path(paths)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the app Anatomy Crop step before requesting result_type='anatomy'."
        )
    with np.load(path, allow_pickle=True) as d:
        return int(np.asarray(d["anatomy_radius_binned"]).item())


def crop_bounds(center_yx: tuple[int, int], radius: int, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    y, x = center_yx
    height, width = image_shape
    return (
        max(0, int(y) - int(radius)),
        min(height, int(y) + int(radius) + 1),
        max(0, int(x) - int(radius)),
        min(width, int(x) + int(radius) + 1),
    )


def ensure_binned_movie(paths) -> np.memmap:
    cache_path = binned_movie_path(paths)
    if cache_path.exists():
        movie = np.load(cache_path, mmap_mode="r")
        frame_means = np.asarray(movie.mean(axis=(1, 2)), dtype=np.float32)
        zero_frames = int(np.sum(frame_means == 0))
        if zero_frames == 0:
            print(f"Using binned astro movie: {cache_path}")
            return movie
        print(f"Rebuilding invalid binned astro movie cache: {zero_frames}/{movie.shape[0]} frames are zero.")
        del movie

    print(f"Building binned astro movie once: {cache_path}")
    astro = read_astro_movie(str(paths.astro_movie_path), as_memmap=True)
    sample = spatial_bin_movie(np.asarray(astro[:1], dtype=np.float32), SPATIAL_BIN)
    tmp_path = cache_path.with_suffix(".tmp.npy")
    out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(astro.shape[0], *sample.shape[1:]))
    for start in range(0, astro.shape[0], 512):
        stop = min(start + 512, astro.shape[0])
        out[start:stop] = spatial_bin_movie(np.asarray(astro[start:stop], dtype=np.float32), SPATIAL_BIN)
        print(f"  cached frames {start}-{stop}")
    out.flush()
    del out
    tmp_path.replace(cache_path)
    return np.load(cache_path, mmap_mode="r")


def load_behavioral_frame_groups(paths) -> dict[str, np.ndarray]:
    path = paths.sta_dir / BEHAVIORAL_SEGMENTS_FILE
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


def ensure_behavioral_binned_movie(paths, group: str, source_frames: np.ndarray) -> np.memmap:
    cache_path = behavioral_binned_movie_path(paths, group)
    source_frames = np.asarray(source_frames, dtype=np.int32)
    if cache_path.exists():
        movie = np.load(cache_path, mmap_mode="r")
        if movie.shape[0] == source_frames.size:
            print(f"Using behavioral {group} binned astro movie: {cache_path}")
            return movie
        print(
            f"Rebuilding behavioral {group} movie cache: "
            f"{cache_path} has {movie.shape[0]} frames, expected {source_frames.size}."
        )
        del movie

    full_movie = ensure_binned_movie(paths)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp.npy")
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
    tmp_path.replace(cache_path)
    return np.load(cache_path, mmap_mode="r")


def frame_settings(fps: int):
    pre = int(round(PRE_SECONDS * fps))
    post = int(round(POST_SECONDS * fps))
    baseline = min(int(round(BASELINE_SECONDS * fps)), pre)
    half = ANALYSIS_HALF_WINDOW_FRAMES
    if half is None:
        half = int(round(ANALYSIS_HALF_WINDOW_SECONDS * fps))
    pre_window = (-half, -1)
    post_window = (0, half)
    analysis_window = (-half, half)
    baseline_window = (-pre, -(half + 1)) if pre > half + 2 else (-pre, -max(2, pre // 2))
    return pre, post, baseline, pre_window, post_window, analysis_window, baseline_window


def relative_indices(window: tuple[int, int], pre: int, post: int) -> np.ndarray:
    rel = np.arange(-pre, post + 1)
    return np.flatnonzero((rel >= window[0]) & (rel <= window[1]))


def valid_centers(n_frames: int, pre: int, post: int) -> np.ndarray:
    return np.arange(pre + 1, n_frames - post, dtype=np.int32)


def f0_for_valid_centers(movie: np.ndarray, centers: np.ndarray, pre: int, baseline: int) -> np.ndarray:
    intensity = np.asarray(movie.mean(axis=(1, 2)), dtype=np.float32)
    windows = sliding_window_view(intensity, baseline)
    f0 = np.median(windows[centers - pre], axis=1).astype(np.float32)
    return np.maximum(np.abs(f0), np.float32(1e-12))


def crop_f0_for_valid_centers(crop: tuple[int, int, int, int], centers: np.ndarray, pre: int, baseline: int) -> np.ndarray:
    if _MOVIE is None:
        raise RuntimeError("Worker was not initialized.")
    y0, y1, x0, x1 = crop
    intensity = np.asarray(_MOVIE[:, y0:y1, x0:x1].mean(axis=(1, 2)), dtype=np.float32)
    windows = sliding_window_view(intensity, baseline)
    f0 = np.median(windows[centers - pre], axis=1).astype(np.float32)
    return np.maximum(np.abs(f0), np.float32(1e-12))


def init_worker(movie_path: str, centers: np.ndarray, f0_by_pos: np.ndarray, suite2p_path: str, anatomy_radius: int | None) -> None:
    global _MOVIE, _MOVIE_FLAT, _MOVIE_SHAPE, _VALID_CENTERS, _F0_BY_POS, _SUITE2P_CTX, _ANATOMY_RADIUS
    movie = np.load(movie_path, mmap_mode="r")
    _MOVIE = movie
    _MOVIE_SHAPE = tuple(int(x) for x in movie.shape)
    _MOVIE_FLAT = movie.reshape(movie.shape[0], -1)
    _VALID_CENTERS = np.asarray(centers, dtype=np.int32)
    _F0_BY_POS = np.asarray(f0_by_pos, dtype=np.float32)
    _ANATOMY_RADIUS = anatomy_radius
    _SUITE2P_CTX = load_suite2p_context(suite2p_path) if anatomy_radius is not None else None


def list_cells() -> list[int]:
    found = []
    for path in spike_analysis_dir().glob("cell_*"):
        if not (path / LABEL_FILE).exists():
            continue
        try:
            found.append(int(path.name.split("_", 1)[1]))
        except ValueError:
            pass
    found = sorted(found)
    if CELL_IDS == "all":
        return found
    wanted = {int(x) for x in CELL_IDS}
    return [cell_id for cell_id in found if cell_id in wanted]


def load_spikes(cell_id: int, n_frames: int, pre: int, post: int) -> dict[str, np.ndarray]:
    with np.load(spike_analysis_dir() / f"cell_{cell_id}" / LABEL_FILE, allow_pickle=True) as d:
        peaks = np.asarray(d["peaks_all"], dtype=np.int32)
        labels = np.char.lower(np.asarray(d["labels"]).astype("U8"))
    return {
        "all": filter_valid_peaks(peaks, n_frames, pre, post).astype(np.int32),
        "simple": filter_valid_peaks(peaks[labels == "simple"], n_frames, pre, post).astype(np.int32),
        "complex": filter_valid_peaks(peaks[labels == "complex"], n_frames, pre, post).astype(np.int32),
    }


def load_behavioral_spikes(
    cell_id: int,
    source_frames: np.ndarray,
    n_frames: int,
    pre: int,
    post: int,
) -> np.ndarray:
    source_frames = np.asarray(source_frames, dtype=np.int32)
    with np.load(spike_analysis_dir() / f"cell_{cell_id}" / LABEL_FILE, allow_pickle=True) as d:
        peaks = np.asarray(d["peaks_all"], dtype=np.int32)

    frame_pos = np.searchsorted(source_frames, peaks)
    in_bounds = frame_pos < source_frames.size
    matched = np.zeros(peaks.size, dtype=bool)
    matched[in_bounds] = source_frames[frame_pos[in_bounds]] == peaks[in_bounds]
    remapped = frame_pos[matched].astype(np.int32)
    return filter_valid_peaks(remapped, n_frames, pre, post).astype(np.int32)


def event_matrix(peaks: np.ndarray, shifts: np.ndarray | None = None, f0_by_pos: np.ndarray | None = None) -> csr_matrix:
    if _VALID_CENTERS is None or _F0_BY_POS is None:
        raise RuntimeError("Worker was not initialized.")
    f0 = _F0_BY_POS if f0_by_pos is None else np.asarray(f0_by_pos, dtype=np.float32)
    peaks = np.asarray(peaks, dtype=np.int32)
    peak_pos = np.searchsorted(_VALID_CENTERS, peaks)
    if peaks.size == 0:
        n_rows = 1 if shifts is None else len(shifts)
        return csr_matrix((n_rows, _VALID_CENTERS.size), dtype=np.float32)

    if np.any(_VALID_CENTERS[peak_pos] != peaks):
        raise ValueError("All peaks must be valid centers.")

    if shifts is None:
        rows = np.zeros(peaks.size, dtype=np.int32)
        cols = peak_pos.astype(np.int32)
        data = (1.0 / (peaks.size * f0[cols])).astype(np.float32)
        return csr_matrix((data, (rows, cols)), shape=(1, _VALID_CENTERS.size), dtype=np.float32)

    cols2d = (peak_pos[None, :] + shifts[:, None]) % _VALID_CENTERS.size
    rows = np.repeat(np.arange(len(shifts), dtype=np.int32), peaks.size)
    cols = cols2d.reshape(-1).astype(np.int32)
    data = (1.0 / (peaks.size * f0[cols])).astype(np.float32)
    return csr_matrix((data, (rows, cols)), shape=(len(shifts), _VALID_CENTERS.size), dtype=np.float32)


def sta_for_rels(events: csr_matrix, rels: np.ndarray, crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    if _MOVIE is None or _MOVIE_FLAT is None or _VALID_CENTERS is None or _MOVIE_SHAPE is None:
        raise RuntimeError("Worker was not initialized.")
    rels = np.asarray(rels, dtype=np.int32)
    if crop is None:
        height, width = _MOVIE_SHAPE[1], _MOVIE_SHAPE[2]
        n_pixels = _MOVIE_FLAT.shape[1]
    else:
        y0, y1, x0, x1 = crop
        height, width = y1 - y0, x1 - x0
        n_pixels = height * width
    out = np.empty((events.shape[0], rels.size, n_pixels), dtype=np.float32)
    for i, rel in enumerate(rels):
        frames = _VALID_CENTERS + int(rel)
        if crop is None:
            frame_data = np.asarray(_MOVIE_FLAT[frames], dtype=np.float32)
        else:
            y0, y1, x0, x1 = crop
            frame_data = np.asarray(_MOVIE[frames, y0:y1, x0:x1], dtype=np.float32).reshape(frames.size, -1)
        # Astro movies are read in GEVI-flipped polarity, so their baseline is
        # negative while the normalization denominator is abs(F0). This computes
        # (frame - F0) / abs(F0), which is frame / abs(F0) + 1 for flipped astro.
        out[:, i, :] = np.asarray(events @ frame_data, dtype=np.float32) + 1.0
    return out.reshape(events.shape[0], rels.size, height, width)


def permutation_effects(events: csr_matrix, pre: int, post: int, windows: tuple, crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    baseline_idx = relative_indices(windows[3], pre, post)
    analysis_idx = relative_indices(windows[2], pre, post)
    all_rels = np.arange(-pre, post + 1, dtype=np.int32)
    baseline_rels = all_rels[baseline_idx]
    analysis_rels = all_rels[analysis_idx]

    baseline_map = sta_for_rels(events, baseline_rels, crop).mean(axis=1)
    effects = np.empty((events.shape[0], analysis_rels.size, *baseline_map.shape[1:]), dtype=np.float32)
    for i, rel in enumerate(analysis_rels):
        effects[:, i] = sta_for_rels(events, np.asarray([rel], dtype=np.int32), crop)[:, 0] - baseline_map
    return effects


def cluster_records(effect: np.ndarray, threshold: float) -> list[dict]:
    structure = generate_binary_structure(3, CLUSTER_CONNECTIVITY)
    records = []
    for sign, mask in (("positive", effect >= threshold), ("negative", effect <= -threshold)):
        labels, _ = label(mask, structure=structure)
        for label_id, slc in enumerate(find_objects(labels), start=1):
            if slc is None:
                continue
            local = labels[slc] == label_id
            size = int(local.sum())
            if size < MIN_CLUSTER_SIZE:
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


def summarize_clusters(effect: np.ndarray, threshold: float, null_max: np.ndarray) -> tuple[list[dict], np.ndarray]:
    records = sorted(cluster_records(effect, threshold), key=lambda c: c["mass"], reverse=True)
    sig_mask = np.zeros_like(effect, dtype=bool)
    summary = []
    for cluster_id, record in enumerate(records, start=1):
        p_value = float((1 + np.sum(null_max >= record["mass"])) / (len(null_max) + 1))
        is_sig = bool(p_value <= CLUSTER_ALPHA)
        if is_sig:
            sig_mask |= record["mask"]
        summary.append({
            "cluster_id": cluster_id,
            "sign": record["sign"],
            "mass": float(record["mass"]),
            "size": int(record["size"]),
            "peak_value": float(record["peak_value"]),
            "p_value": p_value,
            "is_significant": is_sig,
        })
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
        permutation_shifts=np.asarray(result["permutation_shifts"], dtype=np.int32),
        pre_frames=np.asarray(result["pre_frames"], dtype=np.int32),
        post_frames=np.asarray(result["post_frames"], dtype=np.int32),
        baseline_frames=np.asarray(result["baseline_frames"], dtype=np.int32),
        pre_window=np.asarray(result["pre_window"], dtype=np.int32),
        post_window=np.asarray(result["post_window"], dtype=np.int32),
        analysis_window=np.asarray(result["analysis_window"], dtype=np.int32),
        baseline_window=np.asarray(result["baseline_window"], dtype=np.int32),
        spatial_bin=np.asarray(SPATIAL_BIN, dtype=np.int32),
        voxel_alpha=np.asarray(VOXEL_ALPHA, dtype=np.float32),
        cluster_alpha=np.asarray(CLUSTER_ALPHA, dtype=np.float32),
        cluster_connectivity=np.asarray(CLUSTER_CONNECTIVITY, dtype=np.int32),
        min_cluster_size=np.asarray(MIN_CLUSTER_SIZE, dtype=np.int32),
        min_cluster_mass=np.asarray(MIN_CLUSTER_MASS, dtype=np.float32),
        merge_similar_clusters=np.asarray(MERGE_SIMILAR_CLUSTERS),
        merge_max_distance_px=np.asarray(MERGE_MAX_DISTANCE_PX, dtype=np.float32),
        merge_min_trace_corr=np.asarray(MERGE_MIN_TRACE_CORR, dtype=np.float32),
    )
    for key in ("anatomy_radius_binned", "anatomy_center_yx_binned", "anatomy_crop_bounds"):
        if key in result:
            payload[key] = np.asarray(result[key], dtype=np.int32)
    for key in ("behavioral_group", "source_behavioral_segments"):
        if key in result:
            payload[key] = np.asarray(result[key])
    np.savez(path, **payload)


def analyze_group(
    cell_id: int,
    group: str,
    peaks: np.ndarray,
    shifts: np.ndarray,
    pre: int,
    post: int,
    baseline: int,
    windows: tuple,
    out_dir: Path,
    crop: tuple[int, int, int, int] | None = None,
    f0_by_pos: np.ndarray | None = None,
    extra_result: dict | None = None,
) -> dict:
    print(f"cell {cell_id} {group}: {peaks.size} peaks")
    observed_events = event_matrix(peaks, f0_by_pos=f0_by_pos)
    rels = np.arange(-pre, post + 1, dtype=np.int32)
    observed_sta = sta_for_rels(observed_events, rels, crop)[0]
    sta_name = "astro_sta_movie" if group == "all" else f"astro_sta_movie_{group}"
    if SAVE_BINNED_STA_MOVIES:
        np.save(out_dir / f"{sta_name}_bin{SPATIAL_BIN}.npy", observed_sta.astype(np.float32))

    clusters = []
    npz_name = "astro_sta_cluster_permutation.npz" if group == "all" else f"astro_sta_cluster_permutation_{group}.npz"
    cluster_path = out_dir / npz_name
    if peaks.size >= MIN_SPIKES_FOR_CLUSTER:
        observed_effect = compute_effect_movie(observed_sta, pre, post, *windows)
        perm_events = event_matrix(peaks, shifts, f0_by_pos=f0_by_pos)
        perm_effects = permutation_effects(perm_events, pre, post, windows, crop)
        threshold = float(np.quantile(np.abs(perm_effects), 1.0 - VOXEL_ALPHA))
        null_max = np.asarray([max((c["mass"] for c in cluster_records(effect, threshold)), default=0.0) for effect in perm_effects], dtype=np.float32)
        clusters, sig_mask = summarize_clusters(observed_effect["effect_movie"], threshold, null_max)
        result = {
            "cell_id": cell_id,
            "spike_group": group,
            "peaks": peaks,
            "observed_sta_movie": observed_sta,
            "observed_effect_movie": observed_effect["effect_movie"],
            "observed_post_pre_map": observed_effect["post_pre_map"],
            "significant_cluster_mask": sig_mask,
            "perm_effects": perm_effects,
            "null_max_cluster_mass": null_max,
            "cluster_summary": clusters,
            "cluster_threshold": threshold,
            "permutation_shifts": shifts,
            "pre_frames": pre,
            "post_frames": post,
            "baseline_frames": baseline,
            "pre_window": windows[0],
            "post_window": windows[1],
            "analysis_window": windows[2],
            "baseline_window": windows[3],
        }
        if extra_result:
            result.update(extra_result)
        if OVERWRITE or not cluster_path.exists():
            save_cluster_result(cluster_path, result)
    elif OVERWRITE and cluster_path.exists():
        cluster_path.unlink()
        cluster_path = ""
    else:
        cluster_path = ""

    return {
        "cell_id": cell_id,
        "spike_group": group,
        "n_peaks": int(peaks.size),
        "n_clusters": int(len(clusters)),
        "n_significant_clusters": int(sum(c["is_significant"] for c in clusters)),
        "max_cluster_mass": float(max((c["mass"] for c in clusters), default=0.0)),
        "cluster_npz_path": str(cluster_path),
    }


def anatomy_analysis_kwargs(cell_id: int, pre: int, baseline: int) -> dict:
    if _MOVIE_SHAPE is None or _VALID_CENTERS is None or _SUITE2P_CTX is None or _ANATOMY_RADIUS is None:
        raise RuntimeError("Anatomy result requested but anatomy crop settings were not initialized.")
    center = neuron_centroid_binned(cell_id, SPATIAL_BIN, _MOVIE_SHAPE[1:], _SUITE2P_CTX)
    crop = crop_bounds(center, _ANATOMY_RADIUS, _MOVIE_SHAPE[1:])
    return {
        "crop": crop,
        "f0_by_pos": crop_f0_for_valid_centers(crop, _VALID_CENTERS, pre, baseline),
        "extra_result": {
            "anatomy_radius_binned": _ANATOMY_RADIUS,
            "anatomy_center_yx_binned": center,
            "anatomy_crop_bounds": crop,
        },
    }


def analysis_cfg(pre: int, post: int, baseline: int, windows: tuple, n_frames: int, centers: np.ndarray, **extra) -> dict:
    return {
        "pre": pre,
        "post": post,
        "baseline": baseline,
        "windows": tuple(windows),
        "n_frames": int(n_frames),
        "n_valid_centers": int(len(centers)),
        **extra,
    }


def cell_output_dir(sta_dir_s: str, cell_id: int) -> Path:
    out_dir = Path(sta_dir_s) / f"cell_{cell_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def cell_shifts(cell_id: int, n_valid_centers: int) -> np.ndarray:
    return np.random.default_rng(RNG_SEED + int(cell_id)).integers(
        0,
        int(n_valid_centers),
        size=N_PERMUTATIONS,
        dtype=np.int32,
    )


def save_neuron_sta_metadata(out_dir: Path, cell_id: int, pre: int, post: int, baseline: int, suffix: str = "", **extra) -> None:
    payload = {
        "cell_id": np.asarray(cell_id, dtype=np.int32),
        "pre_frames": np.asarray(pre, dtype=np.int32),
        "post_frames": np.asarray(post, dtype=np.int32),
        "baseline_frames": np.asarray(baseline, dtype=np.int32),
        "source_spike_labels": str(spike_analysis_dir() / f"cell_{cell_id}" / LABEL_FILE),
        **extra,
    }
    np.savez(out_dir / f"neuron_sta{suffix}.npz", **payload)


def run_pool(worker_fn, cfg: dict, cells: list[int], sta_dir: Path, init_args: tuple) -> list[dict]:
    rows = []
    with mp.Pool(N_WORKERS, initializer=init_worker, initargs=init_args) as pool:
        for cell_rows in pool.imap_unordered(worker_fn, [(cell_id, cfg, str(sta_dir)) for cell_id in cells]):
            rows.extend(cell_rows)
    return rows


def process_cell(args: tuple[int, dict, str]) -> list[dict]:
    cell_id, cfg, sta_dir_s = args
    pre, post, baseline = cfg["pre"], cfg["post"], cfg["baseline"]
    windows = cfg["windows"]
    out_dir = cell_output_dir(sta_dir_s, cell_id)

    spike_groups = load_spikes(cell_id, cfg["n_frames"], pre, post)
    shifts = cell_shifts(cell_id, cfg["n_valid_centers"])
    save_neuron_sta_metadata(out_dir, cell_id, pre, post, baseline)

    rows = []
    for result_type in cfg.get("result_types", RESULT_TYPES):
        peaks = spike_groups["all"] if result_type == "anatomy" else spike_groups[result_type]
        kwargs = anatomy_analysis_kwargs(cell_id, pre, baseline) if result_type == "anatomy" else {}
        rows.append(
            analyze_group(
                cell_id,
                result_type,
                peaks,
                shifts,
                pre,
                post,
                baseline,
                windows,
                out_dir,
                **kwargs,
            )
        )
    return rows


def process_cell_behavioral(args: tuple[int, dict, str]) -> list[dict]:
    cell_id, cfg, sta_dir_s = args
    pre, post, baseline = cfg["pre"], cfg["post"], cfg["baseline"]
    windows = cfg["windows"]
    behavioral_group = cfg["behavioral_group"]
    source_frames = np.asarray(cfg["source_frames"], dtype=np.int32)
    out_dir = cell_output_dir(sta_dir_s, cell_id)

    peaks = load_behavioral_spikes(cell_id, source_frames, cfg["n_frames"], pre, post)
    shifts = cell_shifts(cell_id, cfg["n_valid_centers"])
    group = f"behavioral_{behavioral_group}"
    save_neuron_sta_metadata(
        out_dir,
        cell_id,
        pre,
        post,
        baseline,
        suffix=f"_{group}",
        source_behavioral_segments=str(cfg["source_behavioral_segments"]),
        behavioral_group=np.asarray(behavioral_group),
    )

    return [
        analyze_group(
            cell_id,
            group,
            peaks,
            shifts,
            pre,
            post,
            baseline,
            windows,
            out_dir,
            extra_result={
                "behavioral_group": np.asarray(behavioral_group),
                "source_behavioral_segments": np.asarray(str(cfg["source_behavioral_segments"])),
            },
        )
    ]


def skipped_behavioral_rows(cells: list[int], behavioral_group: str, source_frames: np.ndarray, reason: str) -> list[dict]:
    group = f"behavioral_{behavioral_group}"
    return [
        {
            "cell_id": int(cell_id),
            "spike_group": group,
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
        "n_peaks",
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
    apply_config(load_config(parse_args().config))
    resolved_home = home_dir()
    paths = resolve_sta_paths(resolved_home)
    fps = int(get_movie_parameters(f"{resolved_home.as_posix()}/", is_dir=True)["fps"])
    pre, post, baseline, *windows = frame_settings(fps)
    movie = ensure_binned_movie(paths)
    centers = valid_centers(int(movie.shape[0]), pre, post)
    f0_by_pos = f0_for_valid_centers(movie, centers, pre, baseline)
    anatomy_radius = load_anatomy_radius(paths) if "anatomy" in RESULT_TYPES else None
    cells = list_cells()
    if not cells:
        raise RuntimeError(f"No automated spike labels found in {spike_analysis_dir()}")

    normal_result_types = tuple(result_type for result_type in RESULT_TYPES if result_type != "behavioral")
    cfg = analysis_cfg(
        pre,
        post,
        baseline,
        windows,
        movie.shape[0],
        centers,
        result_types=normal_result_types,
    )
    print(f"Running sparse STA maps for {len(cells)} cells with {N_WORKERS} workers")
    print(f"Home dir: {resolved_home}")
    print(f"Result types: {', '.join(RESULT_TYPES)}")
    if anatomy_radius is not None:
        print(f"Anatomy crop radius: {anatomy_radius} binned pixels")
    print(f"Saving to {paths.sta_dir}")

    rows = []
    if normal_result_types:
        init_args = (str(binned_movie_path(paths)), centers, f0_by_pos, str(paths.suite2p_path), anatomy_radius)
        rows.extend(run_pool(process_cell, cfg, cells, paths.sta_dir, init_args))

    if "behavioral" in RESULT_TYPES:
        frame_groups = load_behavioral_frame_groups(paths)
        source_behavioral_segments = paths.sta_dir / BEHAVIORAL_SEGMENTS_FILE
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
            behavioral_movie = ensure_behavioral_binned_movie(paths, behavioral_group, source_frames)
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
            behavioral_f0 = f0_for_valid_centers(behavioral_movie, behavioral_centers, pre, baseline)
            behavioral_cfg = analysis_cfg(
                pre,
                post,
                baseline,
                windows,
                behavioral_movie.shape[0],
                behavioral_centers,
                behavioral_group=behavioral_group,
                source_frames=frame_groups[behavioral_group],
                source_behavioral_segments=str(source_behavioral_segments),
            )
            init_args = (
                str(behavioral_binned_movie_path(paths, behavioral_group)),
                behavioral_centers,
                behavioral_f0,
                str(paths.suite2p_path),
                None,
            )
            rows.extend(run_pool(process_cell_behavioral, behavioral_cfg, cells, paths.sta_dir, init_args))
        if skipped_groups:
            skipped_path = paths.sta_dir / "sta_maps_skipped_behavioral_groups.json"
            skipped_path.write_text(json.dumps(skipped_groups, indent=2), encoding="utf-8")
            print(f"Behavioral subgroup skip report saved to {skipped_path}")

    summary_path = Path(SUMMARY_CSV) if SUMMARY_CSV else paths.sta_dir / "sta_maps_from_automated_spikes_sparse_summary.csv"
    if not summary_path.is_absolute():
        summary_path = resolved_home / summary_path
    write_summary(rows, summary_path)
    print(f"Done. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
