from pathlib import Path
import numpy as np

from utils.movie_utils import get_movie_parameters, read_astro_movie, spatial_bin_movie, apply_keep_mask_to_movie
from utils.spikes_utils import compute_peaks
from utils.suite2p_utils import load_suite2p_context
from utils.clusters_utils import * 
from utils.db_utils import *
from utils.paths_utils import *

# =========================
# Peak And STA Construction
# =========================
def reconstruct_cell_peaks(
    cell_trace: np.ndarray,
    threshold_factor: float,
    min_distance: int,
    n_frames: int,
    pre_frames: int,
    post_frames: int,
) -> np.ndarray:
    peaks = compute_peaks(threshold_factor, min_distance, cell_trace)
    peaks = filter_valid_peaks(peaks, n_frames, pre_frames, post_frames)
    return peaks


def filter_valid_peaks(peaks: np.ndarray, n_frames: int, pre_frames: int, post_frames: int) -> np.ndarray:
    peaks = np.asarray(peaks, dtype=int)
    keep = (peaks > pre_frames) & (peaks < (n_frames - post_frames))
    peaks = peaks[keep]
    return peaks


def make_valid_centers(n_frames: int, pre_frames: int, post_frames: int) -> np.ndarray:
    valid = np.zeros(n_frames, dtype=bool)
    valid[pre_frames + 1 : n_frames - post_frames] = True
    return np.flatnonzero(valid)


def circular_shift_peaks_within_valid(peaks: np.ndarray, valid_centers: np.ndarray, shift_steps: int) -> np.ndarray:
    peaks = np.asarray(peaks, dtype=int)
    valid_centers = np.asarray(valid_centers, dtype=int)
    center_to_pos = np.full(valid_centers.max() + 1, -1, dtype=int)
    center_to_pos[valid_centers] = np.arange(valid_centers.size)
    peak_pos = center_to_pos[peaks]
    if np.any(peak_pos < 0):
        raise ValueError("All peaks must lie inside the valid-centers ring.")
    return valid_centers[(peak_pos + shift_steps) % valid_centers.size]


def compute_sta_movie_from_peaks(
    astro_movie: np.ndarray,
    peaks: np.ndarray,
    pre_frames: int,
    post_frames: int,
    baseline_frames: int,
    spatial_bin: int = 1,
) -> np.ndarray:
    peaks = np.asarray(peaks, dtype=int)
    sta_window = pre_frames + post_frames + 1
    sample = np.asarray(astro_movie[0:1], dtype=np.float32)
    sample = spatial_bin_movie(sample, spatial_bin)
    _, n_rows, n_cols = sample.shape
    sta_movie = np.zeros((sta_window, n_rows, n_cols), dtype=np.float32)

    if peaks.size == 0:
        return sta_movie

    used_peaks = 0
    for peak in peaks:
        start = peak - pre_frames
        stop = peak + post_frames + 1
        segment = np.asarray(astro_movie[start:stop], dtype=np.float32)
        if segment.shape[0] != sta_window:
            continue
        segment = spatial_bin_movie(segment, spatial_bin)
        segment_intensity = segment.mean(axis=(1, 2))
        baseline_count = min(baseline_frames, segment.shape[0], max(1, pre_frames))
        f0 = float(np.median(segment_intensity[:baseline_count]))
        segment_norm = (segment - f0) / (abs(f0) + 1e-12)
        sta_movie += segment_norm
        used_peaks += 1

    if used_peaks == 0:
        raise ValueError("No peaks survived STA window extraction.")
    return sta_movie / used_peaks


# =========================
# Effect Movie Windows
# =========================
def _relative_window_to_index(window: tuple[int, int], pre_frames: int, post_frames: int) -> np.ndarray:
    lo, hi = window
    rel = np.arange(-pre_frames, post_frames + 1)
    return np.flatnonzero((rel >= lo) & (rel <= hi))


def compute_effect_movie(
    sta_movie: np.ndarray,
    pre_frames: int,
    post_frames: int,
    pre_window: tuple[int, int] = (-20, -1),
    post_window: tuple[int, int] = (0, 20),
    analysis_window: tuple[int, int] | None = None,
    baseline_window: tuple[int, int] | None = None,
) -> dict:
    if analysis_window is None:
        analysis_window = (pre_window[0], post_window[1])
    if baseline_window is None:
        baseline_window = pre_window

    pre_idx = _relative_window_to_index(pre_window, pre_frames, post_frames)
    post_idx = _relative_window_to_index(post_window, pre_frames, post_frames)
    analysis_idx = _relative_window_to_index(analysis_window, pre_frames, post_frames)
    baseline_idx = _relative_window_to_index(baseline_window, pre_frames, post_frames)
    if baseline_idx.size == 0:
        raise ValueError("baseline_window selects no frames.")
    rel = np.arange(-pre_frames, post_frames + 1)
    baseline_map = sta_movie[baseline_idx].mean(axis=0)
    effect_movie = sta_movie[analysis_idx] - baseline_map[None]
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


# =========================
# Cluster Extraction
# =========================
def _max_cluster_mass(
    effect_movie: np.ndarray,
    threshold: float,
    connectivity: int = 2,
    min_cluster_size: int = 1,
    min_cluster_mass: float = 0.0,
    merge_similar: bool = True,
    merge_max_distance_px: float = 3.0,
    merge_min_trace_corr: float = 0.9,
) -> float:
    clusters, _ = cluster_list_from_effect(
        effect_movie=effect_movie,
        threshold=threshold,
        connectivity=connectivity,
        min_cluster_size=min_cluster_size,
        min_cluster_mass=min_cluster_mass,
        merge_similar=merge_similar,
        merge_max_distance_px=merge_max_distance_px,
        merge_min_trace_corr=merge_min_trace_corr,
    )
    if not clusters:
        return 0.0
    return float(max(cluster["mass"] for cluster in clusters))


def _summarize_clusters_and_significance(
    effect_movie: np.ndarray,
    threshold: float,
    null_max_cluster_mass: np.ndarray,
    cluster_alpha: float,
    connectivity: int = 2,
    min_cluster_size: int = 1,
    min_cluster_mass: float = 0.0,
    merge_similar: bool = True,
    merge_max_distance_px: float = 3.0,
    merge_min_trace_corr: float = 0.9,
) -> tuple[list[dict], np.ndarray]:
    clusters = extract_clusters(
        effect_movie=effect_movie,
        threshold=threshold,
        connectivity=connectivity,
        min_cluster_size=min_cluster_size,
        min_cluster_mass=min_cluster_mass,
        merge_similar=merge_similar,
        merge_max_distance_px=merge_max_distance_px,
        merge_min_trace_corr=merge_min_trace_corr,
        null_max=null_max_cluster_mass,
        cluster_alpha=cluster_alpha,
    )
    significant_cluster_mask = np.zeros_like(effect_movie, dtype=bool)
    cluster_summary = []
    for cluster in clusters:
        if cluster["is_significant"]:
            significant_cluster_mask |= cluster["mask3d"]
        cluster_summary.append(
            {
                "cluster_id": cluster["cluster_id"],
                "sign": cluster["sign"],
                "mass": cluster["mass"],
                "size": cluster["size"],
                "peak_value": cluster["peak_value"],
                "p_value": cluster["p_value"],
                "is_significant": cluster["is_significant"],
            }
        )
    return cluster_summary, significant_cluster_mask


# =========================
# Permutation Testing
# =========================
def cluster_permutation_test(
    sta_movie: np.ndarray,
    astro_movie: np.ndarray,
    peaks: np.ndarray,
    pre_frames: int,
    post_frames: int,
    baseline_frames: int,
    n_frames: int,
    n_permutations: int = 200,
    spatial_bin: int = 8,
    pre_window: tuple[int, int] = (-20, -1),
    post_window: tuple[int, int] = (0, 20),
    analysis_window: tuple[int, int] | None = None,
    baseline_window: tuple[int, int] | None = None,
    voxel_alpha: float = 0.01,
    cluster_alpha: float = 0.05,
    rng_seed: int = 0,
    cluster_connectivity: int = 2,
    min_cluster_size: int = 3,
    min_cluster_mass: float = 0.0,
    merge_similar_clusters: bool = True,
    merge_max_distance_px: float = 3.0,
    merge_min_trace_corr: float = 0.9,
) -> dict:
    rng = np.random.default_rng(rng_seed)

    sta_movie = spatial_bin_movie(sta_movie, spatial_bin)
    observed_effect = compute_effect_movie(
        sta_movie,
        pre_frames=pre_frames,
        post_frames=post_frames,
        pre_window=pre_window,
        post_window=post_window,
        analysis_window=analysis_window,
        baseline_window=baseline_window,
    )

    effect_shape = observed_effect["effect_movie"].shape
    perm_effects = np.zeros((n_permutations, *effect_shape), dtype=np.float32)

    valid_centers = make_valid_centers(
        n_frames=n_frames,
        pre_frames=pre_frames,
        post_frames=post_frames,
    )
    if valid_centers.size == 0:
        raise ValueError("No valid frame centers available for circular shifting.")

    unique_shifts = rng.integers(0, valid_centers.size, size=n_permutations)
    for perm_idx, shift_steps in enumerate(unique_shifts):
        print(f"Permutation {perm_idx+1}/{len(unique_shifts)}")
        shifted_peaks = circular_shift_peaks_within_valid(peaks, valid_centers, int(shift_steps))
        perm_sta_movie = compute_sta_movie_from_peaks(
            astro_movie=astro_movie,
            peaks=shifted_peaks,
            pre_frames=pre_frames,
            post_frames=post_frames,
            baseline_frames=baseline_frames,
            spatial_bin=spatial_bin,
        )
        perm_effects[perm_idx] = compute_effect_movie(
            perm_sta_movie,
            pre_frames=pre_frames,
            post_frames=post_frames,
            pre_window=pre_window,
            post_window=post_window,
            analysis_window=analysis_window,
            baseline_window=baseline_window,
        )["effect_movie"]

    print("Finished permuation tests. Building results...")
    abs_perm = np.abs(perm_effects)
    cluster_threshold = float(np.quantile(abs_perm, 1.0 - voxel_alpha))

    obs_effect_movie = observed_effect["effect_movie"]

    null_max_cluster_mass = np.zeros(n_permutations, dtype=np.float32)
    for perm_idx in range(n_permutations):
        null_max_cluster_mass[perm_idx] = _max_cluster_mass(
            effect_movie=perm_effects[perm_idx],
            threshold=cluster_threshold,
            connectivity=cluster_connectivity,
            min_cluster_size=min_cluster_size,
            min_cluster_mass=min_cluster_mass,
            merge_similar=merge_similar_clusters,
            merge_max_distance_px=merge_max_distance_px,
            merge_min_trace_corr=merge_min_trace_corr,
        )

    cluster_summary, significant_cluster_mask = _summarize_clusters_and_significance(
        effect_movie=obs_effect_movie,
        threshold=cluster_threshold,
        null_max_cluster_mass=null_max_cluster_mass,
        cluster_alpha=cluster_alpha,
        connectivity=cluster_connectivity,
        min_cluster_size=min_cluster_size,
        min_cluster_mass=min_cluster_mass,
        merge_similar=merge_similar_clusters,
        merge_max_distance_px=merge_max_distance_px,
        merge_min_trace_corr=merge_min_trace_corr,
    )

    return {
        "observed_sta_movie": sta_movie,
        "observed_effect_movie": obs_effect_movie,
        "observed_post_pre_map": observed_effect["post_pre_map"],
        "analysis_relative_frames": observed_effect["analysis_relative_frames"],
        "cluster_threshold": cluster_threshold,
        "significant_cluster_mask": significant_cluster_mask,
        "cluster_summary": cluster_summary,
        "null_max_cluster_mass": null_max_cluster_mass,
        "perm_effects": perm_effects,
        "permutation_shifts": unique_shifts,
        "spatial_bin": int(spatial_bin),
        "pre_window": np.asarray(pre_window, dtype=int),
        "post_window": np.asarray(post_window, dtype=int),
        "analysis_window": np.asarray(
            analysis_window if analysis_window is not None else (pre_window[0], post_window[1]),
            dtype=int,
        ),
        "baseline_window": np.asarray(
            baseline_window if baseline_window is not None else pre_window,
            dtype=int,
        ),
        "voxel_alpha": float(voxel_alpha),
        "cluster_alpha": float(cluster_alpha),
        "cluster_connectivity": int(cluster_connectivity),
        "min_cluster_size": int(min_cluster_size),
        "min_cluster_mass": float(min_cluster_mass),
        "merge_similar_clusters": bool(merge_similar_clusters),
        "merge_max_distance_px": float(merge_max_distance_px),
        "merge_min_trace_corr": float(merge_min_trace_corr),
    }


# =========================
# Cell-Level Analysis Prep
# =========================
def prepare_cell_analysis_inputs(home_dir: str | Path, cell_id: int, as_memmap: bool = True) -> dict:
    paths = resolve_sta_paths(home_dir)
    params = get_movie_parameters(f"{paths.home_dir.as_posix()}/", is_dir=True)
    suite2p = load_suite2p_context(paths.suite2p_path)
    cell_dir = paths.sta_dir / f"cell_{cell_id}"
    meta = load_sta_metadata(cell_dir)
    astro_movie = read_astro_movie(str(paths.astro_movie_path), as_memmap=as_memmap)
    cell_trace = suite2p["F_norm"][cell_id]
    peaks = reconstruct_cell_peaks(
        cell_trace=cell_trace,
        threshold_factor=meta["threshold_factor"],
        min_distance=meta["min_distance"],
        n_frames=astro_movie.shape[0],
        pre_frames=meta["pre_frames"],
        post_frames=meta["post_frames"],
    )
    observed_sta_movie = load_sta_movie(cell_dir)
    return {
        "paths": paths,
        "cell_dir": cell_dir,
        "cell_trace": cell_trace,
        "peaks": peaks,
        "observed_sta_movie": observed_sta_movie,
        "astro_movie": astro_movie,
        "fps": int(params["fps"]),
        "pre_frames": meta["pre_frames"],
        "post_frames": meta["post_frames"],
        "baseline_frames": min(int(params["fps"]), meta["pre_frames"]),
        "n_frames": int(astro_movie.shape[0]),
    }


# =========================
# Run Wrappers
# =========================
def run_cell_cluster_permutation_analysis(home_dir: str | Path, cell_id: int, as_memmap: bool = True, **kwargs) -> dict:
    inputs = prepare_cell_analysis_inputs(home_dir=home_dir, cell_id=cell_id, as_memmap=as_memmap)
    result = _run_cell_cluster_permutation_analysis_from_inputs(
        inputs=inputs,
        cell_id=cell_id,
        sta_movie=inputs["observed_sta_movie"],
        astro_movie=inputs["astro_movie"],
        **kwargs,
    )
    return result


def run_cell_cluster_permutation_analysis_masked(home_dir, cell_id, keep_mask=None, as_memmap=False, **kwargs):
    inp = prepare_cell_analysis_inputs(home_dir=home_dir, cell_id=cell_id, as_memmap=as_memmap)

    sta_movie = apply_keep_mask_to_movie(inp["observed_sta_movie"], keep_mask)
    astro_movie_local = apply_keep_mask_to_movie(inp["astro_movie"], keep_mask)

    res = _run_cell_cluster_permutation_analysis_from_inputs(
        inputs=inp,
        cell_id=cell_id,
        sta_movie=sta_movie,
        astro_movie=astro_movie_local,
        **kwargs,
    )
    res["used_blood_vessel_mask"] = bool(keep_mask is not None)
    return res


def _run_cell_cluster_permutation_analysis_from_inputs(
    inputs: dict,
    cell_id: int,
    sta_movie: np.ndarray,
    astro_movie: np.ndarray,
    **kwargs,
) -> dict:
    result = cluster_permutation_test(
        sta_movie=sta_movie,
        astro_movie=astro_movie,
        peaks=inputs["peaks"],
        pre_frames=inputs["pre_frames"],
        post_frames=inputs["post_frames"],
        baseline_frames=inputs["baseline_frames"],
        n_frames=inputs["n_frames"],
        **kwargs,
    )
    print("Updating results...")
    result.update(
        {
            "cell_id": int(cell_id),
            "peaks": inputs["peaks"],
            "fps": int(inputs["fps"]),
            "pre_frames": int(inputs["pre_frames"]),
            "post_frames": int(inputs["post_frames"]),
            "baseline_frames": int(inputs["baseline_frames"]),
            "cell_dir": str(inputs["cell_dir"]),
        }
    )
    return result


# =========================
# Saving Results
# =========================
def _make_non_overwriting_path(save_path: str | Path) -> Path:
    p = Path(save_path)
    if not p.exists():
        return p
    stem = p.stem
    suffix = p.suffix
    parent = p.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_v{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def save_cluster_permutation_result(
    result: dict,
    save_path: str | Path,
    *,
    overwrite: bool = False,
    auto_version_if_exists: bool = True,
) -> Path:
    save_path = Path(save_path)
    if save_path.exists() and not overwrite and not auto_version_if_exists:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {save_path}. "
            "Use overwrite=True or auto_version_if_exists=True."
        )
    if save_path.exists() and auto_version_if_exists:
        save_path = _make_non_overwriting_path(save_path)

    cluster_array = np.asarray(
        [
            (
                c["cluster_id"],
                c["sign"],
                c["mass"],
                c["size"],
                c["peak_value"],
                c["p_value"],
                c["is_significant"],
            )
            for c in result["cluster_summary"]
        ],
        dtype=object,
    )
    np.savez_compressed(
        save_path,
        cell_id=result["cell_id"],
        observed_effect_movie=np.asarray(result["observed_effect_movie"], dtype=np.float32),
        observed_post_pre_map=np.asarray(result["observed_post_pre_map"], dtype=np.float32),
        null_max_cluster_mass=np.asarray(result["null_max_cluster_mass"], dtype=np.float32),
        cluster_summary=cluster_array,
        cluster_threshold=np.asarray(result["cluster_threshold"], dtype=np.float32),
        pre_window=np.asarray(result["pre_window"], dtype=int),
        post_window=np.asarray(result["post_window"], dtype=int),
        analysis_window=np.asarray(result["analysis_window"], dtype=int),
        baseline_window=np.asarray(result["baseline_window"], dtype=int),
        spatial_bin=np.asarray(result["spatial_bin"], dtype=int),
        voxel_alpha=np.asarray(result["voxel_alpha"], dtype=np.float32),
        cluster_alpha=np.asarray(result["cluster_alpha"], dtype=np.float32),
        cluster_connectivity=np.asarray(result["cluster_connectivity"], dtype=np.int32),
        min_cluster_size=np.asarray(result["min_cluster_size"], dtype=np.int32),
        min_cluster_mass=np.asarray(result["min_cluster_mass"], dtype=np.float32),
        merge_similar_clusters=np.asarray(result["merge_similar_clusters"]),
        merge_max_distance_px=np.asarray(result["merge_max_distance_px"], dtype=np.float32),
        merge_min_trace_corr=np.asarray(result["merge_min_trace_corr"], dtype=np.float32),
    )
    return save_path


def analyze_cell_from_saved(
    cell_id,
    sta_dir,
    suite2p_ctx,
    *,
    selection_mode: str = "latest",
    selected_version: int | None = None,
    verbal=True,
):
    neuron_dir = Path(sta_dir) / f"cell_{cell_id}"
    npz_path = pick_cluster_npz_by_mode(
        neuron_dir,
        selection_mode=selection_mode,
        selected_version=selected_version,
    )
    if npz_path is None:
        target_desc = (
            "latest available file"
            if str(selection_mode).lower().strip() == "latest"
            else f"version v{int(selected_version)}"
        )
        raise FileNotFoundError(
            f"No cluster permutation NPZ found for cell {cell_id} in {neuron_dir} "
            f"for selection: {target_desc}"
        )
    
    if verbal:
        print(f"[saved-load] cell {cell_id}: {npz_path.name} (mode={selection_mode})")
    npz_data = load_saved_npz(npz_path)
    effect, post_pre, t_rel, spatial_bin, threshold, cluster_alpha, null_max, cluster_connectivity, min_cluster_size, min_cluster_mass, merge_similar_clusters, merge_max_distance_px, merge_min_trace_corr = itemgetter(
        "effect", "post_pre", "t_rel", "spatial_bin", "threshold", "cluster_alpha", "null_max",
        "cluster_connectivity", "min_cluster_size", "min_cluster_mass", "merge_similar_clusters", "merge_max_distance_px", "merge_min_trace_corr",
    )(npz_data)
    
    neuron_yx = neuron_centroid_binned(cell_id, spatial_bin, post_pre.shape, suite2p_ctx)

    clusters = extract_clusters(
        effect_movie=effect,
        threshold=threshold,
        connectivity=cluster_connectivity,
        min_cluster_size=min_cluster_size,
        min_cluster_mass=min_cluster_mass,
        merge_similar=merge_similar_clusters,
        merge_max_distance_px=merge_max_distance_px,
        merge_min_trace_corr=merge_min_trace_corr,
        null_max=null_max,
        cluster_alpha=cluster_alpha,
    )
    sig_clusters = [c for c in clusters if c["is_significant"]]
    nonsig_clusters = [c for c in clusters if not c["is_significant"]]

    sig_mask3d = np.zeros_like(effect, dtype=bool)
    for c in sig_clusters:
        sig_mask3d |= c["mask3d"]

    regions = []
    if sig_mask3d.any():
        regions = build_regions_from_sig_mask(sig_mask3d, effect, t_rel, neuron_yx)

    return {
        "cell_id": int(cell_id),
        "t_rel": t_rel,
        "effect": effect,
        "post_pre": post_pre,
        "spatial_bin": spatial_bin,
        "neuron_yx": neuron_yx,
        "clusters": clusters,
        "sig_clusters": sig_clusters,
        "nonsig_clusters": nonsig_clusters,
        "regions": regions,
        "threshold": threshold,
        'cluster_alpha': float(cluster_alpha),
        'null_max': null_max,
    }
