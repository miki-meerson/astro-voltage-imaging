import numpy as np
from scipy.ndimage import generate_binary_structure, label
from utils.sta_stats_utils import passes_extent, per_sign_p_values


def cluster_mask2d_from_cluster(
    cluster_like,
    *,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Return a 2D boolean mask from a cluster-like mapping/record.

    Supported inputs (checked in this order):
    - "mask2d": already 2D
    - "mask3d": collapsed across time with any(axis=0)
    - "mask":  either 2D or 3D (3D collapsed with any(axis=0))
    - ("ypix", "xpix"): pixel coordinate arrays (requires image_shape)
    """
    if "mask2d" in cluster_like and cluster_like["mask2d"] is not None:
        return np.asarray(cluster_like["mask2d"], dtype=bool)

    if "mask3d" in cluster_like and cluster_like["mask3d"] is not None:
        return np.asarray(cluster_like["mask3d"], dtype=bool).any(axis=0)

    if "mask" in cluster_like and cluster_like["mask"] is not None:
        m = np.asarray(cluster_like["mask"], dtype=bool)
        if m.ndim == 2:
            return m
        if m.ndim == 3:
            return m.any(axis=0)

    if (
        "ypix" in cluster_like
        and "xpix" in cluster_like
        and cluster_like["ypix"] is not None
        and cluster_like["xpix"] is not None
    ):
        if image_shape is None:
            raise ValueError("image_shape is required when building mask from ypix/xpix.")
        yy = np.asarray(cluster_like["ypix"], dtype=int)
        xx = np.asarray(cluster_like["xpix"], dtype=int)
        m = np.zeros(image_shape, dtype=bool)
        m[yy, xx] = True
        return m

    return None


def _cluster_spatial_centroid(mask3d: np.ndarray) -> tuple[float, float]:
    _, yy, xx = np.where(mask3d)
    if yy.size == 0:
        return 0.0, 0.0
    return float(np.mean(yy)), float(np.mean(xx))


def _cluster_trace(effect_movie: np.ndarray, mask3d: np.ndarray) -> np.ndarray:
    mask2d = cluster_mask2d_from_cluster({"mask3d": mask3d})
    return np.asarray(effect_movie[:, mask2d].mean(axis=1), dtype=np.float32)


def _merge_cluster_group(effect_movie: np.ndarray, group: list[dict]) -> dict:
    sign_name = str(group[0]["sign"])
    merged_mask = np.zeros_like(group[0]["mask"], dtype=bool)
    for c in group:
        merged_mask |= c["mask"]
    cluster_values = effect_movie[merged_mask]
    peak_index = int(np.argmax(np.abs(cluster_values)))
    peak_value = float(cluster_values[peak_index])
    return {
        "sign": sign_name,
        "mass": float(np.abs(cluster_values).sum()),
        "size": int(merged_mask.sum()),
        "peak_value": peak_value,
        "mask": merged_mask,
    }


def _trace_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size == 0:
        return 0.0
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    da = float(np.sqrt(np.sum(a0 * a0)))
    db = float(np.sqrt(np.sum(b0 * b0)))
    if da <= 1e-12 or db <= 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.sum(a0 * b0) / (da * db))


def _merge_similar_clusters(
    cluster_records: list[dict],
    effect_movie: np.ndarray,
    merge_max_distance_px: float,
    merge_min_trace_corr: float,
) -> list[dict]:
    if len(cluster_records) <= 1:
        return cluster_records

    n = len(cluster_records)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    centroids = [_cluster_spatial_centroid(c["mask"]) for c in cluster_records]
    traces = [_cluster_trace(effect_movie, c["mask"]) for c in cluster_records]

    for i in range(n):
        for j in range(i + 1, n):
            if cluster_records[i]["sign"] != cluster_records[j]["sign"]:
                continue
            dy = centroids[i][0] - centroids[j][0]
            dx = centroids[i][1] - centroids[j][1]
            dist = float(np.hypot(dy, dx))
            if dist > merge_max_distance_px:
                continue
            if _trace_corr(traces[i], traces[j]) < merge_min_trace_corr:
                continue
            union(i, j)

    groups: dict[int, list[dict]] = {}
    for i, c in enumerate(cluster_records):
        root = find(i)
        groups.setdefault(root, []).append(c)

    merged = [_merge_cluster_group(effect_movie, g) for g in groups.values()]
    return merged



def cluster_list_from_effect(
    effect_movie: np.ndarray,
    threshold: float,
    connectivity: int = 2,
    min_cluster_size: int = 1,
    min_cluster_mass: float = 0.0,
    merge_similar: bool = True,
    merge_max_distance_px: float = 3.0,
    merge_min_trace_corr: float = 0.9,
    min_footprint_px: int = 0,
    min_frames: int = 0,
    neg_threshold: float | None = None,
) -> tuple[list[dict], np.ndarray]:
    # neg_threshold (a positive magnitude) is only given by the opt-in per_sign_null rule; None = one symmetric threshold.
    structure = generate_binary_structure(3, connectivity)
    cluster_records: list[dict] = []
    significant = np.zeros_like(effect_movie, dtype=bool)
    neg_thr = threshold if neg_threshold is None else neg_threshold

    for sign_name, mask in (
        ("positive", effect_movie >= threshold),
        ("negative", effect_movie <= -neg_thr),
    ):
        labels, n_labels = label(mask, structure=structure)
        for label_id in range(1, n_labels + 1):
            cluster_mask = labels == label_id
            cluster_size = int(cluster_mask.sum())
            if cluster_size < int(min_cluster_size):
                continue
            if (min_footprint_px or min_frames) and not passes_extent(cluster_mask, min_footprint_px, min_frames):
                continue
            cluster_values = effect_movie[cluster_mask]
            cluster_mass = float(np.abs(cluster_values).sum())
            if cluster_mass < float(min_cluster_mass):
                continue
            peak_index = np.argmax(np.abs(cluster_values))
            peak_value = float(cluster_values[peak_index])
            cluster_records.append(
                {
                    "sign": sign_name,
                    "mass": cluster_mass,
                    "size": cluster_size,
                    "peak_value": peak_value,
                    "mask": cluster_mask,
                }
            )

    if merge_similar and len(cluster_records) > 1:
        cluster_records = _merge_similar_clusters(
            cluster_records=cluster_records,
            effect_movie=effect_movie,
            merge_max_distance_px=float(merge_max_distance_px),
            merge_min_trace_corr=float(merge_min_trace_corr),
        )

    for c in cluster_records:
        significant |= c["mask"]
    return cluster_records, significant


def extract_clusters(
    effect_movie: np.ndarray,
    threshold: float,
    connectivity: int = 2,
    min_cluster_size: int = 1,
    min_cluster_mass: float = 0.0,
    merge_similar: bool = True,
    merge_max_distance_px: float = 3.0,
    merge_min_trace_corr: float = 0.9,
    null_max: np.ndarray | None = None,
    cluster_alpha: float | None = None,
    min_footprint_px: int = 0,
    min_frames: int = 0,
    per_sign_null: bool = False,
    neg_threshold: float | None = None,
    null_max_neg: np.ndarray | None = None,
) -> list[dict]:
    """Clusters of an effect movie, with permutation p-values when a null is given.

    Legacy (per_sign_null False): one threshold, one pooled `null_max`.
    per_sign_null True: `threshold` is the positive threshold, `neg_threshold` the negative one, `null_max` the
    positive-cluster max-mass null and `null_max_neg` the negative one; p_value is the Bonferroni-corrected
    min(1, 2 * p_value_sign) (see utils.sta_stats_utils.per_sign_p_values) and p_value_sign is the uncorrected one.
    """
    clusters, _ = cluster_list_from_effect(
        effect_movie=effect_movie,
        threshold=threshold,
        connectivity=connectivity,
        min_cluster_size=min_cluster_size,
        min_cluster_mass=min_cluster_mass,
        merge_similar=merge_similar,
        merge_max_distance_px=merge_max_distance_px,
        merge_min_trace_corr=merge_min_trace_corr,
        min_footprint_px=min_footprint_px,
        min_frames=min_frames,
        neg_threshold=neg_threshold if per_sign_null else None,
    )
    clusters = sorted(clusters, key=lambda c: c["mass"], reverse=True)

    for i, c in enumerate(clusters, start=1):
        c["cluster_id"] = i
        c["mask3d"] = c.pop("mask")
        if per_sign_null:
            if null_max is not None and null_max_neg is not None and cluster_alpha is not None:
                p_sign, p = per_sign_p_values(c["mass"], c["sign"], null_max, null_max_neg)
                c["p_value_sign"] = p_sign
                c["p_value"] = p
                c["is_significant"] = bool(p <= cluster_alpha)
        elif null_max is not None and cluster_alpha is not None:
            p = (1 + np.sum(null_max >= c["mass"])) / (len(null_max) + 1)
            c["p_value"] = float(p)
            c["is_significant"] = bool(p <= cluster_alpha)
    return clusters


def saved_cluster_args(d) -> dict:
    """Keyword arguments for `extract_clusters` (all but effect_movie) from an opened STA result npz.

    Applies the same rule the result was produced with: when the file says `per_sign_null`, the per-sign thresholds
    and the two per-sign nulls are passed; otherwise the legacy single threshold / pooled null (also for old files
    without the flag). Callers that need a different null (leave-one-out) override the null_max* entries.
    """
    def scalar(key, cast, default):
        return cast(np.asarray(d[key]).item()) if key in d else default

    args = dict(
        threshold=float(np.asarray(d["cluster_threshold"]).item()),
        connectivity=int(np.asarray(d["cluster_connectivity"]).item()),
        min_cluster_size=int(np.asarray(d["min_cluster_size"]).item()),
        min_cluster_mass=float(np.asarray(d["min_cluster_mass"]).item()),
        merge_similar=bool(np.asarray(d["merge_similar_clusters"]).item()),
        merge_max_distance_px=float(np.asarray(d["merge_max_distance_px"]).item()),
        merge_min_trace_corr=float(np.asarray(d["merge_min_trace_corr"]).item()),
        null_max=np.asarray(d["null_max_cluster_mass"], dtype=np.float32),
        cluster_alpha=float(np.asarray(d["cluster_alpha"]).item()),
        min_footprint_px=scalar("min_cluster_footprint_px", int, 0),
        min_frames=scalar("min_cluster_frames", int, 0),
    )
    if scalar("per_sign_null", bool, False):
        args.update(
            per_sign_null=True,
            threshold=float(np.asarray(d["cluster_threshold_pos"]).item()),
            neg_threshold=float(np.asarray(d["cluster_threshold_neg"]).item()),
            null_max=np.asarray(d["null_max_cluster_mass_pos"], dtype=np.float32),
            null_max_neg=np.asarray(d["null_max_cluster_mass_neg"], dtype=np.float32),
        )
    return args


def per_sign_overrides(saved: dict) -> dict:
    """extract_clusters keyword overrides for a `load_saved_npz` dict: {} for legacy results, the per-sign rule otherwise."""
    if not saved.get("per_sign_null"):
        return {}
    return dict(per_sign_null=True, neg_threshold=saved["threshold_neg"], null_max=saved["null_max_pos"], null_max_neg=saved["null_max_neg"])
