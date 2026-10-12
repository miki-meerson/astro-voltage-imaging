"""Significant clusters of a saved STA result file (used for the cluster outlines on the cropped STA movie)."""
from __future__ import annotations

import numpy as np

from utils.clusters_utils import extract_clusters, saved_cluster_args


def significant_clusters(d, limit: int | None = None) -> list[dict]:
    """Significant clusters of an opened STA result npz, largest mass first. Reads only the small keys (no permutations)."""
    effect = np.asarray(d["observed_effect_movie"], dtype=np.float32)
    clusters = extract_clusters(effect_movie=effect, **saved_cluster_args(d))  # honours per_sign_null when the file says so
    sig = [c for c in clusters if c.get("is_significant", False)]
    return sig if limit is None else sig[:limit]
