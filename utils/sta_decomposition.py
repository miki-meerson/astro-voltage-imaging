"""PCA and ICA of the STA movie of one cell, over the whole field.

The movie is the saved spike-triggered average (frames x H x W, relative change). It is taken as change from the baseline (the same baseline map the cluster test subtracts,
utils/sta_stats_utils.py `sta_baseline_map`) and written as a matrix of time x pixels. Nothing else is removed from it, so the variance is the variance about the baseline: a component
that is the response to the spike counts, a constant offset does not.

PCA: the singular value decomposition of that matrix. Component k has a spatial map (a unit vector over the pixels) and a time course (in the units of the movie, the map's amplitude
times its time course rebuilds the movie), and explains `s_k^2 / sum(s^2)` of the variance. Components are kept up to the smallest number that explains `variance_target` of it (at least
2, at most `max_components`).

ICA: spatial ICA of the kept PCA components: the pixels are the samples, so the sources are independent spatial maps; their time courses are the least-squares time courses of the
movie on those maps. It is FastICA (symmetric, tanh), started from a random rotation of a seed that is saved with the result. ICA components have no singular value: the variance each one
accounts for is its share of the movie rebuilt from the ICA components alone.

The sign of a component is arbitrary in both methods. It is fixed here so that the largest value of the component's time course is positive: the map then shows where the signal
goes the same way as the response (positive) or the opposite way (negative). The signs are NOT the polarity of the clusters.

Pure numpy, no movie files: the cluster job reads the movie and calls `decompose_sta_movie`, the tests call it on synthetic movies."""
from __future__ import annotations

import numpy as np

from utils.sta_stats_utils import sta_baseline_map

VARIANCE_TARGET = 0.9  # the share of the variance the kept PCA components explain, at least
MAX_COMPONENTS = 10
MIN_COMPONENTS = 2  # ICA of one component would be the component itself
SAVED_VARIANCES = 20  # how many explained variances are kept for the scree plot
RANK_TOLERANCE = 1e-6  # a singular value below this share of the first one is not a component


def _orient(maps: np.ndarray, courses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip every component (map and course together) so that the largest value of its time course is positive."""
    peak = courses[np.argmax(np.abs(courses), axis=0), np.arange(courses.shape[1])]
    sign = np.where(peak < 0, -1.0, 1.0).astype(np.float32)
    return maps * sign[:, None], courses * sign[None, :]


def _symmetric_decorrelation(w: np.ndarray) -> np.ndarray:
    """The rows of `w` made orthonormal, as little changed as possible: (w w^T)^(-1/2) w."""
    values, vectors = np.linalg.eigh(w @ w.T)
    return vectors @ np.diag(values ** -0.5) @ vectors.T @ w


def _fast_ica(whitened: np.ndarray, seed: int, max_iter: int = 500, tolerance: float = 1e-7) -> tuple[np.ndarray, bool]:
    """(unmixing matrix, converged) of symmetric FastICA with the tanh nonlinearity on whitened data (components x samples)."""
    n_components, n_samples = whitened.shape
    w = _symmetric_decorrelation(np.random.default_rng(seed).standard_normal((n_components, n_components)))
    for _ in range(max_iter):
        y = np.tanh(w @ whitened)
        w_new = _symmetric_decorrelation(y @ whitened.T / n_samples - (1.0 - y ** 2).mean(axis=1)[:, None] * w)
        change = np.max(np.abs(np.abs(np.einsum("ij,ij->i", w_new, w)) - 1.0))
        w = w_new
        if change < tolerance:
            return w, True
    return w, False


def _spatial_ica(pca_maps: np.ndarray, seed: int) -> tuple[np.ndarray, bool]:
    """(independent maps (K x pixels), converged) of the PCA maps (K x pixels): centred over the pixels, whitened, then FastICA."""
    centred = pca_maps - pca_maps.mean(axis=1, keepdims=True)
    values, vectors = np.linalg.eigh(centred @ centred.T / centred.shape[1])
    whitening = np.diag(np.maximum(values, np.finfo(np.float64).tiny) ** -0.5) @ vectors.T
    unmixing, converged = _fast_ica(whitening @ centred, seed)
    return (unmixing @ whitening @ centred).astype(np.float32), converged


def decompose_sta_movie(sta_movie: np.ndarray, pre_frames: int, post_frames: int, baseline_window: tuple[int, int], seed: int = 0,
                        variance_target: float = VARIANCE_TARGET, max_components: int = MAX_COMPONENTS) -> dict:
    """The PCA and ICA of an STA movie (frames x H x W; frame i is `i - pre_frames` frames from the spike; `baseline_window` in frames from the spike).

    Returns {rels (frames from the spike), n_components, variance_explained (the first SAVED_VARIANCES of PCA, shares of 1), pca_maps (K x H x W), pca_courses (T x K),
    ica_maps, ica_courses, ica_variance (the share of the variance of the movie rebuilt from the ICA components that each one accounts for), ica_converged, seed}. Components are
    ordered by variance. Raises ValueError for a movie that is not frames x H x W, has non-finite values, has no change from the baseline at all, or has no frame inside the baseline."""
    movie = np.asarray(sta_movie, dtype=np.float32)
    if movie.ndim != 3:
        raise ValueError(f"The STA movie must be frames x height x width, not {movie.shape}.")
    if not np.isfinite(movie).all():
        raise ValueError("The STA movie has values that are not numbers (NaN or infinite): it cannot be decomposed.")
    rels = np.arange(-int(pre_frames), int(post_frames) + 1)
    if len(rels) != movie.shape[0]:
        raise ValueError(f"The STA movie has {movie.shape[0]} frames but the window -{pre_frames}..{post_frames} has {len(rels)}.")
    if not ((rels >= baseline_window[0]) & (rels <= baseline_window[1])).any():
        raise ValueError(f"No frame of the STA movie lies inside the baseline window {tuple(baseline_window)}.")
    frames, height, width = movie.shape
    data = (movie - sta_baseline_map(movie, rels, baseline_window)).reshape(frames, height * width)
    # singular value decomposition through the frames x frames matrix: the movie has far more pixels than frames
    values, vectors = np.linalg.eigh((data @ data.T).astype(np.float64))
    order = np.argsort(values)[::-1]
    squares, vectors = np.maximum(values[order], 0.0), vectors[:, order]
    if squares[0] <= 0:
        raise ValueError("The STA movie has no change from its baseline at all: there is nothing to decompose.")
    explained = squares / squares.sum()
    rank = int(np.sum(np.sqrt(squares) > RANK_TOLERANCE * np.sqrt(squares[0])))
    k = min(max(int(np.searchsorted(np.cumsum(explained), variance_target) + 1), MIN_COMPONENTS), int(max_components), rank)
    singular = np.sqrt(squares[:k])
    pca_maps = ((data.T @ vectors[:, :k]) / singular).T.astype(np.float32)  # K x pixels, unit length
    pca_courses = (vectors[:, :k] * singular).astype(np.float32)  # frames x K
    pca_maps, pca_courses = _orient(pca_maps, pca_courses)
    ica_maps, converged = _spatial_ica(pca_maps, seed) if k >= 2 else (pca_maps, True)
    ica_courses = np.linalg.lstsq(ica_maps.T, data.T, rcond=None)[0].T.astype(np.float32)  # frames x K: the least-squares time course of the movie on each map
    energy = np.sum(ica_courses ** 2, axis=0) * np.sum(ica_maps ** 2, axis=1)
    by_variance = np.argsort(energy)[::-1]
    ica_maps, ica_courses = _orient(ica_maps[by_variance], ica_courses[:, by_variance])
    return {
        "rels": rels, "n_components": k, "variance_explained": explained[:SAVED_VARIANCES].astype(np.float32),
        "pca_maps": pca_maps.reshape(k, height, width), "pca_courses": pca_courses,
        "ica_maps": ica_maps.reshape(k, height, width), "ica_courses": ica_courses, "ica_variance": (energy[by_variance] / energy.sum()).astype(np.float32),
        "ica_converged": bool(converged), "seed": int(seed),
    }
