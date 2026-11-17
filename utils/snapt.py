"""SNAPT propagation maps of the astrocyte STA movie of one cell (the idea of Cohen's SNAPT, adapted: the 'soma' is the significant clusters of the cell and the kernel is their STA trace).

For each sign of the cluster (positive and negative are never averaged together):
 1. kernel: the mean STA trace (dF/F, the one the cluster plots show) of the cell's significant clusters of that sign, smoothed a little, turned upside down for the negative sign so that
    the response is a rise. The kernel is cut from KERNEL_PRE_SECONDS before to KERNEL_POST_SECONDS after its peak, and scaled to 0..1.
 2. movie: the STA movie rebuilt from the first DENOISE_COMPONENTS PCA components (utils/sta_decomposition.py), cut to the same frames, turned upside down for the negative sign, smoothed
    over space by a small Gaussian and scaled to 0..1.
 3. fit: every pixel's time course is fitted by  a * kernel(c * (t - dt)) + b  (the kernel is stretched about its own peak, `shift_kernel`): a = amplitude, b = offset, 1/c = width
    (above 1: slower than the kernel), dt = delay in frames (positive: later). Unconstrained least squares (Levenberg-Marquardt) from [1, 0, 1, 0], like MATLAB's nlinfit.
 4. constraints: a pixel is accepted when a > 0, the width lies in WIDTH_RANGE and the delay within +-DELAY_LIMIT_SECONDS. The amplitude of the others is set to 0, width and delay are clipped.
 5. the fitted movie: the fitted curves of the accepted pixels on a finer time axis (UPSAMPLE times), the baseline taken off, scaled to 0..1.

There is no test of significance here: these maps describe how the response of every pixel compares with the clusters' response, they are not a statistical result.
Pure numpy/scipy: the cluster job reads the files and calls `snapt_cell` (utils/snapt_run.py)."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.stats import t as student_t

from utils import globals_utils as g

KERNEL_PRE_SECONDS = 0.5  # the kernel starts this long before its peak
KERNEL_POST_SECONDS = 1.0  # ... and ends this long after it
KERNEL_SMOOTH_SECONDS = 0.07  # the mean trace is smoothed by a Gaussian of this standard deviation before its peak is taken and it is cut
DENOISE_COMPONENTS = 5  # PCA components of the decomposition that the movie is rebuilt from
SMOOTH_SIGMA_PX = 1.5  # spatial smoothing of the movie, in pixels of the (binned) movie
SMOOTH_TRUNCATE = 1.0  # the Gaussian is cut at this many sigma: a 5 x 5 window
WIDTH_RANGE = (0.2, 4.0)  # accepted widths (1 = as wide as the kernel)
DELAY_LIMIT_SECONDS = 0.5  # accepted delays: within plus and minus this
CONFIDENCE_ALPHA = 0.318  # the interval of the delay is the 68% one (one standard error)
UPSAMPLE = 3  # the fitted movie has this many times more frames than the kernel
MIN_KERNEL_FRAMES = 6  # four parameters are fitted: fewer frames than this leave no degrees of freedom to speak of
FIT_PARAMETERS = 4
MOVIE_PERCENTILES = (1, 99)  # the fitted movie is scaled between these percentiles of its values


def build_kernel(traces: list[tuple[np.ndarray, np.ndarray]], sign: str, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """(kernel scaled 0..1, its frames from the spike) of the clusters' STA traces of one `sign`: `traces` is [(frames from the spike, dF/F trace), ...], one per cluster (the clusters of
    different results may cover different frames: the mean is over the frames they all have). Raises ValueError for a kernel that is too short or flat."""
    t_rel = traces[0][0]
    for t, _ in traces[1:]:
        t_rel = np.intersect1d(t_rel, t)
    mean = gaussian_filter1d(np.mean([np.asarray(trace, dtype=np.float64)[np.isin(t, t_rel)] for t, trace in traces], axis=0), KERNEL_SMOOTH_SECONDS * fps, mode="nearest")
    if sign == g.NEGATIVE_SIGN:
        mean = -mean
    peak = t_rel[int(np.argmax(mean))]
    keep = (t_rel >= peak - round(KERNEL_PRE_SECONDS * fps)) & (t_rel <= peak + round(KERNEL_POST_SECONDS * fps))
    kernel = mean[keep]
    if kernel.size < MIN_KERNEL_FRAMES:
        raise ValueError(f"The {sign} kernel has only {kernel.size} frames around its peak (at least {MIN_KERNEL_FRAMES} are needed).")
    if kernel.max() <= kernel.min():
        raise ValueError(f"The {sign} kernel is flat.")
    return (kernel - kernel.min()) / (kernel.max() - kernel.min()), t_rel[keep]


def prepare_movie(denoised: np.ndarray, rels: np.ndarray, kernel_rels: np.ndarray, sign: str) -> np.ndarray:
    """The movie to fit: the frames `kernel_rels` of `denoised` (frames `rels` from the spike), upside down for the negative sign, smoothed over space, scaled to 0..1."""
    if not np.isin(kernel_rels, rels).all():
        raise ValueError("The kernel covers frames that the decomposed STA movie does not have.")
    movie = denoised[np.searchsorted(rels, kernel_rels)].astype(np.float64)
    if sign == g.NEGATIVE_SIGN:
        movie = -movie
    movie = gaussian_filter(movie, sigma=(0, SMOOTH_SIGMA_PX, SMOOTH_SIGMA_PX), mode="nearest", truncate=SMOOTH_TRUNCATE)
    if movie.max() <= movie.min():
        raise ValueError("The decomposed STA movie does not change in the frames of the kernel.")
    return (movie - movie.min()) / (movie.max() - movie.min())


def shift_kernel(kernel: np.ndarray, amplitude: float, offset: float, scale: float, delay: float, times: np.ndarray | None = None) -> np.ndarray:
    """`amplitude * kernel(scale * (t - delay)) + offset` at the frames `times` (default: the frames of the kernel), the kernel stretched about its own peak, held at its end values beyond its
    ends, linear between its frames (the shiftkern of the SNAPT code)."""
    n = len(kernel)
    padded = np.concatenate(([kernel[0]], kernel, [kernel[-1]])) * amplitude + offset
    peak = int(np.argmax(padded)) - 1
    times = np.arange(n) if times is None else times
    return np.interp(peak + (times - peak - delay) * scale, np.arange(-1, n + 1), padded)


def fit_pixel(kernel: np.ndarray, trace: np.ndarray) -> dict | None:
    """{amplitude, offset, scale, delay (frames), delay_error (frames, the 68% interval), r2, converged} of one pixel's trace, or None when the trace is flat or the fit stops with no result.
    `converged` is False when the fit gave a warning (the covariance could not be estimated)."""
    if np.ptp(trace) == 0:
        return None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            beta, covariance = curve_fit(lambda _, a, b, c, dt: shift_kernel(kernel, a, b, c, dt), np.arange(len(kernel)), trace, p0=[1.0, 0.0, 1.0, 0.0], method="lm", maxfev=2000)
        except RuntimeError:
            return None
    residual = trace - shift_kernel(kernel, *beta)
    half_interval = student_t.ppf(1 - CONFIDENCE_ALPHA / 2, max(len(kernel) - FIT_PARAMETERS, 1)) * np.sqrt(covariance[3, 3])
    return {"amplitude": beta[0], "offset": beta[1], "scale": beta[2], "delay": beta[3], "delay_error": half_interval, "r2": 1 - residual.var() / trace.var(),
            "converged": not caught and bool(np.isfinite(covariance).all())}


def fit_movie(kernel: np.ndarray, movie: np.ndarray) -> dict:
    """`fit_pixel` of every pixel of the movie (frames x H x W): {name: H x W array} (NaN where there is no fit; `converged` is a boolean map)."""
    height, width = movie.shape[1:]
    names = ("amplitude", "offset", "scale", "delay", "delay_error", "r2")
    out = {name: np.full((height, width), np.nan) for name in names}
    out["converged"] = np.zeros((height, width), dtype=bool)
    for y, x in np.ndindex(height, width):
        fit = fit_pixel(kernel, movie[:, y, x])
        if fit is not None:
            for name in names:
                out[name][y, x] = fit[name]
            out["converged"][y, x] = fit["converged"]
    return out


def constrain(fit: dict, fps: float) -> dict:
    """The fit with the constraints applied: {amplitude (0 where not accepted), offset, scale, width (clipped), delay (frames, clipped), delay_seconds (clipped), delay_error_seconds,
    r2 (0 where there is no fit), accepted, converged}. A pixel with no fit has 0 in every map (never NaN: the maps go to the page as JSON)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        width = 1.0 / fit["scale"]
    limit = DELAY_LIMIT_SECONDS * fps
    accepted = (fit["amplitude"] > 0) & (width > WIDTH_RANGE[0]) & (width < WIDTH_RANGE[1]) & (fit["delay"] > -limit) & (fit["delay"] < limit)
    delay = np.clip(fit["delay"], -limit, limit)
    out = {"amplitude": np.where(accepted, fit["amplitude"], 0.0), "offset": fit["offset"], "scale": 1.0 / np.clip(width, *WIDTH_RANGE), "width": np.clip(width, *WIDTH_RANGE), "delay": delay,
           "delay_seconds": delay / fps, "delay_error_seconds": fit["delay_error"] / fps, "r2": fit["r2"], "accepted": accepted, "converged": fit["converged"]}
    return {name: np.nan_to_num(value) for name, value in out.items()}


def fitted_movie(kernel: np.ndarray, fit: dict) -> np.ndarray:
    """The fitted curves of the accepted pixels on a time axis UPSAMPLE times finer: frames x H x W, uint8 (the baseline of each pixel taken off, the values between MOVIE_PERCENTILES scaled to 0..255)."""
    n = len(kernel)
    times = np.linspace(0, n - 1, UPSAMPLE * n)
    movie = np.zeros((len(times), *fit["amplitude"].shape))
    for y, x in zip(*np.nonzero(fit["accepted"])):
        movie[:, y, x] = shift_kernel(kernel, fit["amplitude"][y, x], fit["offset"][y, x], fit["scale"][y, x], fit["delay"][y, x], times)
    movie -= movie[: max(n // 4, 1) * UPSAMPLE].mean(axis=0)
    low, high = np.percentile(movie, MOVIE_PERCENTILES)
    return np.round(np.clip((movie - low) / max(high - low, 1e-12), 0, 1) * 255).astype(np.uint8)


def snapt_cell(denoised: np.ndarray, rels: np.ndarray, clusters: list[dict], fps: float) -> dict[str, dict]:
    """The propagation maps of one cell for each sign that has significant clusters: {sign: {n_clusters, origin ([y, x] in pixels of the movie: the centre of the clusters'
    footprints), kernel, kernel_rels, movie_rels (frames from the spike of the fitted movie), movie (fitted movie), and the arrays of `constrain`}}. `denoised` is the STA movie rebuilt from the
    decomposition (frames `rels` from the spike); `clusters` are the significant clusters of the 'all' and the anatomy results: {sign, type (the result type), t (frames from the spike), trace (dF/F),
    mask (the 2D footprint on the pixel grid of the movie)}. Raises ValueError when no sign gives a result (the reason is in the message)."""
    results, reasons = {}, []
    for sign in (g.POSITIVE_SIGN, g.NEGATIVE_SIGN):
        chosen = [c for c in clusters if c["sign"] == sign]
        if not chosen:
            continue
        try:
            kernel, kernel_rels = build_kernel([(c["t"], c["trace"]) for c in chosen], sign, fps)
            fit = constrain(fit_movie(kernel, prepare_movie(denoised, rels, kernel_rels, sign)), fps)
        except ValueError as exc:
            reasons.append(str(exc))
            continue
        results[sign] = {"n_clusters": len(chosen), "n_anatomy": sum(c["type"] == g.ANATOMY_GROUP for c in chosen), "origin": np.argwhere(np.any([c["mask"] for c in chosen], axis=0)).mean(axis=0), "kernel": kernel, "kernel_rels": kernel_rels, "movie_rels": kernel_rels[0] + np.linspace(0, len(kernel) - 1, UPSAMPLE * len(kernel)),
                         "movie": fitted_movie(kernel, fit), **fit}
    if not results:
        raise ValueError(" ".join(reasons) or "The cell has no significant clusters.")
    return results
