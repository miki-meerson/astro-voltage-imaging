from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.signal import butter, filtfilt
from utils.suite2p_utils import load_suite2p_context


@dataclass(frozen=True)
class PeakLabels:
    peaks_all: np.ndarray
    labels: np.ndarray
    simple_peaks: np.ndarray
    complex_peaks: np.ndarray
    excluded_peaks: np.ndarray


def highpass_traces(traces: np.ndarray, fps: int, cutoff_hz: float, order: int = 3) -> np.ndarray:
    nyquist = fps / 2.0
    traces = np.asarray(traces, dtype=np.float32)
    if cutoff_hz <= 0:
        return traces
    b, a = butter(order, cutoff_hz / nyquist, btype="high")
    return filtfilt(b, a, traces, axis=1).astype(np.float32)


def mad_threshold(trace: np.ndarray, factor: float) -> float:
    median = float(np.median(trace))
    sigma = float(1.4826 * np.median(np.abs(trace - median)))
    return median + factor * sigma


def collapse_nearby_peaks(trace: np.ndarray, peaks: np.ndarray, window_frames: int) -> np.ndarray:
    peaks = np.asarray(peaks, dtype=np.int32)
    if peaks.size == 0 or int(window_frames) <= 1:
        return peaks

    kept = []
    group = [int(peaks[0])]
    for peak in peaks[1:]:
        peak = int(peak)
        if peak - group[-1] < int(window_frames):
            group.append(peak)
        else:
            kept.append(max(group, key=lambda p: trace[p]))
            group = [peak]
    kept.append(max(group, key=lambda p: trace[p]))
    return np.asarray(kept, dtype=np.int32)


def align_peaks_to_trace(trace: np.ndarray, peaks: np.ndarray, radius: int) -> np.ndarray:
    aligned = []
    for peak in np.asarray(peaks, dtype=np.int32):
        lo = max(0, int(peak) - int(radius))
        hi = min(trace.size, int(peak) + int(radius) + 1)
        aligned.append(lo + int(np.argmax(trace[lo:hi])))
    return np.unique(np.asarray(aligned, dtype=np.int32))


def detect_peaks_mad(
    original_trace: np.ndarray,
    detection_trace: np.ndarray,
    mad_factor: float,
    duplicate_window_frames: int,
    align_radius_frames: int,
    min_distance: int = 0,
) -> np.ndarray:
    distance = None if int(min_distance) <= 0 else int(min_distance)
    peaks, _ = find_peaks(detection_trace, height=mad_threshold(detection_trace, mad_factor), distance=distance)
    peaks = collapse_nearby_peaks(detection_trace, peaks, duplicate_window_frames)
    return align_peaks_to_trace(original_trace, peaks, align_radius_frames)


def spike_features(
    trace: np.ndarray,
    peaks: np.ndarray,
    fps: int,
    subthreshold_window_seconds: float,
    baseline_window_seconds: float,
) -> dict[str, np.ndarray]:
    peaks = np.asarray(peaks, dtype=np.int32)
    if peaks.size == 0:
        return {
            "amplitude": np.empty(0, dtype=np.float32),
            "width": np.empty(0, dtype=np.float32),
            "subthreshold_depolarization": np.empty(0, dtype=np.float32),
        }

    widths = peak_widths(trace, peaks, rel_height=0.5)[0].astype(np.float32)
    depol_window = max(3, int(round(subthreshold_window_seconds * fps)))
    baseline_window = max(3, int(round(baseline_window_seconds * fps)))
    gap = max(1, int(round(0.03 * fps)))

    amplitudes = np.empty(peaks.size, dtype=np.float32)
    subthreshold = np.empty(peaks.size, dtype=np.float32)
    for i, peak in enumerate(peaks):
        baseline_lo = max(0, int(peak) - depol_window - baseline_window)
        baseline_hi = max(baseline_lo + 1, int(peak) - depol_window)
        depol_lo = max(0, int(peak) - depol_window)
        depol_hi = max(depol_lo + 1, int(peak) - gap)
        baseline = float(np.median(trace[baseline_lo:baseline_hi]))
        amplitudes[i] = float(trace[peak] - baseline)
        subthreshold[i] = float(np.mean(trace[depol_lo:depol_hi]) - baseline)

    return {
        "amplitude": amplitudes,
        "width": widths,
        "subthreshold_depolarization": subthreshold,
    }


def classify_spikes(
    trace: np.ndarray,
    peaks: np.ndarray,
    fps: int,
    min_amplitude: float,
    min_width: float,
    min_subthreshold: float,
    min_ratio: float,
    subthreshold_window_seconds: float = 0.2,
    baseline_window_seconds: float = 1.5,
) -> tuple[PeakLabels, dict[str, np.ndarray]]:
    peaks = np.asarray(peaks, dtype=np.int32)
    features = spike_features(trace, peaks, fps, subthreshold_window_seconds, baseline_window_seconds)
    labels = np.full(peaks.size, "simple", dtype="U8")

    if peaks.size == 0:
        features["score"] = np.empty(0, dtype=np.float32)
    else:
        amplitude = features["amplitude"]
        width = features["width"]
        subthreshold = features["subthreshold_depolarization"]
        ratio = subthreshold / (amplitude + 1e-12)
        complex_mask = (
            (amplitude >= min_amplitude)
            & (width >= min_width)
            & (subthreshold >= min_subthreshold)
            & (ratio >= min_ratio)
        )
        labels[complex_mask] = "complex"
        features["subthreshold_ratio"] = ratio.astype(np.float32)
        features["score"] = np.minimum.reduce(
            [
                amplitude / min_amplitude,
                width / min_width,
                subthreshold / min_subthreshold,
                ratio / min_ratio,
            ]
        ).astype(np.float32)

    labels = np.char.lower(labels)
    return (
        PeakLabels(
            peaks_all=peaks,
            labels=labels,
            simple_peaks=peaks[labels == "simple"],
            complex_peaks=peaks[labels == "complex"],
            excluded_peaks=peaks[labels == "exclude"],
        ),
        features,
    )


def _suite2p_trace_dff(
    neuron_id: int,
    *,
    suite2p_ctx: dict | None = None,
    suite2p_path=None,
    F_raw: np.ndarray | None = None,
    baseline_percentile: float = 10.0,
) -> np.ndarray:
    """Convert a raw Suite2p fluorescence trace to dF/F using a robust per-neuron baseline."""
    if F_raw is None:
        if suite2p_ctx is None:
            if suite2p_path is None:
                raise ValueError("Pass suite2p_ctx, suite2p_path, or F_raw to _suite2p_trace_dff.")
            suite2p_ctx = load_suite2p_context(suite2p_path)
        F_raw = np.asarray(suite2p_ctx["F"], dtype=np.float32)
    else:
        F_raw = np.asarray(F_raw, dtype=np.float32)
    trace = np.asarray(F_raw[int(neuron_id)], dtype=np.float32).ravel()
    f0 = float(np.nanpercentile(trace, baseline_percentile))
    if not np.isfinite(f0) or abs(f0) <= 1e-6:
        f0 = float(np.nanmedian(trace))
    if not np.isfinite(f0) or abs(f0) <= 1e-6:
        return np.full_like(trace, np.nan, dtype=np.float32)
    return ((trace - f0) / (abs(f0) + 1e-6)).astype(np.float32)
