import numpy as np
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt


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
) -> np.ndarray:
    peaks, _ = find_peaks(detection_trace, height=mad_threshold(detection_trace, mad_factor))
    peaks = collapse_nearby_peaks(detection_trace, peaks, duplicate_window_frames)
    return align_peaks_to_trace(original_trace, peaks, align_radius_frames)
