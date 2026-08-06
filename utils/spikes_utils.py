from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import butter, filtfilt
from IPython.display import display, clear_output
from utils.db_utils import get_peak_array_for_neuron
from utils.suite2p_utils import load_suite2p_context

@dataclass(frozen=True)
class PeakLabels:
    peaks_all: np.ndarray
    labels: np.ndarray
    simple_peaks: np.ndarray
    complex_peaks: np.ndarray
    excluded_peaks: np.ndarray


def compute_peaks(threshold_factor, min_distance, cell_trace, window_size=15):
    threshold = np.mean(cell_trace) + threshold_factor * np.std(cell_trace)
    distance = None if int(min_distance) <= 0 else int(min_distance)
    initial_peaks, _ = find_peaks(cell_trace, height=threshold, distance=distance)
    return collapse_nearby_peaks(cell_trace, initial_peaks, window_size)


def _highpass_filter(trace, fps, cutoff_hz, order=3):
    nyquist = fps / 2.0
    if cutoff_hz <= 0:
        return trace
    if cutoff_hz >= nyquist:
        raise ValueError(f"cutoff_hz ({cutoff_hz}) must be < Nyquist ({nyquist:.1f} Hz)")
    b, a = butter(order, cutoff_hz / nyquist, btype="high")
    return filtfilt(b, a, trace)


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


def detect_peaks_interactive(cell_trace, bad_segments, fps, threshold_factor=3.5, min_distance=70):
    trace_hp = _highpass_filter(cell_trace, fps=fps, cutoff_hz=0.05, order=3)

    fig, ax = plt.subplots(figsize=(12, 4))
    plt.subplots_adjust(bottom=0.25)
    
    # Main trace
    ax.plot(trace_hp, lw=0.5)
    ax.set_xlim(0, len(cell_trace))

    # Plot bad segments once
    for s, e in bad_segments:
        ax.axvspan(s, e, color='red', alpha=0.1)

    # Scatter for peaks (initially empty)
    scat = ax.scatter([], [], c='red', s=6)

    # Slider axes
    ax_threshold_factor = plt.axes([0.2, 0.12, 0.6, 0.03])
    ax_min_distance = plt.axes([0.2, 0.06, 0.6, 0.03])

    threshold_factor_slider = Slider(ax_threshold_factor, 'Threshold factor', 0.5, 6.0, valinit=threshold_factor, valstep=0.1)
    min_distance_slider = Slider(ax_min_distance, 'Min peak dist', 10, 300, valinit=min_distance, valstep=5)

    peaks_final = np.array([])
    threshold_factor_final = threshold_factor
    min_distance_final = min_distance

    def update(val=None):
        nonlocal peaks_final, threshold_factor_final, min_distance_final

        threshold_factor_final = float(threshold_factor_slider.val)
        min_distance_final = int(min_distance_slider.val)
        peaks_final = compute_peaks(threshold_factor_final, min_distance_final, trace_hp)

        if len(peaks_final) > 0:
            scat.set_offsets(np.c_[peaks_final, trace_hp[peaks_final]])
        else:
            scat.set_offsets(np.empty((0, 2)))

        fig.canvas.draw_idle()

    # Connect sliders
    threshold_factor_slider.on_changed(update)
    min_distance_slider.on_changed(update)

    # Initial draw
    update()
    plt.show(block=True)

    return peaks_final, threshold_factor_final, min_distance_final


def compute_sta_from_peaks(neuron_trace, peaks, pre_frames, post_frames, baseline_frames):
    t = np.asarray(neuron_trace, dtype=float).ravel()
    sta_len = int(pre_frames + post_frames + 1)
    if t.size == 0:
        return np.full(sta_len, np.nan, dtype=float)
    pks = np.asarray(peaks, dtype=np.int32).ravel()
    if pks.size == 0:
        return np.full(sta_len, np.nan, dtype=float)

    trials = []
    for p in pks:
        a = int(p - pre_frames)
        b = int(p + post_frames + 1)
        if a < 0 or b > t.size:
            continue
        seg = np.asarray(t[a:b], dtype=float)
        if seg.size != sta_len:
            continue
        bcount = min(int(baseline_frames), seg.size, max(1, int(pre_frames)))
        f0 = float(np.nanmedian(seg[:bcount]))
        seg_dff = (seg - f0) / (f0 + 1e-12)
        trials.append(seg_dff)

    if len(trials) == 0:
        return np.full(sta_len, np.nan, dtype=float)
    return np.nanmean(np.stack(trials, axis=0), axis=0)


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
                raise ValueError("Pass suite2p_ctx, suite2p_path, or F_raw to plot_stacked_dff_traces.")
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


def plot_stacked_dff_traces(
    neuron_ids,
    title,
    *,
    fps,
    neuron_df=None,
    suite2p_ctx: dict | None = None,
    suite2p_path=None,
    F_raw: np.ndarray | None = None,
    spacing=2.5,
    save_path=None,
    baseline_percentile: float = 10.0,
):
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    traces = [
        _suite2p_trace_dff(
            neuron_id,
            suite2p_ctx=suite2p_ctx,
            suite2p_path=suite2p_path,
            F_raw=F_raw,
            baseline_percentile=baseline_percentile,
        )
        for neuron_id in neuron_ids
    ]

    if not traces:
        raise ValueError("neuron_ids is empty.")

    trace_len = traces[0].shape[0]
    time_s = np.arange(trace_len, dtype=np.float32) / float(fps)

    offsets = np.arange(len(neuron_ids), dtype=float) * float(spacing)
    peak_marker_offset = 0.42 * float(spacing)
    label_x = -0.015 * float(time_s[-1])

    fig, ax = plt.subplots(figsize=(14, max(7, 0.12 * len(neuron_ids))), constrained_layout=True)
    for offset, neuron_id, tr in zip(offsets, neuron_ids, traces):
        ax.plot(time_s, tr + offset, color="black", lw=0.35, alpha=0.85)
        ax.text(label_x, offset, str(int(neuron_id)), ha="right", va="center", fontsize=6, color="black")

        if neuron_df is not None:
            all_peaks = get_peak_array_for_neuron(neuron_df, int(neuron_id), "all_peak_frames")
            all_peaks = all_peaks[(all_peaks >= 0) & (all_peaks < trace_len)]
            marker_y = offset + peak_marker_offset

            if all_peaks.size:
                ax.scatter(
                    time_s[all_peaks],
                    np.full(all_peaks.size, marker_y),
                    s=10,
                    marker="o",
                    color="red",
                    edgecolors="none",
                    alpha=0.9,
                    zorder=6,
                )

    if len(offsets):
        y_min = offsets[0] - spacing
        y_max = offsets[-1] + spacing
        ax.set_ylim(y_min, y_max)
    else:
        y_min, y_max = -1, 1

    scale_seconds = 20.0
    scale_dff = 0.20
    scale_x1 = float(time_s[-1]) - 0.03 * float(time_s[-1])
    scale_x0 = scale_x1 - scale_seconds
    scale_y = y_min + 0.35 * spacing
    ax.plot([scale_x0, scale_x1], [scale_y, scale_y], color="black", lw=1.6, solid_capstyle="butt")
    ax.text((scale_x0 + scale_x1) / 2, scale_y + 0.16 * spacing, "20 s", ha="center", va="bottom", fontsize=8)

    scale_vx = scale_x0
    ax.plot([scale_vx, scale_vx], [scale_y, scale_y + scale_dff], color="black", lw=1.6, solid_capstyle="butt")
    ax.text(
        scale_vx - 0.008 * float(time_s[-1]),
        scale_y + scale_dff / 2,
        "20% dF/F",
        ha="right",
        va="center",
        fontsize=8,
        rotation=90,
    )

    ax.set_xlim(label_x - 0.01 * float(time_s[-1]), float(time_s[-1]))
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()

    if save_path: fig.savefig(save_path, format="svg", bbox_inches="tight")

    plt.show()
