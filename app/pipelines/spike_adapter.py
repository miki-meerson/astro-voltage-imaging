from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from utils.movie_utils import read_frame_rate
from utils.suite2p_utils import load_suite2p_context
from utils.spikes_utils import classify_spikes, detect_peaks_mad, highpass_traces
from utils.paths_utils import PathKind, path_for
from utils.spike_params import spike_params_with_defaults


def _save_review_png(path: Path, trace, peaks, labels) -> None:
    w, h = 1200, 260
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    y = np.asarray(trace, dtype=float)
    lo, hi = np.nanpercentile(y, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    def xy(frame, val):
        x = int(frame / max(1, y.size - 1) * (w - 40)) + 20
        yy = h - 28 - int((float(val) - lo) / max(hi - lo, 1e-6) * (h - 54))
        return x, yy
    step = max(1, int(np.ceil(y.size / 1000)))
    pts = [xy(i, y[i]) for i in range(0, y.size, step)]
    if len(pts) > 1:
        draw.line(pts, fill="#222222", width=1)
    colors = {"simple": "#2f80ed", "complex": "#d97706", "exclude": "#d64550"}
    for frame, label in zip(peaks, labels):
        if 0 <= int(frame) < y.size:
            x, yy = xy(int(frame), y[int(frame)])
            draw.ellipse((x - 3, yy - 3, x + 3, yy + 3), fill=colors.get(str(label), "#777777"))
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--experiment", required=True); ap.add_argument("--output", required=True); ap.add_argument("--params-json", default="{}")
    args = ap.parse_args(); home = Path(args.experiment); out = Path(args.output); out.mkdir(parents=True, exist_ok=True); params = json.loads(args.params_json or "{}")
    fps = int(read_frame_rate(home))
    suite2p = load_suite2p_context(path_for(PathKind.SUITE2P_PLANE0, home))
    traces = np.asarray(suite2p["F_norm"], dtype=np.float32)
    p = spike_params_with_defaults(params)
    detection = highpass_traces(traces, fps, p["highpass_cutoff_hz"])
    rows = []
    for cell_id, (trace, det) in enumerate(zip(traces, detection)):
        peaks = detect_peaks_mad(trace, det, mad_factor=p["mad_factor"], duplicate_window_frames=p["duplicate_window_frames"], align_radius_frames=p["align_radius_frames"])
        labels, features = classify_spikes(trace, peaks, fps, min_amplitude=p["min_amplitude"], min_width=p["min_width"], min_subthreshold=p["min_subthreshold"], subthreshold_window_seconds=0.2, baseline_window_seconds=1.5)
        path_for(PathKind.SPIKE_CELL_DIR, out, cell=cell_id).mkdir(exist_ok=True)
        payload = {"cell_id": np.asarray(cell_id, dtype=np.int32), "peaks_all": labels.peaks_all, "labels": labels.labels, "simple_peaks": labels.simple_peaks, "complex_peaks": labels.complex_peaks, "excluded_peaks": labels.excluded_peaks, **features}
        np.savez_compressed(path_for(PathKind.SPIKE_PEAKS, out, cell=cell_id), **payload); np.savez_compressed(path_for(PathKind.SPIKE_LABELS, out, cell=cell_id), **payload)
        np.savez(path_for(PathKind.SPIKE_PEAK_PARAMS, out, cell=cell_id), threshold_method="mad", mad_threshold_factor=p["mad_factor"], highpass_cutoff_hz=p["highpass_cutoff_hz"])
        _save_review_png(path_for(PathKind.SPIKE_REVIEW_PNG, out, cell=cell_id), trace, labels.peaks_all, labels.labels)
        rows.append({"cell_id": cell_id, "all": int(labels.peaks_all.size), "simple": int(labels.simple_peaks.size), "complex": int(labels.complex_peaks.size)})
    path_for(PathKind.RUN_METRICS_JSON, out).write_text(json.dumps({"cells": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
