from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from utils.movie_utils import get_movie_parameters
from utils.suite2p_utils import load_suite2p_context
from utils.spikes_utils import classify_spikes, detect_peaks_mad, highpass_traces


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
    fps = int(get_movie_parameters(f"{home.as_posix()}/", is_dir=True)["fps"])
    suite2p = load_suite2p_context(home / "pipeline_results/for_suite2p/suite2p/plane0")
    traces = np.asarray(suite2p["F_norm"], dtype=np.float32)
    detection = highpass_traces(traces, fps, 0.07)
    rows = []
    for cell_id, (trace, det) in enumerate(zip(traces, detection)):
        peaks = detect_peaks_mad(trace, det, mad_factor=float(params.get("mad_factor", 3.5)), duplicate_window_frames=30, align_radius_frames=5, min_distance=0)
        labels, features = classify_spikes(trace, peaks, fps, min_amplitude=float(params.get("min_amplitude", 0.30)), min_width=float(params.get("min_width", 18.0)), min_subthreshold=float(params.get("min_subthreshold", 0.10)), min_ratio=float(params.get("min_ratio", 0.30)), subthreshold_window_seconds=0.2, baseline_window_seconds=1.5)
        cell_dir = out / f"cell_{cell_id}"; cell_dir.mkdir(exist_ok=True)
        payload = {"cell_id": np.asarray(cell_id, dtype=np.int32), "peaks_all": labels.peaks_all, "labels": labels.labels, "simple_peaks": labels.simple_peaks, "complex_peaks": labels.complex_peaks, "excluded_peaks": labels.excluded_peaks, **features}
        np.savez_compressed(cell_dir / "simple_complex_peaks_for_analysis.npz", **payload); np.savez_compressed(cell_dir / "simple_complex_spike_labels.npz", **payload)
        np.savez(cell_dir / "peak_detection_params.npz", threshold_method="mad", mad_threshold_factor=float(params.get("mad_factor", 3.5)), min_distance=0, highpass_cutoff_hz=0.07)
        _save_review_png(cell_dir / "automatic_spike_classification_review.png", trace, labels.peaks_all, labels.labels)
        rows.append({"cell_id": cell_id, "all": int(labels.peaks_all.size), "simple": int(labels.simple_peaks.size), "complex": int(labels.complex_peaks.size)})
    (out / "metrics.json").write_text(json.dumps({"cells": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
