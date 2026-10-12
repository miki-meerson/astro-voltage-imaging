from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, tifffile
from PIL import Image, ImageDraw
from utils.movie_utils import get_movie_parameters
from utils.thorsync_utils import (
    TS_SAMPLING_RATE_HZ,
    find_ts_h5_files,
    load_ts_behavior,
    load_ts_encoder_velocity_at_times,
)


def _save_speed_png(path: Path, speed, fps: float, threshold: float) -> None:
    w, h = 1000, 260
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    y = np.asarray(speed, dtype=float)
    finite = y[np.isfinite(y)]
    if finite.size:
        lo, hi = float(np.nanpercentile(finite, 1)), float(np.nanpercentile(finite, 99))
    else:
        lo, hi = 0.0, 1.0
    hi = max(hi, threshold * 1.2, lo + 1e-6)
    def xy(frame, val):
        x = int(frame / max(1, y.size - 1) * (w - 50)) + 35
        yy = h - 30 - int((float(val) - lo) / max(hi - lo, 1e-6) * (h - 58))
        return x, yy
    step = max(1, int(np.ceil(y.size / 1000)))
    pts = [xy(i, y[i]) for i in range(0, y.size, step) if np.isfinite(y[i])]
    if len(pts) > 1:
        draw.line(pts, fill="#d97706", width=2)
    ty = xy(0, threshold)[1]
    draw.line((35, ty, w - 15, ty), fill="#222222", width=1)
    draw.text((35, 12), f"Behavior speed, threshold={threshold:g}, fps={fps:g}", fill="#222222")
    img.save(path)


def _pb_frame_indices(home: Path) -> np.ndarray | None:
    trace_path = home / "pipeline_results/qc/photobleaching_qc_trace_astro.npz"
    if not trace_path.exists():
        trace_path = home / "pipeline_results/qc/photobleaching_qc_trace.npz"
    if not trace_path.exists():
        return None
    with np.load(trace_path, allow_pickle=False) as d:
        if "frame_idx" not in d:
            return None
        return np.asarray(d["frame_idx"], dtype=int)


def _kept_frames(home: Path, fps: float) -> tuple[int, int, np.ndarray, np.ndarray, dict]:
    with tifffile.TiffFile(home / "Image_scan_1_region_0_0.tif") as tif:
        raw_frames = tif.series[0].shape[0] // 2
    with tifffile.TiffFile(home / "pipeline_results/pb_corrected/astro_photobleaching_corrected.tif") as tif:
        pb_frames = tif.series[0].shape[0]
    bad_path = home / "pipeline_results/qc/bad_segments.json"
    bad = json.loads(bad_path.read_text()) if bad_path.exists() else []
    removed = np.array(sorted({f for s, e in bad for f in range(int(s), int(e))}), dtype=int)
    kept_untrimmed = np.setdiff1d(np.arange(raw_frames), removed)
    pb_idx = _pb_frame_indices(home)
    trim_meta = {
        "pb_trimmed_frames": 0,
        "pb_trimmed_seconds": 0.0,
        "pb_frame_index_source": "",
    }
    if pb_idx is not None and pb_idx.size == pb_frames:
        if pb_idx.size and (pb_idx.min() < 0 or pb_idx.max() >= kept_untrimmed.size):
            raise ValueError(
                f"Photobleaching frame indices span {int(pb_idx.min())}-{int(pb_idx.max())}, "
                f"but original minus bad segments has {len(kept_untrimmed)} frames."
            )
        kept = kept_untrimmed[pb_idx]
        trimmed_frames = int(pb_idx[0]) if pb_idx.size else 0
        trim_meta = {
            "pb_trimmed_frames": trimmed_frames,
            "pb_trimmed_seconds": float(trimmed_frames / max(fps, 1e-9)),
            "pb_frame_index_source": str(home / "pipeline_results/qc/photobleaching_qc_trace_astro.npz"),
        }
    else:
        trimmed_frames = len(kept_untrimmed) - pb_frames
        if trimmed_frames >= 0:
            kept = kept_untrimmed[trimmed_frames:]
            trim_meta = {
                "pb_trimmed_frames": int(trimmed_frames),
                "pb_trimmed_seconds": float(trimmed_frames / max(fps, 1e-9)),
                "pb_frame_index_source": "inferred_from_pb_frame_count",
            }
        else:
            kept = kept_untrimmed
    if len(kept) != pb_frames:
        raise ValueError(f"PB movie has {pb_frames} frames, original minus bad segments has {len(kept)}.")
    return raw_frames, pb_frames, kept, kept / fps, trim_meta


def _load_csv_speed(home: Path, raw_frames: int, frame_times: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray, str, dict]:
    behavior_files = sorted(home.glob("imaging_*.csv"))
    if len(behavior_files) != 1:
        raise FileNotFoundError(f"Expected one imaging_<i>.csv, found {len(behavior_files)}")
    v = pd.read_csv(behavior_files[0])
    tv = v.iloc[:, 0].to_numpy(float)
    speed = v.iloc[:, 3].to_numpy(float) / 10
    speed[(tv < 1) | (speed < 0)] = 0
    speed = pd.Series(speed).rolling(int(params.get("smoothing_window_samples", 50)), center=True, min_periods=1).mean().to_numpy()
    chosen = str(params.get("chosen_alignment", "behavior_to_imaging_end"))
    fps = float(params["_fps"])
    offsets = {"behavior_to_imaging_start": tv[0], "behavior_to_imaging_end": tv[-1] - ((raw_frames - 1) / fps)}
    if chosen not in offsets:
        chosen = "behavior_to_imaging_end"
    aligned = np.interp(frame_times, tv - offsets[chosen], speed, left=np.nan, right=np.nan)
    covered = np.flatnonzero((frame_times >= (tv - offsets[chosen])[0]) & (frame_times <= (tv - offsets[chosen])[-1]))
    return aligned, covered, chosen, {"behavior_source": "csv", "behavior_file": str(behavior_files[0])}


def _has_single_csv_behavior(home: Path) -> bool:
    return len(sorted(home.glob("imaging_*.csv"))) == 1


def _load_ts_speed(home: Path, raw_frames: int, kept: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray, str, dict]:
    fps = float(params["_fps"])
    sample_rate = float(params.get("ts_sampling_rate_hz", TS_SAMPLING_RATE_HZ))
    ts = load_ts_behavior(home, "encoder", sample_rate_hz=sample_rate)
    if ts.frame_times_s is not None and ts.frame_times_s.size >= raw_frames:
        raw_time = ts.frame_times_s[:raw_frames]
        target_time = raw_time[kept]
        chosen = "ts_frameout"
    else:
        movie_duration = (raw_frames - 1) / max(fps, 1e-9)
        ts_duration = (ts.velocity.size - 1) / max(sample_rate, 1e-9)
        offset = ts_duration - movie_duration
        target_time = kept.astype(float) / fps + offset
        chosen = "ts_to_imaging_end"
    aligned, source_meta = load_ts_encoder_velocity_at_times(
        home,
        target_time,
        sample_rate_hz=sample_rate,
        wheel_diameter_cm=float(params.get("ts_wheel_diameter_cm", 14.0)),
        pulses_per_revolution=float(params.get("ts_encoder_pulses_per_revolution", 200.0)),
        invert_direction=bool(params.get("ts_invert_direction", True)),
        absolute=False,
    )
    window = int(params.get("ts_smoothing_frames", params.get("smoothing_window_samples", 50)))
    if window > 1:
        aligned = pd.Series(aligned).rolling(window, center=True, min_periods=1).mean().to_numpy()
    ts_duration = (ts.velocity.size - 1) / max(sample_rate, 1e-9)
    covered = np.flatnonzero((target_time >= 0) & (target_time <= ts_duration) & np.isfinite(aligned))
    source_meta["ts_frame_alignment"] = chosen
    return aligned, covered, chosen, {
        **source_meta,
        "behavior_file": source_meta.get("behavior_file", str(ts.h5_path)),
    }


def _split_and_save(home: Path, out: Path, params: dict, speed_trace: np.ndarray, covered: np.ndarray, chosen: str, kept: np.ndarray, fps: float, source_meta: dict) -> None:
    velocity_trace = np.asarray(speed_trace, dtype=float)
    if source_meta.get("behavior_source") == "ts":
        speed_trace = np.abs(velocity_trace)
    else:
        speed_trace = np.asarray(speed_trace, dtype=float)
        if bool(params.get("zero_negative_speed", True)):
            speed_trace[speed_trace < 0] = 0
    pb_frames = speed_trace.size
    threshold = float(params.get("speed_threshold", 5))
    min_s = float(params.get("min_epoch_duration_s", 1))
    active_raw = speed_trace >= threshold
    inactive_raw = speed_trace <= threshold
    masks = {}
    for period, raw in {"active": active_raw, "inactive": inactive_raw}.items():
        raw = raw & np.isfinite(speed_trace)
        bout = pd.Series(raw).ne(pd.Series(raw).shift(fill_value=False)).cumsum()
        dur = bout.map(bout.value_counts()).to_numpy() / fps
        masks[period] = raw & (dur >= min_s)
    covered_mask = np.zeros(pb_frames, dtype=bool)
    covered_mask[covered] = True
    active, inactive = masks["active"], masks["inactive"]
    labels = np.full(pb_frames, "no_behavior", dtype="U12")
    labels[covered_mask] = "unlabeled"
    labels[active] = "active"
    labels[inactive] = "inactive"
    start = int(covered[0]) if covered.size else 0
    np.savez_compressed(
        out / "behavioral_segements.npz",
        chosen_alignment=np.asarray(chosen),
        fps=np.asarray(fps, dtype=np.float32),
        speed_threshold=np.asarray(threshold, dtype=np.float32),
        min_epoch_duration_s=np.asarray(min_s, dtype=np.float32),
        start_frame_with_behavior=np.asarray(start, dtype=np.int32),
        start_original_frame_with_behavior=np.asarray(int(kept[start]) if kept.size else 0, dtype=np.int32),
        active_frames=np.flatnonzero(active).astype(np.int32),
        inactive_frames=np.flatnonzero(inactive).astype(np.int32),
        unlabeled_frames=np.flatnonzero(covered_mask & ~(active | inactive)).astype(np.int32),
        active_mask=active,
        inactive_mask=inactive,
        behavior_covered_mask=covered_mask,
        frame_labels=labels,
        pb_frame_times_s=(kept / fps).astype(np.float32),
        original_frame_indices=kept.astype(np.int32),
        speed_trace=speed_trace.astype(np.float32),
        velocity_trace=velocity_trace.astype(np.float32),
        behavior_source=np.asarray(source_meta.get("behavior_source", "")),
        behavior_file=np.asarray(source_meta.get("behavior_file", "")),
        ts_fallback_reason=np.asarray(source_meta.get("ts_fallback_reason", "")),
        ts_velocity_key=np.asarray(source_meta.get("ts_velocity_key", "")),
        ts_sampling_rate_hz=np.asarray(float(source_meta.get("ts_sampling_rate_hz", 0.0)), dtype=np.float32),
        ts_encoder_mode=np.asarray(source_meta.get("ts_encoder_mode", "")),
        ts_velocity_signed=np.asarray(bool(source_meta.get("ts_velocity_signed", False))),
        ts_wheel_diameter_cm=np.asarray(float(source_meta.get("ts_wheel_diameter_cm", 0.0)), dtype=np.float32),
        ts_encoder_pulses_per_revolution=np.asarray(float(source_meta.get("ts_encoder_pulses_per_revolution", 0.0)), dtype=np.float32),
        ts_counts_per_revolution=np.asarray(float(source_meta.get("ts_counts_per_revolution", 0.0)), dtype=np.float32),
        ts_direction_inverted=np.asarray(bool(source_meta.get("ts_direction_inverted", False))),
        pb_trimmed_frames=np.asarray(int(source_meta.get("pb_trimmed_frames", 0)), dtype=np.int32),
        pb_trimmed_seconds=np.asarray(float(source_meta.get("pb_trimmed_seconds", 0.0)), dtype=np.float32),
        pb_frame_index_source=np.asarray(source_meta.get("pb_frame_index_source", "")),
    )
    _save_speed_png(out / "behavior_speed_summary.png", speed_trace, fps, threshold)
    metrics = {"active_frames": int(active.sum()), "inactive_frames": int(inactive.sum()), **source_meta}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--experiment", required=True); ap.add_argument("--output", required=True); ap.add_argument("--params-json", default="{}")
    args = ap.parse_args(); home = Path(args.experiment); out = Path(args.output); out.mkdir(parents=True, exist_ok=True); params = json.loads(args.params_json or "{}")
    fps = float(get_movie_parameters(f"{home.as_posix()}/", is_dir=True)["fps"])
    params["_fps"] = fps
    raw_frames, _, kept, frame_times, trim_meta = _kept_frames(home, fps)
    if find_ts_h5_files(home):
        try:
            speed_trace, covered, chosen, source_meta = _load_ts_speed(home, raw_frames, kept, params)
        except Exception as exc:
            if not _has_single_csv_behavior(home):
                raise
            speed_trace, covered, chosen, source_meta = _load_csv_speed(home, raw_frames, frame_times, params)
            source_meta["behavior_source"] = "csv_fallback_from_ts"
            source_meta["ts_fallback_reason"] = f"{type(exc).__name__}: {exc}"
    else:
        speed_trace, covered, chosen, source_meta = _load_csv_speed(home, raw_frames, frame_times, params)
    source_meta = {**source_meta, **trim_meta}
    _split_and_save(home, out, params, speed_trace, covered, chosen, kept, fps, source_meta)


if __name__ == "__main__": main()
