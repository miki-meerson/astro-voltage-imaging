from __future__ import annotations
from collections import OrderedDict
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import threading
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
import numpy as np
from app.db.core import connect, migrate, rows, tx
from app.services.scientific_db import confirm_neuron, unconfirm_neuron
from app.services.experiments import add_experiment, remove_experiment_from_registry, scan_parent, validate_experiment
from app.services.runs import create_run, launch_run, prune_to_latest_runs, rerun_failed_steps, run_dir
from app.services.visualizations import render_sta_cell_summary
from app.settings import load_settings

STEP_LABELS = {"spike": "Spike Detection", "behavior": "Behavior Split", "anatomy": "Anatomy Crop", "sta": "STA Maps"}
STA_RESULT_TYPES = [
    ("all", "All"),
    ("simple", "Simple"),
    ("complex", "Complex"),
    ("anatomy", "Anatomy"),
    ("behavioral", "Behavioral"),
]
STATUS_PROGRESS = {
    "pending": 8,
    "queued": 18,
    "running": 55,
    "completed": 100,
    "failed": 100,
    "cancelled": 100,
    "blocked": 100,
    "invalid inputs": 100,
}
_STA_SIGNIFICANT_CACHE: dict[int, dict[int, int]] = {}
_STA_SIGNIFICANT_CACHE_LOCK = threading.Lock()
_BEHAVIOR_PAYLOAD_CACHE: dict[tuple[int, int], dict] = {}
_BEHAVIOR_PAYLOAD_CACHE_LOCK = threading.Lock()
_MOVIE_META_CACHE: dict[tuple[str, int], dict] = {}
_MOVIE_META_CACHE_LOCK = threading.Lock()
_MOVIE_ARRAY_CACHE: dict[tuple[str, int], dict] = {}
_MOVIE_ARRAY_CACHE_LOCK = threading.Lock()
_MOVIE_JPEG_CACHE: OrderedDict[tuple[str, int, int, int, str], bytes] = OrderedDict()
_MOVIE_JPEG_CACHE_LOCK = threading.Lock()
_MOVIE_JPEG_CACHE_MAX = 160
_STA_SUMMARY_WARMUPS: set[tuple[int, int]] = set()
_STA_SUMMARY_WARMUPS_LOCK = threading.Lock()


def _run_view(row, settings):
    run = dict(row)
    run["label"] = STEP_LABELS.get(run["step"], run["step"].title())
    run["progress"] = STATUS_PROGRESS.get(run["status"], 0)
    run["output_dir"] = str(run_dir(settings, int(run["id"]), bool(run["is_test"])))
    run["output_name"] = Path(run["output_dir"]).name
    run["failure_reason"] = _failure_reason(run)
    try:
        run["artifacts"] = json.loads(run["output_manifest_json"] or "[]")
    except json.JSONDecodeError:
        run["artifacts"] = []
    run["artifact_count"] = len(run["artifacts"])
    return run


def _run_view_for_neuron(row, settings, pipeline_neuron_id: str):
    run = _run_view(row, settings)
    output_name = run["output_name"]
    cell_prefixes = (
        f"{output_name}/cell_{pipeline_neuron_id}/",
        f"{output_name}/sta_algorithm_home/sta-analysis/cell_{pipeline_neuron_id}/",
    )
    if run["step"] == "behavior":
        filtered = [
            a for a in run["artifacts"]
            if Path(a.get("path", "")).name in {"behavioral_segements.npz", "behavior_speed_summary.png", "metrics.json"}
        ]
    else:
        filtered = [a for a in run["artifacts"] if str(a.get("path", "")).replace("\\", "/").startswith(cell_prefixes)]
    run["artifacts"] = filtered
    run["artifact_count"] = len(filtered)
    run["preview_artifacts"] = [
        a for a in filtered
        if str(a.get("path", "")).lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return run


def _validation_rows(report: dict):
    checks = report.get("checks", {}) if isinstance(report, dict) else {}
    result = []
    for name, info in checks.items():
        if isinstance(info, dict):
            if "exists" in info:
                result.append({"name": name.replace("_", " ").title(), "status": "OK" if info["exists"] else "Missing", "detail": Path(info.get("path", "")).name})
            elif "count" in info:
                result.append({"name": name.replace("_", " ").title(), "status": "OK" if info.get("count") == 1 else "Check", "detail": f"{info.get('count', 0)} file(s)"})
    return result


def _validation_all_ok(validation_rows: list[dict]) -> bool:
    return bool(validation_rows) and all(row.get("status") == "OK" for row in validation_rows)


def _read_tail(path: str | None, limit: int = 12000) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    data = p.read_text(encoding="utf-8", errors="replace")
    return data[-limit:]


def _run_stdout(run: dict) -> str:
    chunks = [_read_tail(run.get("stdout_path"))]
    out_dir = Path(run["output_dir"]) if run.get("output_dir") else (Path(run["stdout_path"]).parent if run.get("stdout_path") else None)
    if out_dir is not None:
        cluster = _read_tail(str(out_dir / "cluster_stdout.log"))
        if cluster:
            chunks.append("\n\n--- Cluster stdout ---\n" + cluster)
    return "".join(chunks)


def _run_stderr(run: dict) -> str:
    chunks = [_read_tail(run.get("stderr_path"))]
    out_dir = Path(run["output_dir"]) if run.get("output_dir") else (Path(run["stderr_path"]).parent if run.get("stderr_path") else None)
    if out_dir is not None:
        cluster = _read_tail(str(out_dir / "cluster_stderr.log"))
        if cluster:
            chunks.append("\n\n--- Cluster stderr ---\n" + cluster)
    return "".join(chunks)


def _failure_reason(run: dict, limit: int = 360) -> str:
    if run.get("status") != "failed":
        return ""
    stderr = _read_tail(run.get("stderr_path"), limit=2000).strip()
    if stderr:
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        return " | ".join(lines[-4:])[-limit:]
    try:
        command = json.loads(run.get("command_json") or "{}")
    except json.JSONDecodeError:
        command = run.get("command_json") or ""
    if isinstance(command, dict) and command.get("error"):
        return str(command["error"])[:limit]
    return str(command)[:limit] if command else "No error details were captured."


def _latest_completed_run(conn, experiment_id: int, step: str):
    return conn.execute(
        """
        SELECT * FROM analysis_runs
        WHERE experiment_id=? AND step=? AND is_test=0 AND status='completed'
        ORDER BY id DESC LIMIT 1
        """,
        (experiment_id, step),
    ).fetchone()


def _sta_result_type_options(source_path: str | Path | None) -> list[dict]:
    options = []
    for value, label in STA_RESULT_TYPES:
        options.append(
            {
                "value": value,
                "label": label,
                "checked": True,
                "disabled": False,
                "note": "",
            }
        )
    return options


def _sta_significant_cells(settings, conn, experiment_id: int) -> dict[int, int]:
    run = _latest_completed_run(conn, experiment_id, "sta")
    if run is None:
        return {}
    run_id = int(run["id"])
    with _STA_SIGNIFICANT_CACHE_LOCK:
        cached = _STA_SIGNIFICANT_CACHE.get(run_id)
    if cached is not None:
        return cached
    sta_dir = run_dir(settings, int(run["id"]), bool(run["is_test"])) / "sta_algorithm_home" / "sta-analysis"
    if not sta_dir.exists():
        return {}
    try:
        from utils.db_utils import collect_cluster_result_metadata

        result_df = collect_cluster_result_metadata(sta_dir)
    except Exception:
        return {}
    if result_df.empty or "cell_id" not in result_df or "n_significant_clusters" not in result_df:
        return {}
    latest = result_df.sort_values("version").groupby(["cell_id", "result_type"], as_index=False).tail(1)
    counts = latest.groupby("cell_id")["n_significant_clusters"].sum()
    result = {int(cell_id): int(count) for cell_id, count in counts.items() if int(count) > 0}
    with _STA_SIGNIFICANT_CACHE_LOCK:
        _STA_SIGNIFICANT_CACHE[run_id] = result
    return result


def _warm_sta_summaries(settings, experiment_id: int, sta_run_id: int, first_neuron_id: int | None = None) -> None:
    def _pipeline_order(neuron_row: dict) -> int:
        try:
            return int(neuron_row["pipeline_neuron_id"])
        except (TypeError, ValueError):
            return 0

    try:
        with connect(settings) as conn:
            neurons = rows(
                conn,
                """
                SELECT n.*,e.canonical_id,e.source_path,e.id AS experiment_id
                FROM neurons n
                JOIN experiments e ON e.id=n.experiment_id
                WHERE n.experiment_id=?
                ORDER BY CAST(n.pipeline_neuron_id AS INTEGER)
                """,
                (experiment_id,),
            )
            if first_neuron_id is not None:
                neurons.sort(key=lambda n: (int(n["id"]) != first_neuron_id, _pipeline_order(n)))
            for neuron_row in neurons:
                current = _latest_completed_run(conn, experiment_id, "sta")
                if current is None or int(current["id"]) != sta_run_id:
                    break
                try:
                    render_sta_cell_summary(settings, conn, neuron_row)
                except Exception:
                    continue
    finally:
        with _STA_SUMMARY_WARMUPS_LOCK:
            _STA_SUMMARY_WARMUPS.discard((experiment_id, sta_run_id))


def _start_sta_summary_warmup(settings, experiment_id: int, sta_run, first_neuron_id: int | None = None) -> None:
    if sta_run is None:
        return
    sta_run_id = int(sta_run["id"])
    key = (experiment_id, sta_run_id)
    with _STA_SUMMARY_WARMUPS_LOCK:
        if key in _STA_SUMMARY_WARMUPS:
            return
        _STA_SUMMARY_WARMUPS.add(key)
    thread = threading.Thread(
        target=_warm_sta_summaries,
        args=(settings, experiment_id, sta_run_id, first_neuron_id),
        daemon=True,
        name=f"sta-summary-warmup-{experiment_id}-{sta_run_id}",
    )
    thread.start()


def _reconstruct_behavior_speed(source_path: str | Path | None, time_s: np.ndarray, chosen: str, fps: float, run_row) -> np.ndarray:
    if not source_path or not time_s.size:
        return np.full(time_s.shape, np.nan, dtype=float)
    try:
        import pandas as pd
        import tifffile

        home = Path(source_path)
        behavior_files = sorted(home.glob("imaging_*.csv"))
        if len(behavior_files) != 1:
            return np.full(time_s.shape, np.nan, dtype=float)
        params = json.loads(run_row["parameters_json"] or "{}") if run_row is not None else {}
        behavior = pd.read_csv(behavior_files[0])
        tv = behavior.iloc[:, 0].to_numpy(float)
        raw_speed = behavior.iloc[:, 3].to_numpy(float) / 10
        raw_speed[(tv < 1) | (raw_speed < 0)] = 0
        window = int(params.get("smoothing_window_samples", 50))
        raw_speed = pd.Series(raw_speed).rolling(window, center=True, min_periods=1).mean().to_numpy()
        with tifffile.TiffFile(home / "Image_scan_1_region_0_0.tif") as tif:
            raw_frames = tif.series[0].shape[0] // 2
        offsets = {
            "behavior_to_imaging_start": tv[0],
            "behavior_to_imaging_end": tv[-1] - ((raw_frames - 1) / max(fps, 1e-9)),
        }
        offset = offsets.get(chosen, offsets["behavior_to_imaging_end"])
        return np.interp(time_s, tv - offset, raw_speed, left=np.nan, right=np.nan)
    except Exception:
        return np.full(time_s.shape, np.nan, dtype=float)


def _behavior_payload(settings, run_row, source_path: str | Path | None = None) -> dict:
    if run_row is None:
        return {"available": False, "message": "No completed behavior run is available."}
    path = run_dir(settings, int(run_row["id"]), bool(run_row["is_test"])) / "behavioral_segements.npz"
    if not path.exists():
        return {"available": False, "message": "Behavior segments file was not found."}
    cache_key = (int(run_row["id"]), int(path.stat().st_mtime_ns))
    with _BEHAVIOR_PAYLOAD_CACHE_LOCK:
        cached = _BEHAVIOR_PAYLOAD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    max_points = 3000
    with np.load(path, allow_pickle=True) as d:
        time_s = np.asarray(d["pb_frame_times_s"], dtype=float) if "pb_frame_times_s" in d else np.arange(0)
        active = np.asarray(d["active_mask"], dtype=bool) if "active_mask" in d else np.zeros(time_s.shape, dtype=bool)
        inactive = np.asarray(d["inactive_mask"], dtype=bool) if "inactive_mask" in d else np.zeros(time_s.shape, dtype=bool)
        covered = np.asarray(d["behavior_covered_mask"], dtype=bool) if "behavior_covered_mask" in d else np.ones(time_s.shape, dtype=bool)
        threshold = float(np.asarray(d["speed_threshold"]).item()) if "speed_threshold" in d else 0.0
        fps = float(np.asarray(d["fps"]).item()) if "fps" in d else 1.0
        chosen = str(np.asarray(d["chosen_alignment"]).item()) if "chosen_alignment" in d else ""
        behavior_source = str(np.asarray(d["behavior_source"]).item()) if "behavior_source" in d else ""
        ts_fallback_reason = str(np.asarray(d["ts_fallback_reason"]).item()) if "ts_fallback_reason" in d else ""
        ts_velocity_key = str(np.asarray(d["ts_velocity_key"]).item()) if "ts_velocity_key" in d else ""
        behavior_file = str(np.asarray(d["behavior_file"]).item()) if "behavior_file" in d else ""
        if "speed_trace" in d:
            speed = np.asarray(d["speed_trace"], dtype=float)
            trace_source = "saved"
        else:
            speed = _reconstruct_behavior_speed(source_path, time_s, chosen, fps, run_row)
            trace_source = "reconstructed" if np.isfinite(speed).any() else "missing"
        if "velocity_trace" in d:
            velocity = np.asarray(d["velocity_trace"], dtype=float)
        else:
            velocity = speed.copy()
        ts_encoder_mode = str(np.asarray(d["ts_encoder_mode"]).item()) if "ts_encoder_mode" in d else ""
        ts_velocity_signed = bool(np.asarray(d["ts_velocity_signed"]).item()) if "ts_velocity_signed" in d else False
        ts_wheel_diameter_cm = float(np.asarray(d["ts_wheel_diameter_cm"]).item()) if "ts_wheel_diameter_cm" in d else 0.0
        ts_encoder_pulses_per_revolution = float(np.asarray(d["ts_encoder_pulses_per_revolution"]).item()) if "ts_encoder_pulses_per_revolution" in d else 0.0
        ts_counts_per_revolution = float(np.asarray(d["ts_counts_per_revolution"]).item()) if "ts_counts_per_revolution" in d else 0.0
        ts_direction_inverted = bool(np.asarray(d["ts_direction_inverted"]).item()) if "ts_direction_inverted" in d else False
    n = int(min(time_s.size, speed.size, velocity.size, active.size, inactive.size, covered.size))
    time_s, speed, velocity, active, inactive, covered = time_s[:n], speed[:n], velocity[:n], active[:n], inactive[:n], covered[:n]
    state = np.zeros(n, dtype=np.uint8)
    state[covered] = 1
    state[inactive] = 2
    state[active] = 3
    segments = []
    labels_by_state = {0: "no_behavior", 1: "unlabeled", 2: "inactive", 3: "active"}
    if n:
        starts = np.flatnonzero(np.r_[True, state[1:] != state[:-1]])
        stops = np.r_[starts[1:], n]
        for start_idx, stop_idx in zip(starts, stops):
            label = labels_by_state[int(state[start_idx])]
            if label == "no_behavior":
                continue
            segments.append(
                {
                    "t0": float(time_s[int(start_idx)]),
                    "t1": float(time_s[max(int(stop_idx) - 1, int(start_idx))]),
                    "label": label,
                }
            )
    step = max(1, int(np.ceil(n / max_points))) if n else 1
    display_idx = np.arange(0, n, step, dtype=int)
    display_speed = np.asarray(speed[display_idx], dtype=float)
    display_velocity = np.asarray(velocity[display_idx], dtype=float)
    speed_values = [
        round(float(value), 4) if np.isfinite(value) else None
        for value in display_speed
    ]
    velocity_values = [
        round(float(value), 4) if np.isfinite(value) else None
        for value in display_velocity
    ]
    payload = {
        "available": True,
        "run_id": int(run_row["id"]),
        "time_s": np.round(time_s[display_idx], 4).tolist(),
        "speed": speed_values,
        "velocity": velocity_values,
        "segments": segments,
        "threshold": threshold,
        "fps": fps,
        "chosen_alignment": chosen,
        "trace_source": trace_source,
        "behavior_source": behavior_source,
        "ts_fallback_reason": ts_fallback_reason,
        "behavior_file": Path(behavior_file).name if behavior_file else "",
        "ts_velocity_key": ts_velocity_key,
        "ts_encoder_mode": ts_encoder_mode,
        "ts_velocity_signed": ts_velocity_signed,
        "ts_wheel_diameter_cm": ts_wheel_diameter_cm,
        "ts_encoder_pulses_per_revolution": ts_encoder_pulses_per_revolution,
        "ts_counts_per_revolution": ts_counts_per_revolution,
        "ts_direction_inverted": ts_direction_inverted,
        "velocity_units": "cm/s",
        "total_frames": n,
        "display_step": int(step),
        "active_frames": int(np.sum(active)),
        "inactive_frames": int(np.sum(inactive)),
        "unlabeled_frames": int(np.sum(covered & ~(active | inactive))),
        "no_behavior_frames": int(np.sum(~covered)),
    }
    with _BEHAVIOR_PAYLOAD_CACHE_LOCK:
        _BEHAVIOR_PAYLOAD_CACHE.clear()
        _BEHAVIOR_PAYLOAD_CACHE[cache_key] = payload
    return payload


def _pb_movie_path(source_path: str | Path | None, kind: str) -> Path | None:
    if not source_path:
        return None
    names = {
        "astro": "astro_photobleaching_corrected.tif",
        "neuron": "neuron_photobleaching_corrected.tif",
    }
    name = names.get(kind)
    if name is None:
        return None
    return Path(source_path) / "pipeline_results" / "pb_corrected" / name


def _movie_series_shape(series) -> tuple[int, int, int]:
    shape = tuple(int(v) for v in getattr(series, "shape", ()) or ())
    axes = str(getattr(series, "axes", "") or "").upper()
    if len(shape) == 2:
        return 1, shape[0], shape[1]
    if len(shape) < 3:
        raise ValueError(f"Unsupported movie shape: {shape}")
    if axes and len(axes) == len(shape):
        t_axis = next((i for i, axis in enumerate(axes) if axis in {"T", "I", "Q", "Z"}), 0)
        y_axis = axes.find("Y")
        x_axis = axes.find("X")
        if y_axis >= 0 and x_axis >= 0:
            return shape[t_axis], shape[y_axis], shape[x_axis]
    return shape[0], shape[-2], shape[-1]


def _movie_metadata(path: Path, fps: float) -> dict:
    stat_key = int(path.stat().st_mtime_ns)
    cache_key = (str(path), stat_key)
    with _MOVIE_META_CACHE_LOCK:
        cached = _MOVIE_META_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        n_frames, height, width = _movie_series_shape(series)
        dtype = str(series.dtype)
        axes = str(getattr(series, "axes", "") or "")
    payload = {
        "available": True,
        "frames": int(n_frames),
        "height": int(height),
        "width": int(width),
        "fps": float(fps),
        "duration_s": float(n_frames / max(fps, 1e-9)),
        "dtype": dtype,
        "axes": axes,
        "mtime_ns": stat_key,
        "file_name": path.name,
    }
    with _MOVIE_META_CACHE_LOCK:
        _MOVIE_META_CACHE.clear()
        _MOVIE_META_CACHE[cache_key] = dict(payload)
    return payload


def _movie_array_payload(path: Path, invert_sign: bool = False) -> dict:
    import tifffile

    stat_key = int(path.stat().st_mtime_ns)
    cache_key = (str(path), stat_key, bool(invert_sign))
    with _MOVIE_ARRAY_CACHE_LOCK:
        cached = _MOVIE_ARRAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = str(getattr(series, "axes", "") or "").upper()
        movie = np.asarray(series.asarray(out="memmap"))
    if movie.ndim == 2:
        movie = movie[np.newaxis, :, :]
    elif movie.ndim >= 3:
        if axes and len(axes) == movie.ndim:
            t_axis = next((i for i, axis in enumerate(axes) if axis in {"T", "I", "Q", "Z"}), 0)
            y_axis = axes.find("Y")
            x_axis = axes.find("X")
            if y_axis >= 0 and x_axis >= 0:
                movie = np.moveaxis(movie, (t_axis, y_axis, x_axis), (0, 1, 2))
        while movie.ndim > 3:
            movie = movie[:, 0]

    n_frames = int(movie.shape[0])
    sample_frames = np.linspace(0, max(0, n_frames - 1), min(24, n_frames), dtype=int) if n_frames else np.array([], dtype=int)
    samples = []
    for idx in sample_frames:
        frame = np.asarray(movie[int(idx)])
        if invert_sign:
            frame = -frame.astype(np.float32, copy=False)
        y_step = max(1, frame.shape[0] // 220)
        x_step = max(1, frame.shape[1] // 220)
        samples.append(np.asarray(frame[::y_step, ::x_step], dtype=np.float32).ravel())
    if samples:
        values = np.concatenate(samples)
        values = values[np.isfinite(values)]
    else:
        values = np.empty(0, dtype=np.float32)
    if values.size:
        lo, hi = (float(v) for v in np.nanpercentile(values, [1, 99.7]))
    else:
        lo, hi = 0.0, 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    payload = {"movie": movie, "mtime_ns": stat_key, "contrast": (lo, hi), "invert_sign": bool(invert_sign)}
    with _MOVIE_ARRAY_CACHE_LOCK:
        _MOVIE_ARRAY_CACHE.clear()
        _MOVIE_ARRAY_CACHE[cache_key] = payload
    return payload


def _read_movie_frame(path: Path, frame_index: int, invert_sign: bool = False) -> tuple[np.ndarray, int, tuple[float, float]]:
    payload = _movie_array_payload(path, invert_sign)
    movie = payload["movie"]
    frame_index = min(max(0, int(frame_index)), max(0, int(movie.shape[0]) - 1))
    frame = np.asarray(movie[frame_index])
    if invert_sign:
        frame = -frame.astype(np.float32, copy=False)
    return frame, int(payload["mtime_ns"]), payload["contrast"]


def _movie_frame_response(path: Path, frame_index: int, max_size: int, contrast: str, invert_sign: bool = False):
    from PIL import Image

    frame, mtime_ns, contrast_limits = _read_movie_frame(path, frame_index, invert_sign)
    cache_key = (str(path), mtime_ns, int(frame_index), int(max_size), str(contrast), bool(invert_sign))
    with _MOVIE_JPEG_CACHE_LOCK:
        cached = _MOVIE_JPEG_CACHE.get(cache_key)
        if cached is not None:
            _MOVIE_JPEG_CACHE.move_to_end(cache_key)
            return send_file(BytesIO(cached), mimetype="image/jpeg", max_age=600)

    img = np.asarray(frame, dtype=np.float32)
    if contrast == "full":
        finite = img[np.isfinite(img)]
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    else:
        lo, hi = contrast_limits
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((img - lo) / (hi - lo), 0, 1)
    scaled = (scaled * 255).astype(np.uint8)
    image = Image.fromarray(scaled, mode="L")
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
    bio = BytesIO()
    image.save(bio, format="JPEG", quality=82, optimize=False)
    data = bio.getvalue()
    with _MOVIE_JPEG_CACHE_LOCK:
        _MOVIE_JPEG_CACHE[cache_key] = data
        _MOVIE_JPEG_CACHE.move_to_end(cache_key)
        while len(_MOVIE_JPEG_CACHE) > _MOVIE_JPEG_CACHE_MAX:
            _MOVIE_JPEG_CACHE.popitem(last=False)
    return send_file(BytesIO(data), mimetype="image/jpeg", max_age=600)


def _spike_cell_dir(settings, run_row, pipeline_neuron_id: str) -> Path:
    return run_dir(settings, int(run_row["id"]), bool(run_row["is_test"])) / f"cell_{pipeline_neuron_id}"


def _spike_review_info(settings, spike_run, pipeline_neuron_id: str) -> dict:
    if spike_run is None:
        return {"reviewed": False, "n_peaks": 0, "n_simple": 0, "n_complex": 0, "n_excluded": 0}
    cell_dir = _spike_cell_dir(settings, spike_run, pipeline_neuron_id)
    payload = _load_spike_payload(cell_dir / "simple_complex_spike_labels.npz")
    labels = payload["labels"].astype(str)
    return {
        "reviewed": (cell_dir / "manual_spike_review.json").exists(),
        "n_peaks": int(payload["peaks_all"].size),
        "n_simple": int(np.sum(labels == "simple")),
        "n_complex": int(np.sum(labels == "complex")),
        "n_excluded": int(np.sum(labels == "exclude")),
    }


def _load_spike_payload(path: Path) -> dict:
    if not path.exists():
        return {"peaks_all": np.empty(0, dtype=np.int32), "labels": np.empty(0, dtype="<U8")}
    with np.load(path, allow_pickle=True) as d:
        peaks = np.asarray(d["peaks_all"], dtype=np.int32) if "peaks_all" in d else np.empty(0, dtype=np.int32)
        labels = np.asarray(d["labels"]).astype(str) if "labels" in d else np.full(peaks.shape, "simple")
    order = np.argsort(peaks)
    return {"peaks_all": peaks[order], "labels": labels[order]}


def _spike_detection_params(run_row, cell_dir: Path | None = None) -> dict:
    try:
        params = json.loads(run_row["parameters_json"] or "{}") if run_row is not None else {}
    except json.JSONDecodeError:
        params = {}
    result = {
        "mad_factor": float(params.get("mad_factor", 2.8)),
        "min_amplitude": float(params.get("min_amplitude", 0.30)),
        "min_width": float(params.get("min_width", 18.0)),
        "min_subthreshold": float(params.get("min_subthreshold", 0.10)),
        "min_ratio": float(params.get("min_ratio", 0.30)),
        "duplicate_window_frames": int(params.get("duplicate_window_frames", 30)),
        "align_radius_frames": int(params.get("align_radius_frames", 5)),
        "min_distance": int(params.get("min_distance", 0)),
        "highpass_cutoff_hz": float(params.get("highpass_cutoff_hz", 0.07)),
    }
    params_path = cell_dir / "peak_detection_params.npz" if cell_dir is not None else None
    if params_path is not None and params_path.exists():
        with np.load(params_path, allow_pickle=True) as d:
            if "mad_threshold_factor" in d:
                result["mad_factor"] = float(np.asarray(d["mad_threshold_factor"]).item())
            if "min_distance" in d:
                result["min_distance"] = int(np.asarray(d["min_distance"]).item())
            if "highpass_cutoff_hz" in d:
                result["highpass_cutoff_hz"] = float(np.asarray(d["highpass_cutoff_hz"]).item())
    return result


def _bounded_float(data: dict, name: str, default: float, lo: float, hi: float) -> float:
    value = float(data.get(name, default))
    if not np.isfinite(value):
        value = default
    return float(min(hi, max(lo, value)))


def _bounded_int(data: dict, name: str, default: int, lo: int, hi: int) -> int:
    value = int(data.get(name, default))
    return int(min(hi, max(lo, value)))


def _save_spike_payload(cell_dir: Path, peaks: np.ndarray, labels: np.ndarray) -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(peaks)
    peaks = np.asarray(peaks, dtype=np.int32)[order]
    labels = np.asarray(labels).astype("<U8")[order]
    existing_path = cell_dir / "simple_complex_spike_labels.npz"
    payload = {}
    if existing_path.exists():
        with np.load(existing_path, allow_pickle=True) as d:
            payload = {key: np.asarray(d[key]) for key in d.files}
    payload.update({
        "peaks_all": peaks,
        "labels": labels,
        "simple_peaks": peaks[labels == "simple"],
        "complex_peaks": peaks[labels == "complex"],
        "excluded_peaks": peaks[labels == "exclude"],
        "manual_reviewed": np.asarray(True),
    })
    for name in ("simple_complex_spike_labels.npz", "simple_complex_peaks_for_analysis.npz"):
        np.savez_compressed(cell_dir / name, **payload)
    (cell_dir / "manual_spike_review.json").write_text(
        json.dumps(
            {
                "peaks_all": peaks.tolist(),
                "labels": labels.tolist(),
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def create_app() -> Flask:
    settings = load_settings(); migrate(settings); prune_to_latest_runs(settings); app = Flask(__name__)
    @app.get("/")
    def index():
        favorites_only = request.args.get("favorites") == "1"
        with connect(settings) as conn:
            experiments = rows(conn, """
                SELECT e.*,
                       COALESCE(n.neuron_count, 0) AS neuron_count,
                       r.running_step,
                       r.pending_step,
                       r.failed_step,
                       COALESCE(r.completed_steps, 0) AS completed_steps,
                       CASE
                         WHEN r.running_step IS NOT NULL THEN 'running'
                         WHEN r.pending_step IS NOT NULL THEN 'pending'
                         WHEN r.failed_step IS NOT NULL THEN 'failed'
                         WHEN COALESCE(r.completed_steps, 0) >= 4 THEN 'completed'
                         ELSE 'none'
                       END AS job_status,
                       CASE
                         WHEN r.running_step IS NOT NULL THEN r.running_step
                         WHEN r.pending_step IS NOT NULL THEN r.pending_step
                         WHEN r.failed_step IS NOT NULL THEN r.failed_step
                         ELSE ''
                       END AS job_step
                FROM experiments e
                LEFT JOIN (
                    SELECT experiment_id, COUNT(*) AS neuron_count
                    FROM neurons
                    GROUP BY experiment_id
                ) n ON n.experiment_id=e.id
                LEFT JOIN (
                    SELECT r.experiment_id,
                           MIN(CASE WHEN r.status='running' THEN r.step END) AS running_step,
                           MIN(CASE WHEN r.status IN ('pending','queued') THEN r.step END) AS pending_step,
                           MIN(CASE WHEN r.status='failed' THEN r.step END) AS failed_step,
                           COUNT(DISTINCT CASE WHEN r.status='completed' THEN r.step END) AS completed_steps
                    FROM analysis_runs r
                    JOIN (
                        SELECT experiment_id,step,MAX(id) AS run_id
                        FROM analysis_runs
                        WHERE is_test=0
                        GROUP BY experiment_id,step
                    ) latest ON latest.run_id=r.id
                    GROUP BY r.experiment_id
                ) r ON r.experiment_id=e.id
                WHERE (? = 0 OR e.is_favorite = 1)
                ORDER BY substr(e.recording_date, 7, 4) DESC,
                         substr(e.recording_date, 4, 2) DESC,
                         substr(e.recording_date, 1, 2) DESC,
                         e.canonical_id ASC
            """, (1 if favorites_only else 0,))
            last = conn.execute("SELECT * FROM scan_history ORDER BY id DESC LIMIT 1").fetchone()
        active = any(e.get("job_status") in {"running", "pending"} for e in experiments)
        return render_template("index.html", experiments=experiments, settings=settings, last_scan=dict(last) if last else None, active=active, favorites_only=favorites_only, step_labels=STEP_LABELS)
    @app.post("/experiments/add")
    def add():
        try:
            exp_id = add_experiment(request.form["path"].strip().strip('"'), settings)
            with connect(settings) as conn:
                has_runs = conn.execute("SELECT 1 FROM analysis_runs WHERE experiment_id=? LIMIT 1", (exp_id,)).fetchone() is not None
            if not has_runs:
                for step, params in settings.raw.get("defaults", {}).items():
                    if step in {"spike","behavior","anatomy"}:
                        rid = create_run(settings, exp_id, step, params, backend="local"); launch_run(settings, rid)
                return redirect(url_for("experiment", experiment_id=exp_id))
            return redirect(url_for("experiment", experiment_id=exp_id, notice="already_registered"))
        except Exception as exc: return render_template("error.html", title="Add Failed", message=str(exc)), 400
    @app.get("/browse")
    def browse():
        current = Path(request.args.get("path") or settings.scan_parent)
        if not current.exists():
            current = settings.scan_parent
        if current.is_file():
            current = current.parent
        dirs = []
        try:
            dirs = sorted([p for p in current.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
        except OSError as exc:
            return render_template("error.html", title="Browse Failed", message=str(exc)), 400
        parent = current.parent if current.parent != current else None
        return render_template("browse.html", current=current, parent=parent, dirs=dirs)
    @app.post("/scan")
    def scan():
        try:
            parent = request.form.get("parent") or str(settings.scan_parent)
            threading.Thread(target=scan_parent, args=(parent, settings), daemon=True, name="astro-scan").start()
            return redirect(url_for("index"))
        except Exception as exc: return render_template("error.html", title="Scan Failed", message=str(exc)), 400
    @app.post("/runs/rerun-failed")
    def rerun_failed():
        try:
            step = request.form.get("step") or None
            created = rerun_failed_steps(settings, step)
            return redirect(url_for("index", rerun_created=len(created)))
        except Exception as exc:
            return render_template("error.html", title="Rerun Failed", message=str(exc)), 400
    @app.post("/experiments/<int:experiment_id>/remove")
    def remove_experiment(experiment_id):
        try:
            remove_experiment_from_registry(experiment_id, settings)
            return redirect(url_for("index"))
        except Exception as exc:
            return render_template("error.html", title="Remove Failed", message=str(exc)), 400
    @app.post("/experiments/<int:experiment_id>/favorite")
    def favorite_experiment(experiment_id):
        try:
            favorite = 1 if request.form.get("favorite") == "1" else 0
            with tx(settings) as conn:
                updated = conn.execute(
                    "UPDATE experiments SET is_favorite=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (favorite, experiment_id),
                ).rowcount
                if not updated:
                    return render_template("error.html", title="Missing Experiment", message="Experiment not found"), 404
            return redirect(request.form.get("next") or url_for("index"))
        except Exception as exc:
            return render_template("error.html", title="Favorite Failed", message=str(exc)), 400
    @app.get("/experiments/<int:experiment_id>")
    def experiment(experiment_id):
        selected_neuron_id = request.args.get("neuron", "")
        try:
            warmup_first_neuron_id = int(selected_neuron_id) if selected_neuron_id else None
        except ValueError:
            warmup_first_neuron_id = None
        with connect(settings) as conn:
            exp = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if exp is None:
                return render_template("error.html", title="Missing Experiment", message="Experiment not found"), 404
            neurons = rows(conn, "SELECT id,pipeline_neuron_id FROM neurons WHERE experiment_id=? ORDER BY CAST(pipeline_neuron_id AS INTEGER)", (experiment_id,))
            for neuron_row in neurons:
                neuron_row["significant_cluster_count"] = 0
                neuron_row["has_significant_clusters"] = False
            runs = [_run_view(r, settings) for r in conn.execute("SELECT * FROM analysis_runs WHERE experiment_id=? ORDER BY created_at DESC", (experiment_id,))]
            latest_anatomy = _latest_completed_run(conn, experiment_id, "anatomy")
            latest_sta = _latest_completed_run(conn, experiment_id, "sta")
            confirmed_count = conn.execute("SELECT COUNT(*) AS n FROM confirmed_neurons WHERE experiment_id=?", (experiment_id,)).fetchone()["n"]
        _start_sta_summary_warmup(settings, experiment_id, latest_sta, warmup_first_neuron_id)
        active = any(r["status"] in {"pending", "queued", "running"} for r in runs)
        report = json.loads(exp["validation_report"] or "{}")
        validation_rows = _validation_rows(report)
        return render_template(
            "experiment.html",
            exp=dict(exp),
            neurons=neurons,
            runs=runs,
            anatomy_run=_run_view(latest_anatomy, settings) if latest_anatomy else None,
            validation_rows=validation_rows,
            validation_all_ok=_validation_all_ok(validation_rows) and exp["validation_state"] == "valid",
            active=active,
            step_labels=STEP_LABELS,
            sta_result_types=_sta_result_type_options(exp["source_path"]),
            notice=request.args.get("notice"),
            confirmed_count=confirmed_count,
            selected_neuron_id=selected_neuron_id,
        )
    @app.post("/experiments/<int:experiment_id>/validate")
    def validate_again(experiment_id):
        with tx(settings) as conn:
            exp = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone(); _, report = validate_experiment(exp["source_path"], settings)
            conn.execute("UPDATE experiments SET validation_state=?,validation_report=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (report["state"], json.dumps(report), experiment_id))
        return redirect(url_for("experiment", experiment_id=experiment_id))
    @app.post("/experiments/<int:experiment_id>/run")
    def run_step(experiment_id):
        try:
            step = request.form["step"]
            params = json.loads(request.form.get("params") or "{}") if request.form.get("params") else settings.raw.get("defaults", {}).get(step, {})
            params = dict(params)
            if step == "sta":
                requested_types = request.form.getlist("result_types")
                valid_types = {name for name, _ in STA_RESULT_TYPES}
                result_types = [name for name in requested_types if name in valid_types]
                if not result_types:
                    raise ValueError("Choose at least one STA result type.")
                params["result_types"] = result_types
            backend = "cluster" if step == "sta" else "local"
            rid = create_run(settings, experiment_id, step, params, backend=backend, is_test=False); launch_run(settings, rid)
            return redirect(url_for("experiment", experiment_id=experiment_id, new_run=rid) + "#runs")
        except Exception as exc: return render_template("error.html", title="Run Failed", message=str(exc)), 400
    @app.get("/experiments/<int:experiment_id>/behavior/data")
    def behavior_data(experiment_id):
        with connect(settings) as conn:
            run = _latest_completed_run(conn, experiment_id, "behavior")
            exp = conn.execute("SELECT source_path FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        return jsonify(_behavior_payload(settings, run, exp["source_path"] if exp else None))
    @app.get("/experiments/<int:experiment_id>/movie/meta")
    def experiment_movie_meta(experiment_id):
        kind = request.args.get("kind", "astro")
        with connect(settings) as conn:
            exp = conn.execute("SELECT source_path FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if exp is None:
            return jsonify({"available": False, "message": "Experiment not found."}), 404
        path = _pb_movie_path(exp["source_path"], kind)
        if path is None:
            return jsonify({"available": False, "message": "Unknown movie type."}), 400
        if not path.exists():
            return jsonify({"available": False, "message": f"{path.name} was not found."})
        try:
            from utils.movie_utils import get_movie_parameters

            fps = float(get_movie_parameters(f"{Path(exp['source_path']).as_posix()}/", is_dir=True)["fps"])
            payload = _movie_metadata(path, fps)
            payload["kind"] = kind
            payload["gevi_flipped"] = kind == "astro"
            return jsonify(payload)
        except Exception as exc:
            return jsonify({"available": False, "message": str(exc)}), 500
    @app.get("/experiments/<int:experiment_id>/movie/frame")
    def experiment_movie_frame(experiment_id):
        kind = request.args.get("kind", "astro")
        try:
            frame_index = max(0, int(request.args.get("frame", "0")))
            max_size = min(1400, max(240, int(request.args.get("max_size", "720"))))
        except ValueError:
            return jsonify({"error": "Invalid frame or max_size."}), 400
        contrast = request.args.get("contrast", "auto")
        with connect(settings) as conn:
            exp = conn.execute("SELECT source_path FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if exp is None:
            return jsonify({"error": "Experiment not found."}), 404
        path = _pb_movie_path(exp["source_path"], kind)
        if path is None:
            return jsonify({"error": "Unknown movie type."}), 400
        if not path.exists():
            return jsonify({"error": f"{path.name} was not found."}), 404
        try:
            return _movie_frame_response(path, frame_index, max_size, contrast, invert_sign=(kind == "astro"))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
    @app.get("/experiments/<int:experiment_id>/sta-significant-cells/data")
    def sta_significant_cells_data(experiment_id):
        with connect(settings) as conn:
            exists = conn.execute("SELECT 1 FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if exists is None:
                return jsonify({"error": "Experiment not found"}), 404
            cells = _sta_significant_cells(settings, conn, experiment_id)
        return jsonify({"cells": cells})
    @app.get("/experiments/<int:experiment_id>/cell-review/data")
    def cell_review_data(experiment_id):
        with connect(settings) as conn:
            exists = conn.execute("SELECT 1 FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if exists is None:
                return jsonify({"error": "Experiment not found"}), 404
            spike_run = _latest_completed_run(conn, experiment_id, "spike")
            neurons = rows(conn, "SELECT id,pipeline_neuron_id FROM neurons WHERE experiment_id=? ORDER BY CAST(pipeline_neuron_id AS INTEGER)", (experiment_id,))
        review = {}
        for n in neurons:
            review[str(n["id"])] = _spike_review_info(settings, spike_run, str(n["pipeline_neuron_id"]))
        reviewed_count = sum(1 for info in review.values() if info["reviewed"])
        return jsonify(
            {
                "has_spike_run": spike_run is not None,
                "reviewed_count": reviewed_count,
                "total_count": len(neurons),
                "review": review,
            }
        )
    @app.get("/neurons/<int:neuron_id>")
    def neuron(neuron_id):
        with connect(settings) as conn:
            n = conn.execute("SELECT n.*,e.canonical_id,e.id AS experiment_id FROM neurons n JOIN experiments e ON e.id=n.experiment_id WHERE n.id=?", (neuron_id,)).fetchone()
            if n is None:
                return render_template("error.html", title="Missing Cell", message="Cell not found"), 404
            runs = [
                _run_view_for_neuron(r, settings, n["pipeline_neuron_id"])
                for r in conn.execute("SELECT * FROM analysis_runs WHERE experiment_id=? ORDER BY step,id DESC", (n["experiment_id"],))
            ]
            confirmed = conn.execute("SELECT * FROM confirmed_neurons WHERE experiment_id=? AND pipeline_neuron_id=?", (n["experiment_id"], n["pipeline_neuron_id"])).fetchone()
        active = any(r["status"] in {"pending", "queued", "running"} for r in runs)
        runs_by_step = {}
        for r in runs:
            runs_by_step.setdefault(r["step"], r)
        return render_template(
            "neuron.html",
            neuron=dict(n),
            runs_by_step=runs_by_step,
            active=active,
            step_labels=STEP_LABELS,
            confirmed=dict(confirmed) if confirmed else None,
            embedded=request.args.get("embedded") == "1",
        )
    @app.get("/neurons/<int:neuron_id>/sta-summary.png")
    def neuron_sta_summary(neuron_id):
        with connect(settings) as conn:
            n = conn.execute("SELECT n.*,e.canonical_id,e.source_path,e.id AS experiment_id FROM neurons n JOIN experiments e ON e.id=n.experiment_id WHERE n.id=?", (neuron_id,)).fetchone()
            if n is None:
                return render_template("error.html", title="Missing Cell", message="Cell not found"), 404
            img = render_sta_cell_summary(settings, conn, n)
        return send_file(img, mimetype="image/png")
    @app.post("/neurons/<int:neuron_id>/confirm")
    def confirm_cell(neuron_id):
        try:
            confirm_neuron(settings, neuron_id, request.form.get("reviewer_note", ""))
            return redirect(url_for("neuron", neuron_id=neuron_id, embedded=request.form.get("embedded") or None, confirmed="1"))
        except Exception as exc:
            return render_template("error.html", title="Confirm Failed", message=str(exc)), 400

    @app.post("/neurons/<int:neuron_id>/unconfirm")
    def unconfirm_cell(neuron_id):
        try:
            unconfirm_neuron(settings, neuron_id)
            return redirect(url_for("neuron", neuron_id=neuron_id, embedded=request.form.get("embedded") or None, unconfirmed="1"))
        except Exception as exc:
            return render_template("error.html", title="Unconfirm Failed", message=str(exc)), 400
    @app.post("/neurons/<int:neuron_id>/review/mark")
    def mark_cell_reviewed(neuron_id):
        with connect(settings) as conn:
            n = conn.execute("SELECT n.*,e.id AS experiment_id FROM neurons n JOIN experiments e ON e.id=n.experiment_id WHERE n.id=?", (neuron_id,)).fetchone()
            if n is None:
                return jsonify({"error": "Cell not found"}), 404
            spike_run = _latest_completed_run(conn, int(n["experiment_id"]), "spike")
        if spike_run is None:
            return jsonify({"error": "No completed spike run."}), 400
        cell_dir = _spike_cell_dir(settings, spike_run, n["pipeline_neuron_id"])
        payload = _load_spike_payload(cell_dir / "simple_complex_spike_labels.npz")
        _save_spike_payload(cell_dir, payload["peaks_all"], payload["labels"])
        return jsonify({"ok": True, **_spike_review_info(settings, spike_run, n["pipeline_neuron_id"])})
    @app.get("/neurons/<int:neuron_id>/spikes")
    def spike_editor(neuron_id):
        with connect(settings) as conn:
            n = conn.execute("SELECT n.*,e.canonical_id,e.id AS experiment_id FROM neurons n JOIN experiments e ON e.id=n.experiment_id WHERE n.id=?", (neuron_id,)).fetchone()
            if n is None:
                return render_template("error.html", title="Missing Cell", message="Cell not found"), 404
            spike_run = _latest_completed_run(conn, int(n["experiment_id"]), "spike")
        if spike_run is None:
            return render_template("error.html", title="No Spike Run", message="Run spike detection before editing spikes."), 400
        return render_template("spikes.html", neuron=dict(n), embedded=request.args.get("embedded") == "1")
    @app.get("/neurons/<int:neuron_id>/spikes/data")
    def spike_data(neuron_id):
        with connect(settings) as conn:
            n = conn.execute("SELECT n.*,e.source_path,e.id AS experiment_id FROM neurons n JOIN experiments e ON e.id=n.experiment_id WHERE n.id=?", (neuron_id,)).fetchone()
            if n is None:
                return jsonify({"error": "Cell not found"}), 404
            spike_run = _latest_completed_run(conn, int(n["experiment_id"]), "spike")
        if spike_run is None:
            return jsonify({"error": "No completed spike run."}), 400
        from utils.movie_utils import get_movie_parameters
        from utils.spikes_utils import _suite2p_trace_dff
        from utils.suite2p_utils import load_suite2p_context

        suite2p_path = Path(n["source_path"]) / "pipeline_results" / "for_suite2p" / "suite2p" / "plane0"
        suite2p_ctx = load_suite2p_context(suite2p_path)
        trace = np.asarray(_suite2p_trace_dff(int(n["pipeline_neuron_id"]), suite2p_ctx=suite2p_ctx), dtype=float)
        fps = float(get_movie_parameters(f"{Path(n['source_path']).as_posix()}/", is_dir=True)["fps"])
        labels_path = _spike_cell_dir(settings, spike_run, n["pipeline_neuron_id"]) / "simple_complex_spike_labels.npz"
        payload = _load_spike_payload(labels_path)
        params = _spike_detection_params(spike_run, labels_path.parent)
        return jsonify(
            {
                "trace": np.round(trace, 5).tolist(),
                "fps": fps,
                "peaks": payload["peaks_all"].astype(int).tolist(),
                "labels": payload["labels"].astype(str).tolist(),
                "params": params,
            }
        )
    @app.post("/neurons/<int:neuron_id>/spikes/detect")
    def detect_spike_data(neuron_id):
        data = request.get_json(force=True) or {}
        params = {
            "mad_factor": _bounded_float(data, "mad_factor", 2.8, 0.5, 8.0),
            "min_amplitude": _bounded_float(data, "min_amplitude", 0.30, 0.0, 2.0),
            "min_width": _bounded_float(data, "min_width", 18.0, 1.0, 120.0),
            "min_subthreshold": _bounded_float(data, "min_subthreshold", 0.10, 0.0, 1.0),
            "min_ratio": _bounded_float(data, "min_ratio", 0.30, 0.0, 2.0),
            "duplicate_window_frames": _bounded_int(data, "duplicate_window_frames", 30, 1, 240),
            "align_radius_frames": _bounded_int(data, "align_radius_frames", 5, 0, 80),
            "min_distance": _bounded_int(data, "min_distance", 0, 0, 500),
            "highpass_cutoff_hz": _bounded_float(data, "highpass_cutoff_hz", 0.07, 0.0, 1.0),
        }
        with connect(settings) as conn:
            n = conn.execute("SELECT n.*,e.source_path,e.id AS experiment_id FROM neurons n JOIN experiments e ON e.id=n.experiment_id WHERE n.id=?", (neuron_id,)).fetchone()
            if n is None:
                return jsonify({"error": "Cell not found"}), 404
            spike_run = _latest_completed_run(conn, int(n["experiment_id"]), "spike")
        if spike_run is None:
            return jsonify({"error": "No completed spike run."}), 400
        from utils.movie_utils import get_movie_parameters
        from utils.spikes_utils import classify_spikes, detect_peaks_mad, highpass_traces
        from utils.suite2p_utils import load_suite2p_context

        home = Path(n["source_path"])
        suite2p = load_suite2p_context(home / "pipeline_results/for_suite2p/suite2p/plane0")
        cell_id = int(n["pipeline_neuron_id"])
        fps = int(get_movie_parameters(f"{home.as_posix()}/", is_dir=True)["fps"])
        traces = np.asarray(suite2p["F_norm"], dtype=np.float32)
        if cell_id < 0 or cell_id >= traces.shape[0]:
            return jsonify({"error": f"Cell {cell_id} is outside Suite2p traces."}), 400
        trace = np.asarray(traces[cell_id], dtype=np.float32)
        detection = highpass_traces(trace[np.newaxis, :], fps, params["highpass_cutoff_hz"])[0]
        peaks = detect_peaks_mad(
            trace,
            detection,
            mad_factor=params["mad_factor"],
            duplicate_window_frames=params["duplicate_window_frames"],
            align_radius_frames=params["align_radius_frames"],
            min_distance=params["min_distance"],
        )
        labels, features = classify_spikes(
            trace,
            peaks,
            fps,
            min_amplitude=params["min_amplitude"],
            min_width=params["min_width"],
            min_subthreshold=params["min_subthreshold"],
            min_ratio=params["min_ratio"],
            subthreshold_window_seconds=0.2,
            baseline_window_seconds=1.5,
        )
        return jsonify(
            {
                "peaks": labels.peaks_all.astype(int).tolist(),
                "labels": labels.labels.astype(str).tolist(),
                "n_simple": int(labels.simple_peaks.size),
                "n_complex": int(labels.complex_peaks.size),
                "n_total": int(labels.peaks_all.size),
                "params": params,
                "score": np.round(features.get("score", np.empty(0)), 4).tolist(),
            }
        )
    @app.post("/neurons/<int:neuron_id>/spikes/save")
    def save_spike_data(neuron_id):
        data = request.get_json(force=True)
        peaks = np.asarray(data.get("peaks", []), dtype=np.int32)
        labels = np.asarray(data.get("labels", [])).astype("<U8")
        if peaks.size != labels.size:
            return jsonify({"error": "peaks and labels must have the same length"}), 400
        keep = labels != "delete"
        peaks = peaks[keep]
        labels = labels[keep]
        with connect(settings) as conn:
            n = conn.execute("SELECT n.*,e.id AS experiment_id FROM neurons n JOIN experiments e ON e.id=n.experiment_id WHERE n.id=?", (neuron_id,)).fetchone()
            if n is None:
                return jsonify({"error": "Cell not found"}), 404
            spike_run = _latest_completed_run(conn, int(n["experiment_id"]), "spike")
        if spike_run is None:
            return jsonify({"error": "No completed spike run."}), 400
        _save_spike_payload(_spike_cell_dir(settings, spike_run, n["pipeline_neuron_id"]), peaks, labels)
        return jsonify({"ok": True, "n_peaks": int(peaks.size)})
    @app.get("/runs/<int:run_id>")
    def run_detail(run_id):
        with connect(settings) as conn:
            run = conn.execute("SELECT r.*,e.canonical_id,e.source_path FROM analysis_runs r JOIN experiments e ON e.id=r.experiment_id WHERE r.id=?", (run_id,)).fetchone()
            if run is None:
                return render_template("error.html", title="Missing Run", message="Run not found"), 404
        rv = _run_view(run, settings)
        return render_template("run.html", run=rv, stdout=_run_stdout(rv), stderr=_run_stderr(rv), active=rv["status"] in {"pending", "queued", "running"})
    @app.get("/runs/<int:run_id>/logs")
    def run_logs(run_id):
        with connect(settings) as conn:
            run = conn.execute("SELECT * FROM analysis_runs WHERE id=?", (run_id,)).fetchone()
            if run is None:
                return jsonify({"error": "Run not found"}), 404
        return jsonify(
            {
                "status": run["status"],
                "stdout": _run_stdout(dict(run)),
                "stderr": _run_stderr(dict(run)),
            }
        )
    @app.get("/runs/<int:run_id>/artifact/<path:rel_path>")
    def artifact(run_id, rel_path):
        with connect(settings) as conn:
            run = conn.execute("SELECT * FROM analysis_runs WHERE id=?", (run_id,)).fetchone()
            if run is None:
                return render_template("error.html", title="Missing Run", message="Run not found"), 404
        root = settings.test_output_root if run["is_test"] else settings.output_root
        target = (root / rel_path).resolve()
        allowed = root.resolve()
        if allowed not in target.parents and target != allowed:
            return render_template("error.html", title="Unsafe Path", message="Artifact path is outside the output root."), 400
        if not target.exists() or not target.is_file():
            return render_template("error.html", title="Missing Artifact", message=str(target)), 404
        return send_file(target)
    return app
