"""Behaviour (speed / velocity) payload of the experiment page."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.services import caches
from app.services.runs import run_dir
from utils.paths_utils import PathKind, path_for


def behavior_payload(settings, run_row) -> dict:
    if run_row is None:
        return {"available": False, "message": "No completed behavior run is available."}
    path = path_for(PathKind.BEHAVIOR_SEGMENTS, run_dir(settings, int(run_row["id"]), bool(run_row["is_test"])))
    if not path.exists():
        return {"available": False, "message": "Behavior segments file was not found."}
    cache_key = (int(run_row["id"]), int(path.stat().st_mtime_ns))
    cached = caches.BEHAVIOR_PAYLOAD.get(cache_key)
    if cached is not None:
        return cached
    max_points = 3000
    with np.load(path, allow_pickle=True) as d:
        time_s = np.asarray(d["pb_frame_times_s"], dtype=float)
        active = np.asarray(d["active_mask"], dtype=bool)
        inactive = np.asarray(d["inactive_mask"], dtype=bool)
        covered = np.asarray(d["behavior_covered_mask"], dtype=bool)
        threshold = float(np.asarray(d["speed_threshold"]).item())
        fps = float(np.asarray(d["fps"]).item())
        chosen = str(np.asarray(d["chosen_alignment"]).item())
        behavior_source = str(np.asarray(d["behavior_source"]).item())
        ts_fallback_reason = str(np.asarray(d["ts_fallback_reason"]).item())
        ts_velocity_key = str(np.asarray(d["ts_velocity_key"]).item())
        behavior_file = str(np.asarray(d["behavior_file"]).item())
        speed = np.asarray(d["speed_trace"], dtype=float)
        velocity = np.asarray(d["velocity_trace"], dtype=float)
        ts_encoder_mode = str(np.asarray(d["ts_encoder_mode"]).item())
        ts_velocity_signed = bool(np.asarray(d["ts_velocity_signed"]).item())
        ts_wheel_diameter_cm = float(np.asarray(d["ts_wheel_diameter_cm"]).item())
        ts_encoder_pulses_per_revolution = float(np.asarray(d["ts_encoder_pulses_per_revolution"]).item())
        ts_counts_per_revolution = float(np.asarray(d["ts_counts_per_revolution"]).item())
        ts_direction_inverted = bool(np.asarray(d["ts_direction_inverted"]).item())
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
    caches.BEHAVIOR_PAYLOAD.put(cache_key, payload)
    return payload
