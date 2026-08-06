from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TS_SAMPLING_RATE_HZ = 30000.0
ENCODER_PULSES_PER_REVOLUTION = 200.0
WHEEL_DIAMETER_CM = 14.0
QUADRATURE_COUNTS_PER_PULSE = 4.0
DEFAULT_VELOCITY_KEYS = (
    "Encoder_ch_A",
    "Encoder_ch_B",
    "decoder",
    "Decoder",
    "velocity",
    "Velocity",
    "speed",
    "Speed",
    "Motor",
    "AI/Motor",
)


@dataclass(frozen=True)
class TSBehavior:
    h5_path: Path
    velocity_key: str
    velocity: np.ndarray
    sample_rate_hz: float
    frame_times_s: np.ndarray | None
    encoder_b_key: str | None = None


def find_ts_h5_files(experiment_dir: str | Path) -> list[Path]:
    home = Path(experiment_dir)
    ts_dirs = sorted([p for p in home.iterdir() if p.is_dir() and p.name.lower().startswith("ts")])
    files: list[Path] = []
    for ts_dir in ts_dirs:
        files.extend(sorted(ts_dir.rglob("*.h5")))
        files.extend(sorted(ts_dir.rglob("*.hdf5")))
    return files


def list_h5_datasets(h5_path: str | Path) -> list[dict]:
    import h5py

    result: list[dict] = []
    with h5py.File(h5_path, "r") as f:
        def visit(name, obj):
            if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                result.append({"key": name, "shape": tuple(obj.shape), "dtype": str(obj.dtype)})

        f.visititems(visit)
    return result


def _find_dataset(f, requested: str | None):
    import h5py

    datasets = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets[name] = obj

    f.visititems(visit)
    if requested:
        for key, ds in datasets.items():
            if key == requested or key.split("/")[-1] == requested:
                return key, ds
        raise KeyError(f"TS velocity dataset was not found: {requested}. Available keys: {sorted(datasets)}")
    for candidate in DEFAULT_VELOCITY_KEYS:
        for key, ds in datasets.items():
            if key == candidate or key.split("/")[-1] == candidate:
                return key, ds
    raise KeyError(f"No TS velocity dataset found. Available keys: {sorted(datasets)}")


def _dataset_by_leaf(f, leaf: str):
    import h5py

    found = None

    def visit(name, obj):
        nonlocal found
        if found is None and isinstance(obj, h5py.Dataset) and (name == leaf or name.split("/")[-1] == leaf):
            found = (name, obj)

    f.visititems(visit)
    return found


def _read_1d(ds) -> np.ndarray:
    return np.asarray(ds[:]).squeeze()


def _frame_times_from_frameout(f, sample_rate_hz: float) -> np.ndarray | None:
    key = None
    for candidate in ("DI/FrameOut", "FrameOut"):
        if candidate in f:
            key = candidate
            break
    if key is None:
        return None
    frame_out = _read_1d(f[key])
    rising = np.flatnonzero((frame_out[1:] > 0) & (frame_out[:-1] <= 0)) + 1
    if rising.size == 0:
        return None
    return rising.astype(np.float64) / float(sample_rate_hz)


def load_ts_behavior(
    experiment_dir: str | Path,
    velocity_key: str | None = None,
    sample_rate_hz: float = TS_SAMPLING_RATE_HZ,
) -> TSBehavior:
    import h5py

    h5_files = find_ts_h5_files(experiment_dir)
    if not h5_files:
        raise FileNotFoundError(f"No TS Episode .h5 file found in {experiment_dir}")
    if len(h5_files) > 1:
        raise ValueError(f"Expected one TS H5 file, found {len(h5_files)}: {[str(p) for p in h5_files]}")
    h5_path = h5_files[0]
    with h5py.File(h5_path, "r") as f:
        if velocity_key in (None, "", "encoder", "Encoder"):
            a = _dataset_by_leaf(f, "Encoder_ch_A")
            b = _dataset_by_leaf(f, "Encoder_ch_B")
            if a is not None:
                key, ds = a
                velocity = _read_1d(ds).astype(np.float64)
                b_key = b[0] if b is not None else None
            else:
                key, ds = _find_dataset(f, velocity_key)
                velocity = _read_1d(ds).astype(np.float64)
                b_key = None
        else:
            key, ds = _find_dataset(f, velocity_key)
            velocity = _read_1d(ds).astype(np.float64)
            b_key = None
        frame_times = _frame_times_from_frameout(f, sample_rate_hz)
    return TSBehavior(
        h5_path=h5_path,
        velocity_key=key,
        velocity=velocity,
        sample_rate_hz=float(sample_rate_hz),
        frame_times_s=frame_times,
        encoder_b_key=b_key,
    )


def load_ts_encoder_velocity_at_times(
    experiment_dir: str | Path,
    target_times_s: np.ndarray,
    sample_rate_hz: float = TS_SAMPLING_RATE_HZ,
    wheel_diameter_cm: float = WHEEL_DIAMETER_CM,
    pulses_per_revolution: float = ENCODER_PULSES_PER_REVOLUTION,
    invert_direction: bool = True,
    absolute: bool = False,
) -> tuple[np.ndarray, dict]:
    import h5py

    h5_files = find_ts_h5_files(experiment_dir)
    if not h5_files:
        raise FileNotFoundError(f"No TS Episode .h5 file found in {experiment_dir}")
    if len(h5_files) > 1:
        raise ValueError(f"Expected one TS H5 file, found {len(h5_files)}: {[str(p) for p in h5_files]}")
    h5_path = h5_files[0]
    with h5py.File(h5_path, "r") as f:
        a_found = _dataset_by_leaf(f, "Encoder_ch_A")
        b_found = _dataset_by_leaf(f, "Encoder_ch_B")
        if a_found is None and b_found is None:
            raise KeyError("No TS encoder channel found. Expected Encoder_ch_A and/or Encoder_ch_B.")
        a_key, a_ds = a_found if a_found is not None else b_found
        a = _read_1d(a_ds).astype(np.uint8) > 0
        if b_found is not None and a_found is not None:
            b_key, b_ds = b_found
            b = _read_1d(b_ds).astype(np.uint8) > 0
            prev = (a[:-1].astype(np.uint8) << 1) | b[:-1].astype(np.uint8)
            curr = (a[1:].astype(np.uint8) << 1) | b[1:].astype(np.uint8)
            lookup = np.array([0, -1, 1, 0, 1, 0, 0, -1, -1, 0, 0, 1, 0, 1, -1, 0], dtype=np.int8)
            delta = lookup[(prev << 2) | curr].astype(np.float64)
            keys = f"{a_key}+{b_key}"
            counts_per_revolution = float(pulses_per_revolution) * QUADRATURE_COUNTS_PER_PULSE
        else:
            delta = (a[1:] != a[:-1]).astype(np.float64)
            keys = a_key
            counts_per_revolution = float(pulses_per_revolution) * 2.0
        if invert_direction:
            delta = -delta
        cm_per_count = (np.pi * float(wheel_diameter_cm)) / max(counts_per_revolution, 1e-9)
        pos = np.r_[0.0, np.cumsum(delta)]
        sample_times = np.arange(pos.size, dtype=np.float64) / float(sample_rate_hz)
        target_times = np.asarray(target_times_s, dtype=np.float64)
        target_pos = np.interp(target_times, sample_times, pos, left=np.nan, right=np.nan)
        velocity = np.gradient(target_pos, target_times, edge_order=1) if target_times.size > 1 else np.zeros_like(target_pos)
        velocity = velocity * cm_per_count
        if absolute:
            velocity = np.abs(velocity)
        return velocity.astype(np.float64), {
            "behavior_source": "ts",
            "behavior_file": str(h5_path),
            "ts_velocity_key": keys,
            "ts_sampling_rate_hz": float(sample_rate_hz),
            "ts_encoder_mode": "quadrature" if b_found is not None and a_found is not None else "single_channel",
            "ts_velocity_signed": not bool(absolute),
            "ts_wheel_diameter_cm": float(wheel_diameter_cm),
            "ts_encoder_pulses_per_revolution": float(pulses_per_revolution),
            "ts_counts_per_revolution": float(counts_per_revolution),
            "ts_direction_inverted": bool(invert_direction),
        }
