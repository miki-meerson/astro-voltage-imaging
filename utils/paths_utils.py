from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tifffile
import re

@dataclass(frozen=True)
class STAPaths:
    home_dir: Path
    sta_dir: Path
    pb_dir: Path
    mc_dir: Path
    astro_movie_path: Path
    neuron_movie_path: Path
    suite2p_path: Path
    raw_movie_path: Path
    bad_segments_path: Path
    peaks_dir: Path
    peaks_labels_path: Path


def resolve_sta_paths(home_dir: str | Path) -> STAPaths:
    home_dir = Path(home_dir)
    pipeline_dir = home_dir / "pipeline_results"
    pb_dir = pipeline_dir / "pb_corrected"
    suite2p_path = pipeline_dir / "for_suite2p" / "suite2p" / "plane0"
    bad_segments_path = pipeline_dir / "qc" / "bad_segments.json"

    return STAPaths(
        home_dir=home_dir,
        sta_dir=home_dir / "sta-analysis",
        pb_dir=pb_dir,
        mc_dir=pipeline_dir / "motion_corrected",
        astro_movie_path=pb_dir / "astro_photobleaching_corrected.tif",
        neuron_movie_path=pb_dir / "neuron_photobleaching_corrected.tif",
        suite2p_path=suite2p_path,
        raw_movie_path=home_dir / "Image_scan_1_region_0_0.tif",
        bad_segments_path=bad_segments_path,
        peaks_dir = home_dir / "automated-spike-analysis",
        peaks_labels_path = "simple_complex_spike_labels.npz"
    )


def resolve_behavior_path(home_dir: str | Path) -> Path:
    home_dir = Path(home_dir)
    pattern = re.compile(r"^imaging_(\d+)\.csv$")

    matches = []
    for path in home_dir.glob("imaging_*.csv"):
        match = pattern.match(path.name)
        if match:
            matches.append((int(match.group(1)), path))

    if not matches: raise FileNotFoundError(f"No imaging_<i>.csv file found in {home_dir}")
    if len(matches) > 1: raise ValueError(f"More than one behavioral file found")
    return min(matches)[1]


def list_sta_cells(sta_dir: str | Path) -> list[int]:
    sta_dir = Path(sta_dir)
    cell_ids = []
    for path in sta_dir.glob("cell_*"):
        if not path.is_dir():
            continue
        try:
            cell_ids.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(cell_ids)
    

def load_sta_movie(cell_dir: str | Path) -> np.ndarray:
    return tifffile.imread(Path(cell_dir) / "astro_sta_movie.tif").astype(np.float32)


def parse_result_file(path: Path):
    stem = path.stem.removeprefix("astro_sta_cluster_permutation")
    if stem.startswith("_"):
        stem = stem[1:]

    version = 0
    m = re.search(r"_v(\d+)$", stem)
    if m:
        version = int(m.group(1))
        stem = stem[:m.start()]

    return {"result_type": stem or "all", "version": version}
