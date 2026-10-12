from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tifffile

@dataclass(frozen=True)
class STAPaths:
    home_dir: Path
    sta_dir: Path
    pb_dir: Path
    mc_dir: Path
    astro_movie_path: Path
    suite2p_path: Path
    raw_movie_path: Path
    bad_segments_path: Path


def resolve_sta_paths(home_dir: str | Path, sta_home_dir: str | Path | None = None) -> STAPaths:
    home_dir = Path(home_dir)
    pipeline_dir = home_dir / "pipeline_results"
    source_pb_dir = pipeline_dir / "pb_corrected"
    sta_home = Path(sta_home_dir) if sta_home_dir is not None else home_dir
    if (sta_home / "sta_algorithm_home").exists():
        sta_home = sta_home / "sta_algorithm_home"
    suite2p_path = pipeline_dir / "for_suite2p" / "suite2p" / "plane0"

    return STAPaths(
        home_dir=home_dir,
        sta_dir=sta_home / "sta-analysis",
        pb_dir=source_pb_dir,
        mc_dir=pipeline_dir / "motion_corrected",
        astro_movie_path=source_pb_dir / "astro_photobleaching_corrected.tif",
        suite2p_path=suite2p_path,
        raw_movie_path=home_dir / "Image_scan_1_region_0_0.tif",
        bad_segments_path=pipeline_dir / "qc" / "bad_segments.json",
    )


    

def load_sta_movie(cell_dir: str | Path) -> np.ndarray:
    return tifffile.imread(Path(cell_dir) / "astro_sta_movie.tif").astype(np.float32)
