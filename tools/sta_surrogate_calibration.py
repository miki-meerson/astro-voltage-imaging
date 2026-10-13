"""Re-evaluate saved STA anatomy results under alternative statistic settings and calibrate against surrogates.

Read-only. For every cell it loads the anatomy result (utils/globals_utils.py ANATOMY_RESULT; observed + permutation effects),
re-runs the cluster-permutation statistic with the real code from cluster-code/sta_maps_sparse.py under several
configurations, and reports for each:

  * cells with a significant cluster vs. the number expected by chance (each permutation is treated as pseudo-observed data,
    judged against the remaining permutations),
  * where the significant pixels fall in brightness deciles (real vs. chance) -> brightness bias,
  * median cluster size (voxels / footprint px / frames), real vs. chance.

Usage (from the repo root, analysis environment with numpy/scipy; PYTHONPATH must hold the repo root and cluster-code):
  PYTHONPATH=.:cluster-code python -m tools.sta_surrogate_calibration --sta-dir <run>/sta_algorithm_home/sta-analysis \
      --pipeline-dir <experiment>/pipeline_results --out calib.json [--configs configs.json]

`--configs` is an optional JSON {name: {SETTING: value}} of sta_maps_sparse module globals, e.g.
  {"legacy": {"NULL_STANDARDIZATION": "none"}, "z": {"NULL_STANDARDIZATION": "per_pixel_z", "MIN_CLUSTER_FRAMES": 2}}
See docs/statistics.md for how to read the output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from scipy.stats import rankdata

import sta_maps_sparse as sta
from utils import globals_utils as g
from utils.paths_utils import PathKind, cell_dirs, path_for

DEFAULT_CONFIGS = {
    "legacy": dict(NULL_STANDARDIZATION="none", SMOOTH_SIGMA_FULLRES_PX=0.0, MIN_CLUSTER_FOOTPRINT_FULLRES_PX=0.0, MIN_CLUSTER_FRAMES=0),
    "per_pixel_z": dict(NULL_STANDARDIZATION="per_pixel_z", SMOOTH_SIGMA_FULLRES_PX=0.0, MIN_CLUSTER_FOOTPRINT_FULLRES_PX=0.0, MIN_CLUSTER_FRAMES=0),
    "z_plus_extent": dict(NULL_STANDARDIZATION="per_pixel_z", SMOOTH_SIGMA_FULLRES_PX=0.0, MIN_CLUSTER_FOOTPRINT_FULLRES_PX=9.0, MIN_CLUSTER_FRAMES=2),
    "z_smooth_extent": dict(NULL_STANDARDIZATION="per_pixel_z", SMOOTH_SIGMA_FULLRES_PX=1.0, MIN_CLUSTER_FOOTPRINT_FULLRES_PX=9.0, MIN_CLUSTER_FRAMES=2),
}
DECILES = np.arange(0, 101, 10)


def _anatomy_files(sta_dir: str | Path) -> list[Path]:
    """The anatomy result of every cell that has one, ordered by cell number."""
    files = [path_for(PathKind.ANATOMY_RESULT, sta_dir, cell=cell) for cell in cell_dirs(sta_dir)]
    return [f for f in files if f.exists()]


def _geom(mask: np.ndarray) -> tuple[int, int, int]:
    footprint = mask.any(0)
    return int(mask.sum()), int(footprint.sum()), int(mask.any((1, 2)).sum())


def evaluate(files: list[Path], mean_full: np.ndarray, configs: dict) -> dict:
    out = {k: dict(cells=0, cells_sig=0, sur_any=[], real_b=np.zeros(10), sur_b=np.zeros(10), real_g=[], sur_g=[]) for k in configs}
    mean_img = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        if mean_img is None:
            bin_ = int(d["spatial_bin"])
            sta.SPATIAL_BIN = bin_
            h, w = mean_full.shape
            mean_img = mean_full.reshape(h // bin_, bin_, w // bin_, bin_).mean((1, 3)) if bin_ > 1 else mean_full
        y0, y1, x0, x1 = (int(v) for v in d["anatomy_crop_bounds"])
        img = mean_img[y0:y1, x0:x1]
        bright = rankdata(img.ravel()).reshape(img.shape) / img.size * 100
        raw_obs, raw_perm = d["observed_effect_movie"], d["perm_effects"]
        sta.VOXEL_ALPHA, sta.CLUSTER_ALPHA = float(d["voxel_alpha"]), float(d["cluster_alpha"])
        for name, cfg in configs.items():
            for key, value in cfg.items():
                setattr(sta, key, value)
            o = out[name]
            o["cells"] += 1
            obs, perm, _ = sta.prepare_test_statistic(raw_obs, raw_perm)
            thr = float(np.quantile(np.abs(perm), 1 - sta.VOXEL_ALPHA))
            null_max = np.asarray([max((c["mass"] for c in sta.cluster_records(e, thr)), default=0.0) for e in perm], dtype=np.float32)
            summary, sig_mask = sta.summarize_clusters(obs, thr, null_max)
            o["cells_sig"] += int(any(c["is_significant"] for c in summary))
            if sig_mask.any():
                lab, n = ndi.label(sig_mask, ndi.generate_binary_structure(3, 2))
                for i in range(1, n + 1):
                    m = lab == i
                    o["real_g"].append(_geom(m))
                    o["real_b"] += np.histogram(bright[m.any(0)], bins=DECILES)[0]
            hits = 0
            for k in range(perm.shape[0]):
                others = np.delete(null_max, k)
                hit = False
                for rec in sta.cluster_records(perm[k], thr):
                    if (1 + np.sum(others >= rec["mass"])) / (len(others) + 1) <= sta.CLUSTER_ALPHA:
                        hit = True
                        o["sur_g"].append(_geom(rec["mask"]))
                        o["sur_b"] += np.histogram(bright[rec["mask"].any(0)], bins=DECILES)[0]
                hits += hit
            o["sur_any"].append(hits / perm.shape[0])
        print(f"{Path(f).parent.name}: " + ", ".join(f"{k}={out[k]['cells_sig']}" for k in out), flush=True)

    def share(hist, idx):
        return round(float(hist[idx].sum() / max(hist.sum(), 1)), 3)

    summary = {}
    for name, o in out.items():
        rg, sg = np.array(o["real_g"]).reshape(-1, 3), np.array(o["sur_g"]).reshape(-1, 3)
        summary[name] = dict(
            cells=o["cells"],
            cells_with_significant_cluster=o["cells_sig"],
            expected_by_chance=round(float(np.sum(o["sur_any"])), 1),
            real_clusters=len(rg),
            chance_clusters=len(sg),
            real_share_in_brightest_decile=share(o["real_b"], [9]),
            chance_share_in_brightest_decile=share(o["sur_b"], [9]),
            real_median_voxels_footprint_frames=np.median(rg, 0).tolist() if len(rg) else None,
            chance_median_voxels_footprint_frames=np.median(sg, 0).tolist() if len(sg) else None,
        )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sta-dir", required=True, help="folder containing cell_<id>/ result folders")
    ap.add_argument("--pipeline-dir", required=True, help="experiment pipeline_results folder (for the mean astro image)")
    ap.add_argument("--out", default=None, help="write the summary JSON here")
    ap.add_argument("--configs", default=None, help="JSON file {name: {SETTING: value}}")
    ap.add_argument("--max-cells", type=int, default=None)
    args = ap.parse_args()

    configs = json.loads(Path(args.configs).read_text()) if args.configs else DEFAULT_CONFIGS
    files = _anatomy_files(args.sta_dir)
    if args.max_cells:
        files = files[: args.max_cells]
    if not files:
        raise SystemExit(f"no {g.ANATOMY_RESULT} files found")
    mean_full = np.load(path_for(PathKind.MC_ASTRO_MEAN_IMAGE, path_for(PathKind.MOTION_CORRECTED_DIR, Path(args.pipeline_dir).parent))).astype(np.float32)
    result = evaluate(files, mean_full, configs)
    text = json.dumps(result, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
