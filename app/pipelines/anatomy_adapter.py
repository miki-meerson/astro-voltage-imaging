from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.services.visualizations import _font, _gray_rgb
from utils.movie_utils import read_astro_movie, spatial_bin_movie
from utils.paths_utils import resolve_sta_paths
from utils.suite2p_utils import load_suite2p_context, neuron_centroid_binned


def _sampled_mean_binned_movie(movie_path: Path, spatial_bin: int, max_frames: int = 1000) -> np.ndarray:
    movie = read_astro_movie(str(movie_path), as_memmap=True)
    frame_count = int(movie.shape[0])
    step = max(1, int(np.ceil(frame_count / max_frames)))
    indices = np.arange(0, frame_count, step, dtype=int)
    sample = spatial_bin_movie(np.asarray(movie[0:1], dtype=np.float32), spatial_bin)
    acc = np.zeros(sample.shape[1:], dtype=np.float64)
    count = 0
    chunk_size = 64
    for offset in range(0, indices.size, chunk_size):
        chunk_idx = indices[offset : offset + chunk_size]
        chunk = spatial_bin_movie(np.asarray(movie[chunk_idx], dtype=np.float32), spatial_bin)
        acc += np.asarray(chunk, dtype=np.float64).sum(axis=0)
        count += int(chunk.shape[0])
    return (acc / max(count, 1)).astype(np.float32)


def _crop_bounds(center_yx: tuple[int, int], radius: int, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    y, x = center_yx
    h, w = image_shape
    return (
        max(0, int(y) - int(radius)),
        min(h, int(y) + int(radius) + 1),
        max(0, int(x) - int(radius)),
        min(w, int(x) + int(radius) + 1),
    )


def _draw_preview(path: Path, mean_image: np.ndarray, suite2p_ctx: dict, radius: int, spatial_bin: int, max_cells: int = 8) -> None:
    cells = list(range(min(int(suite2p_ctx["n_cells"]), max_cells)))
    tile_w, tile_h = 260, 260
    cols = min(4, max(1, len(cells)))
    rows = int(np.ceil(max(1, len(cells)) / cols))
    margin, title_h, gap = 18, 58, 14
    w = margin * 2 + cols * tile_w + (cols - 1) * gap
    h = title_h + margin + rows * tile_h + (rows - 1) * gap
    canvas = Image.new("RGB", (w, h), "#0e1116")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), f"Anatomy Crop Preview | radius {radius} binned px", fill="#e7edf4", font=_font(22))

    base = _gray_rgb(mean_image).convert("RGB")
    scale = min(tile_w / max(base.width, 1), tile_h / max(base.height, 1))
    shown = base.resize((max(1, int(base.width * scale)), max(1, int(base.height * scale))), Image.Resampling.NEAREST)

    for idx, cell_id in enumerate(cells):
        col, row = idx % cols, idx // cols
        x0 = margin + col * (tile_w + gap)
        y0 = title_h + row * (tile_h + gap)
        tile = Image.new("RGB", (tile_w, tile_h), "#111821")
        ox, oy = (tile_w - shown.width) // 2, (tile_h - shown.height) // 2
        tile.paste(shown, (ox, oy))
        td = ImageDraw.Draw(tile)
        center = neuron_centroid_binned(cell_id, spatial_bin, mean_image.shape, suite2p_ctx)
        cy = oy + int(round(center[0] * scale))
        cx = ox + int(round(center[1] * scale))
        y_min, y_max, x_min, x_max = _crop_bounds(center, radius, mean_image.shape)
        rect = (
            ox + int(round(x_min * scale)),
            oy + int(round(y_min * scale)),
            ox + int(round(x_max * scale)),
            oy + int(round(y_max * scale)),
        )
        td.rectangle(rect, outline="#37c6a6", width=2)
        td.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill="#f7d154")
        td.text((10, 10), f"Cell {cell_id}", fill="#e7edf4", font=_font(15))
        canvas.paste(tile, (x0, y0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--params-json", default="{}")
    args = ap.parse_args()

    home = Path(args.experiment)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    params = json.loads(args.params_json or "{}")
    spatial_bin = int(params.get("spatial_bin", 8))
    radius = int(params.get("anatomy_radius_binned", params.get("default_radius_binned", 8)))
    paths = resolve_sta_paths(home)

    mean_image = _sampled_mean_binned_movie(paths.astro_movie_path, spatial_bin)
    suite2p_ctx = load_suite2p_context(paths.suite2p_path)
    sta_dir = out / "sta-analysis"
    sta_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sta_dir / "anatomy_crop_params.npz",
        anatomy_radius_binned=np.asarray(radius, dtype=np.int32),
        spatial_bin=np.asarray(spatial_bin, dtype=np.int32),
        home_dir=str(home),
        source="astro-analysis-pipeline anatomy_adapter",
        n_review_cells=np.asarray(int(suite2p_ctx["n_cells"]), dtype=np.int32),
    )
    np.save(sta_dir / "mean_astro_binned.npy", mean_image.astype(np.float32))
    _draw_preview(out / "anatomy_crop_preview.png", mean_image, suite2p_ctx, radius, spatial_bin)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "anatomy_radius_binned": radius,
                "spatial_bin": spatial_bin,
                "n_cells": int(suite2p_ctx["n_cells"]),
                "output": "sta-analysis/anatomy_crop_params.npz",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
