from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion

from app.db import neuron_repository
from app.services import caches, polarity
from app.services.runs import run_dir
from app.services.suite2p_data import experiment_stat_cells
from app.settings import Settings
from utils.cluster_result import load_cluster_result
from utils.db_utils import collect_cluster_result_metadata
from utils.movie_utils import get_mean_images
from utils.paths_utils import PathKind, path_for, resolve_sta_paths, sta_dir_of_run
from utils.suite2p_utils import suite2p_roi_mask

RESULT_ORDER = {
    "all": 0,
    "simple": 1,
    "complex": 2,
    "anatomy": 3,
    "behavioral_active": 4,
    "behavioral_inactive": 5,
}
NEURON_COLOR = "#f7d154"
DEP_COLOR = polarity.COLORS[polarity.DEPOLARIZING]
HYPER_COLOR = polarity.COLORS[polarity.HYPERPOLARIZING]
BG = "#10151c"
PANEL = "#0b0f14"
TEXT = "#edf3fa"
MUTED = "#9aa8b7"
TRACE_TEXT = "#d7e1ec"


def _font(size: int):
    for name in ("arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _gray_rgb(image: np.ndarray) -> Image.Image:
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.nanpercentile(image, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(image)), float(np.nanmax(image))
    scaled = np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)
    gray = (scaled * 255).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGBA")


def _image_response(image: Image.Image) -> BytesIO:
    bio = BytesIO()
    image.convert("RGB").save(bio, format="PNG", optimize=False, compress_level=1)
    bio.seek(0)
    return bio


def _bytes_response(data: bytes) -> BytesIO:
    bio = BytesIO(data)
    bio.seek(0)
    return bio


def _mask_outline(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    outline = mask & ~binary_erosion(mask)
    return outline if outline.any() else mask


def _hex_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _overlay_mask_pixels(base: Image.Image, mask: np.ndarray, color: str, alpha: float = 0.9) -> Image.Image:
    if mask is None or not np.asarray(mask).any():
        return base
    if np.asarray(mask).shape != (base.height, base.width):
        return base
    outline = _mask_outline(mask)
    rgba = np.zeros((outline.shape[0], outline.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = _hex_rgb(color)
    rgba[..., 3] = (outline.astype(np.float32) * 255 * float(alpha)).astype(np.uint8)
    return Image.alpha_composite(base.convert("RGBA"), Image.fromarray(rgba, mode="RGBA"))


def _fit_tile(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    tile = Image.new("RGBA", size, PANEL)
    scale = min(size[0] / max(image.width, 1), size[1] / max(image.height, 1))
    work = image.resize((max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))), Image.Resampling.BICUBIC)
    tile.alpha_composite(work.convert("RGBA"), ((size[0] - work.width) // 2, (size[1] - work.height) // 2))
    return tile


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [lo, hi]
    raw = (hi - lo) / max(1, n - 1)
    power = 10 ** np.floor(np.log10(raw))
    step = next((m * power for m in (1, 2, 5, 10) if m * power >= raw), raw)
    first = np.ceil(lo / step) * step
    ticks = []
    value = first
    while value <= hi + step * 0.5:
        ticks.append(float(value))
        value += step
    return ticks


def _fmt_dff(value: float) -> str:
    pct = value * 100
    if abs(pct) >= 10:
        return f"{pct:.0f}%"
    return f"{pct:.1f}%"


def _draw_trace_tile(
    t_rel: np.ndarray,
    traces: list[tuple[np.ndarray, str]],
    size: tuple[int, int],
    tested_window: tuple[float, float] | None = None,
) -> Image.Image:
    tile = Image.new("RGB", size, "#0b0f14")
    draw = ImageDraw.Draw(tile)
    pad_l, pad_r, pad_t, pad_b = 70, 20, 22, 48
    w, h = size
    plot = (pad_l, pad_t, w - pad_r, h - pad_b)
    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=8, fill="#0b0f14", outline="#26303a", width=1)
    x0, x1 = float(np.nanmin(t_rel)), float(np.nanmax(t_rel))
    ys = [np.asarray(y, dtype=float).ravel() for y, _ in traces if np.asarray(y).size]
    if ys:
        y_all = np.concatenate(ys)
        y0, y1 = float(np.nanpercentile(y_all, 1)), float(np.nanpercentile(y_all, 99))
        half = max(abs(y0), abs(y1), 0.005)
        y0, y1 = -half, half
    else:
        y0, y1 = -0.05, 0.05
        draw.text((w // 2, h // 2), "no significant clusters", fill="#7f8c9b", font=_font(16), anchor="mm")

    def xy(t, y):
        px = plot[0] + (float(t) - x0) / max(x1 - x0, 1e-6) * (plot[2] - plot[0])
        py = plot[3] - (float(y) - y0) / max(y1 - y0, 1e-6) * (plot[3] - plot[1])
        return int(round(px)), int(round(py))

    if tested_window is not None and (tested_window[0] > x0 or tested_window[1] < x1):
        # displayed window is wider than the tested one: shade the frames that were actually tested
        sx0, sx1 = xy(max(tested_window[0], x0), y0)[0], xy(min(tested_window[1], x1), y0)[0]
        draw.rectangle((sx0, plot[1], sx1, plot[3]), fill="#16212c")
        draw.text((sx0 + 4, plot[1] + 3), "tested", fill="#6f8296", font=_font(11), anchor="la")
    draw.rectangle(plot, outline="#33404e", width=1)
    for value in _nice_ticks(y0, y1, 5):
        _, py = xy(x0, value)
        draw.line([(plot[0], py), (plot[2], py)], fill="#1d2630", width=1)
        draw.line([(plot[0] - 5, py), (plot[0], py)], fill="#43505f", width=1)
        draw.text((plot[0] - 10, py), _fmt_dff(value), fill="#aeb9c6", font=_font(13), anchor="rm")

    x_ticks = _nice_ticks(x0, x1, 5)
    for value in x_ticks:
        px, _ = xy(value, y0)
        draw.line([(px, plot[3]), (px, plot[3] + 5)], fill="#43505f", width=1)
        draw.text((px, plot[3] + 11), f"{value:g}", fill="#aeb9c6", font=_font(13), anchor="ma")

    if x0 <= 0 <= x1:
        zx = xy(0, y0)[0]
        draw.line([(zx, plot[1]), (zx, plot[3])], fill="#5d6a78", width=1)
    if y0 <= 0 <= y1:
        zy = xy(x0, 0)[1]
        draw.line([(plot[0], zy), (plot[2], zy)], fill="#526071", width=1)

    for y, color in traces:
        y = np.asarray(y, dtype=float).ravel()
        if y.size != t_rel.size:
            continue
        pts = [xy(t, val) for t, val in zip(t_rel, y)]
        if len(pts) > 1:
            draw.line(pts, fill="#071015", width=5)
            draw.line(pts, fill=color, width=3)

    draw.text((w // 2, h - 10), "Frames From Spike", fill=TRACE_TEXT, font=_font(14), anchor="mm")
    draw.text((12, 6), f"dF/F, {polarity.AXIS_NOTE}", fill=TRACE_TEXT, font=_font(13), anchor="la")
    return tile


def _draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill: str = TEXT) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor="mm")


def _message(title: str, message: str) -> BytesIO:
    img = Image.new("RGB", (1000, 320), "#10151c")
    draw = ImageDraw.Draw(img)
    font_big = _font(28)
    font = _font(18)
    draw.text((500, 120), title, fill="#edf3fa", font=font_big, anchor="mm")
    draw.text((500, 175), message[:180], fill="#9aa8b7", font=font, anchor="mm")
    return _image_response(img)


def _cluster_color(cluster: dict) -> str:
    return polarity.polarity_color(polarity.polarity_from_sign(str(cluster.get("sign", ""))))


def _load_cluster_result(path: Path):
    """Cached (in memory, by file and mtime) read of a saved cluster result; the reading itself is utils.cluster_result.load_cluster_result."""
    cache_key = (str(path), int(path.stat().st_mtime_ns))
    cached = caches.CLUSTER_RESULT.get(cache_key)
    if cached is not None:
        return cached
    result = load_cluster_result(path)
    caches.CLUSTER_RESULT.put(cache_key, result)
    return result


def _display_cluster_mask(
    mask: np.ndarray,
    full_shape: tuple[int, int],
    shown_shape: tuple[int, int],
    crop: tuple[int, int, int, int] | None,
) -> np.ndarray | None:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == shown_shape:
        return mask
    if crop is not None and mask.shape == full_shape:
        y0, y1, x0, x1 = crop
        cropped = mask[y0:y1, x0:x1]
        return cropped if cropped.shape == shown_shape else None
    if crop is None and mask.shape == full_shape:
        return mask
    return None


def _mean_binned_image(paths, spatial_bin: int, cache: dict, behavioral_group: str | None = None) -> np.ndarray | None:
    key = (int(spatial_bin), behavioral_group)
    if key in cache:
        return cache[key]
    path = (
        path_for(PathKind.RUN_BEHAVIORAL_BINNED_MOVIE, paths.home_dir, group=behavioral_group, bin=int(spatial_bin))
        if behavioral_group
        else path_for(PathKind.BINNED_ASTRO_MOVIE, paths.home_dir, bin=int(spatial_bin))
    )
    if path.exists():
        movie = np.load(path, mmap_mode="r")
        step = max(1, int(np.ceil(movie.shape[0] / 1000)))
        cache[key] = np.asarray(movie[::step], dtype=np.float32).mean(axis=0)
        return cache[key]
    return None


def _astro_mean_image(paths, spatial_bin: int, cache: dict) -> np.ndarray | None:
    key = ("astro_mean", int(spatial_bin))
    if key in cache:
        return cache[key]
    try:
        astro_mean, _ = get_mean_images(paths.home_dir, spatial_bin=int(spatial_bin))
        # astro_mean_image.npy is raw; the binned caches above are already flipped, so flip this one to match
        cache[key] = polarity.to_biological(astro_mean)
        return cache[key]
    except (OSError, ValueError):  # the motion-corrected mean image is not saved: the tile says so
        return None


def render_sta_cell_summary(settings: Settings, neuron_row) -> BytesIO:
    sta_run = neuron_repository.latest_completed_run(settings, int(neuron_row["experiment_id"]), "sta")
    if sta_run is None:
        return _message("STA Maps", "No completed STA maps run is available for this experiment.")
    sta_run_dir = run_dir(settings, int(sta_run["id"]), bool(sta_run["is_test"]))
    sta_dir = sta_dir_of_run(sta_run_dir)
    if not sta_dir.exists():
        return _message("STA Maps", f"STA output folder was not found: {sta_dir}")

    cell_id = int(neuron_row["pipeline_neuron_id"])
    result_df = collect_cluster_result_metadata(sta_dir, [cell_id])
    if result_df.empty:
        return _message("STA Maps", f"No STA result files were found for cell {cell_id}.")
    rows = result_df.sort_values("version").groupby("result_type", as_index=False).tail(1)
    rows = rows.assign(_order=rows["result_type"].map(lambda x: RESULT_ORDER.get(str(x), 100))).sort_values(["_order", "result_type"])
    result_paths = [Path(row.path) for row in rows.itertuples()]
    source_path = neuron_row["source_path"]
    source_mtimes = []
    for path in result_paths + [path_for(PathKind.SUITE2P_ISCELL, source_path), path_for(PathKind.SUITE2P_STAT, source_path)]:
        source_mtimes.append((str(path), int(path.stat().st_mtime_ns) if path.exists() else 0))
    cache_key = (
        int(sta_run["id"]),
        int(cell_id),
        tuple((str(row.result_type), int(row.version)) for row in rows.itertuples()),
        tuple(source_mtimes),
        9,
        polarity.RAW_TO_BIOLOGICAL,
    )
    cached = caches.STA_SUMMARY_PNG.get(cache_key)
    if cached is not None:
        return _bytes_response(cached)

    paths = resolve_sta_paths(source_path, path_for(PathKind.STA_ALGORITHM_HOME, sta_run_dir))
    try:
        stat_cells = experiment_stat_cells(source_path)
    except (OSError, ValueError, IndexError, KeyError) as exc:
        return _message("STA Maps", f"Could not read Suite2p outlines: {exc}")

    n = max(1, len(rows))
    tile_w, image_h, trace_h = 430, 320, 200
    gap, top_h, title_h, margin = 22, 78, 56, 22
    width = margin * 2 + n * tile_w + (n - 1) * gap
    height = top_h + title_h + image_h + 34 + trace_h + margin
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 24), f"{neuron_row['canonical_id']} | Cell {cell_id} | STA Maps", fill=TEXT, font=_font(25), anchor="la")
    legend_x = margin
    legend_y = 56
    for label, color in (("depolarizing", DEP_COLOR), ("hyperpolarizing", HYPER_COLOR), ("cell ROI", NEURON_COLOR)):
        draw.ellipse((legend_x, legend_y - 6, legend_x + 12, legend_y + 6), fill=color)
        draw.text((legend_x + 18, legend_y), label, fill=MUTED, font=_font(13), anchor="lm")
        legend_x += 132 if label != "hyperpolarizing" else 146
    cache = {}
    prepared = []

    for row in rows.itertuples():
        effect, t_rel, clusters, meta = _load_cluster_result(Path(row.path))
        behavioral_group = meta["behavioral_group"]
        mean_img = _mean_binned_image(paths, int(meta["spatial_bin"]), cache, behavioral_group)
        if mean_img is None and behavioral_group is not None:
            mean_img = _mean_binned_image(paths, int(meta["spatial_bin"]), cache)
        if mean_img is None and behavioral_group is None:
            mean_img = _astro_mean_image(paths, int(meta["spatial_bin"]), cache)
        title = str(row.result_type).replace("_", " ").title()
        if mean_img is None:
            image_tile = Image.new("RGBA", (tile_w, image_h), PANEL)
            ImageDraw.Draw(image_tile).text((tile_w // 2, image_h // 2), "missing mean image", fill=MUTED, font=_font(17), anchor="mm")
            traces = []
        else:
            neuron = suite2p_roi_mask(cell_id, stat_cells, mean_img.shape, int(meta["spatial_bin"]))
            crop = meta["anatomy_crop_bounds"] if str(row.result_type) == "anatomy" else None
            shown_img, shown_neuron, shown_clusters = mean_img, neuron, clusters
            if crop is not None:
                y0, y1, x0, x1 = crop
                shown_img = mean_img[y0:y1, x0:x1]
                shown_neuron = neuron[y0:y1, x0:x1]
                shown_clusters = [
                    (c, display_mask)
                    for c, m in clusters
                    for display_mask in [_display_cluster_mask(m, mean_img.shape, shown_img.shape, crop)]
                    if display_mask is not None
                ]
            else:
                shown_clusters = [
                    (c, display_mask)
                    for c, m in clusters
                    for display_mask in [_display_cluster_mask(m, mean_img.shape, shown_img.shape, None)]
                    if display_mask is not None
                ]
            image = _gray_rgb(shown_img)
            for cluster, mask in shown_clusters:
                image = _overlay_mask_pixels(image, mask, _cluster_color(cluster), 0.95)
            image = _overlay_mask_pixels(image, shown_neuron, NEURON_COLOR, 0.95)
            image_tile = _fit_tile(image, (tile_w, image_h))
            traces = []
        t_plot = t_rel
        if mean_img is not None:
            for cluster, mask in clusters:
                if np.asarray(mask).shape != effect.shape[1:]:
                    continue
                wide_trace = cluster.get("wide_trace")
                if wide_trace is not None:
                    t_plot, trace = wide_trace  # displayed over a wider window than the tested one
                else:
                    trace = meta["trace_effect"][:, mask].mean(axis=1)
                traces.append((trace, _cluster_color(cluster)))
        trace_tile = _draw_trace_tile(t_plot, traces, (tile_w, trace_h), tested_window=meta["tested_window"])
        prepared.append((title, int(row.n_peaks), int(row.n_significant_clusters), image_tile, trace_tile))

    for col, (title, n_peaks, n_sig, image_tile, trace_tile) in enumerate(prepared):
        x = margin + col * (tile_w + gap)
        _draw_centered(draw, (x + tile_w // 2, top_h + 18), title, _font(19), TEXT)
        _draw_centered(draw, (x + tile_w // 2, top_h + 43), f"{n_peaks} peaks   {n_sig} significant clusters", _font(14), MUTED)
        canvas.paste(image_tile.convert("RGB"), (x, top_h + title_h))
        canvas.paste(trace_tile, (x, top_h + title_h + image_h + 34))

    response = _image_response(canvas)
    caches.STA_SUMMARY_PNG.put(cache_key, response.getvalue())
    response.seek(0)
    return response


MOTION_SERIES_LABELS = {
    "dy": "Registration shift dy (px)",
    "dx": "Registration shift dx (px)",
    "shift_mag": "Shift magnitude (px)",
    "jerk": "Frame-to-frame jerk (px)",
    "reg_corr": "Registration quality (corr. to mean)",
}


def _draw_motion_tile(rels, observed, null_mean, null_sd, title: str, p: float, z: float, size: tuple[int, int]) -> Image.Image:
    """Spike-triggered average of one motion series (colored) over its random-frame null (grey band, +-2 sd)."""
    tile = Image.new("RGB", size, "#0b0f14")
    draw = ImageDraw.Draw(tile)
    w, h = size
    pad_l, pad_r, pad_t, pad_b = 62, 14, 30, 34
    plot = (pad_l, pad_t, w - pad_r, h - pad_b)
    draw.rounded_rectangle((0, 0, w - 1, h - 1), radius=8, fill="#0b0f14", outline="#26303a", width=1)
    significant = bool(np.isfinite(p) and p <= 0.01)
    draw.text((12, 9), title, fill=TEXT, font=_font(14), anchor="lm")
    stat = "no spikes" if not np.isfinite(p) else f"p={p:.3f}  z={z:.1f}"
    draw.text((w - 12, 9), stat, fill=DEP_COLOR if significant else MUTED, font=_font(14), anchor="rm")
    rels = np.asarray(rels, dtype=float)
    obs = np.asarray(observed, dtype=float)
    mu = np.asarray(null_mean, dtype=float)
    sd = np.asarray(null_sd, dtype=float)
    if not np.isfinite(obs).any():
        draw.text((w // 2, h // 2), "no data", fill=MUTED, font=_font(15), anchor="mm")
        return tile
    lo = np.nanmin(np.r_[obs, mu - 2 * sd])
    hi = np.nanmax(np.r_[obs, mu + 2 * sd])
    pad = max((hi - lo) * 0.08, 1e-9)
    y0, y1 = lo - pad, hi + pad
    x0, x1 = float(rels.min()), float(rels.max())

    def xy(t, v):
        return (
            int(round(plot[0] + (float(t) - x0) / max(x1 - x0, 1e-9) * (plot[2] - plot[0]))),
            int(round(plot[3] - (float(v) - y0) / max(y1 - y0, 1e-12) * (plot[3] - plot[1]))),
        )

    draw.rectangle(plot, outline="#33404e", width=1)
    for value in _nice_ticks(y0, y1, 4):
        if not (y0 <= value <= y1):
            continue
        _, py = xy(x0, value)
        draw.line([(plot[0], py), (plot[2], py)], fill="#1d2630", width=1)
        draw.text((plot[0] - 6, py), f"{value:.3g}", fill="#aeb9c6", font=_font(12), anchor="rm")
    for value in _nice_ticks(x0, x1, 5):
        px, _ = xy(value, y0)
        draw.text((px, plot[3] + 8), f"{value:g}", fill="#aeb9c6", font=_font(12), anchor="ma")
    band = [xy(t, m + 2 * s) for t, m, s in zip(rels, mu, sd)] + [xy(t, m - 2 * s) for t, m, s in zip(rels[::-1], mu[::-1], sd[::-1])]
    if len(band) > 3:
        draw.polygon(band, fill="#222c37")
    if x0 <= 0 <= x1:
        zx = xy(0, y0)[0]
        draw.line([(zx, plot[1]), (zx, plot[3])], fill="#5d6a78", width=1)
    pts = [xy(t, v) for t, v in zip(rels, obs)]
    if len(pts) > 1:
        draw.line(pts, fill="#071015", width=5)
        draw.line(pts, fill=DEP_COLOR if significant else NEURON_COLOR, width=3)
    return tile


def render_motion_sta(settings: Settings, neuron_row) -> BytesIO:
    """Cell Detail: is this cell's spiking time-locked to motion / registration quality? (diagnostic only)"""
    sta_run = neuron_repository.latest_completed_run(settings, int(neuron_row["experiment_id"]), "sta")
    if sta_run is None:
        return _message("Motion diagnostics", "No completed STA maps run is available for this experiment.")
    cell_id = int(neuron_row["pipeline_neuron_id"])
    path = path_for(PathKind.MOTION_STA, sta_dir_of_run(run_dir(settings, int(sta_run["id"]), bool(sta_run["is_test"]))), cell=cell_id)
    if not path.exists():
        return _message("Motion diagnostics", "Not computed for this cell. Re-run STA Maps (motion diagnostics need shifts that match the analysed frames).")
    with np.load(path, allow_pickle=True) as d:
        names = [str(n) for n in d["series_names"]]
        rels = np.asarray(d["rels_analysis"])
        observed, null_mean, null_sd = d["observed"], d["null_mean"], d["null_sd"]
        p, z = np.asarray(d["p"], dtype=float), np.asarray(d["z"], dtype=float)
        flag, min_p, alpha = bool(d["flag"]), float(d["min_p"]), float(d["flag_alpha"])
        p_floor = float(d["p_floor"])
        n_peaks = int(d["n_peaks_used"])
    tile_w, tile_h, gap, margin, head = 430, 210, 18, 22, 84
    cols, nrows = 3, int(np.ceil(len(names) / 3))
    width = margin * 2 + cols * tile_w + (cols - 1) * gap
    height = head + nrows * tile_h + (nrows - 1) * gap + margin
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 24), f"{neuron_row['canonical_id']} | Cell {cell_id} | Spike-triggered motion ({n_peaks} spikes)", fill=TEXT, font=_font(22), anchor="la")
    verdict = (
        f"MOTION-LOCKED: spikes are time-locked to motion (min p={min_p:.3f} <= {alpha:g}); treat this cell's STA clusters with caution"
        if flag else
        f"Cannot resolve p <= {alpha:g}: only {int(round(1 / p_floor)) - 1} permutations (smallest possible p = {p_floor:.3f}). Read the curves vs the grey band, or re-run with more permutations"
        if p_floor > alpha else
        f"No spike-locked motion detected (min p={min_p:.3f} > {alpha:g}); shifts shown are the corrected motion, residual motion is only visible via registration quality"
        if np.isfinite(min_p) else "Not enough spikes for the motion test"
    )
    draw.text((margin, 58), verdict, fill=DEP_COLOR if flag else MUTED, font=_font(15), anchor="la")
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        tile = _draw_motion_tile(rels, observed[i], null_mean[i], null_sd[i], MOTION_SERIES_LABELS.get(name, name), float(p[i]), float(z[i]), (tile_w, tile_h))
        canvas.paste(tile, (margin + c * (tile_w + gap), head + r * (tile_h + gap)))
    return _image_response(canvas)
