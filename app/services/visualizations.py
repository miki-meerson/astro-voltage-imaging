from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
from pathlib import Path
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.services.runs import run_dir
from app.settings import Settings

RESULT_ORDER = {
    "all": 0,
    "simple": 1,
    "complex": 2,
    "anatomy": 3,
    "behavioral_active": 4,
    "behavioral_inactive": 5,
}
NEURON_COLOR = "#f7d154"
DEP_COLOR = "#ff5d6c"
HYPER_COLOR = "#4fa3ff"
OTHER_NEURON_COLOR = "#8a7661"
BG = "#10151c"
PANEL = "#0b0f14"
TEXT = "#edf3fa"
MUTED = "#9aa8b7"
TRACE_BG = "#111821"
TRACE_TEXT = "#d7e1ec"
_STA_SUMMARY_CACHE: OrderedDict[tuple, bytes] = OrderedDict()
_STA_SUMMARY_CACHE_LOCK = threading.Lock()
_STA_SUMMARY_CACHE_MAX = 80
_CLUSTER_RESULT_CACHE: OrderedDict[tuple[str, int], tuple] = OrderedDict()
_CLUSTER_RESULT_CACHE_LOCK = threading.Lock()
_CLUSTER_RESULT_CACHE_MAX = 240


def _font(size: int):
    for name in ("arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _prepare_imports(settings: Settings) -> None:
    return None


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


def _cache_get(key: tuple) -> bytes | None:
    with _STA_SUMMARY_CACHE_LOCK:
        value = _STA_SUMMARY_CACHE.get(key)
        if value is not None:
            _STA_SUMMARY_CACHE.move_to_end(key)
        return value


def _cache_put(key: tuple, value: bytes) -> None:
    with _STA_SUMMARY_CACHE_LOCK:
        _STA_SUMMARY_CACHE[key] = value
        _STA_SUMMARY_CACHE.move_to_end(key)
        while len(_STA_SUMMARY_CACHE) > _STA_SUMMARY_CACHE_MAX:
            _STA_SUMMARY_CACHE.popitem(last=False)


def _cluster_cache_get(key: tuple[str, int]):
    with _CLUSTER_RESULT_CACHE_LOCK:
        value = _CLUSTER_RESULT_CACHE.get(key)
        if value is not None:
            _CLUSTER_RESULT_CACHE.move_to_end(key)
        return value


def _cluster_cache_put(key: tuple[str, int], value) -> None:
    with _CLUSTER_RESULT_CACHE_LOCK:
        _CLUSTER_RESULT_CACHE[key] = value
        _CLUSTER_RESULT_CACHE.move_to_end(key)
        while len(_CLUSTER_RESULT_CACHE) > _CLUSTER_RESULT_CACHE_MAX:
            _CLUSTER_RESULT_CACHE.popitem(last=False)


def _mask_outline(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    try:
        from scipy.ndimage import binary_erosion

        outline = mask & ~binary_erosion(mask)
        return outline if outline.any() else mask
    except Exception:
        return mask
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
    new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    work = image.resize(new_size, Image.Resampling.BICUBIC)
    x = (size[0] - work.width) // 2
    y = (size[1] - work.height) // 2
    tile.alpha_composite(work.convert("RGBA"), (x, y))
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


def _draw_trace_tile(t_rel: np.ndarray, traces: list[tuple[np.ndarray, str]], size: tuple[int, int]) -> Image.Image:
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
    draw.text((16, h // 2), "dF/F", fill=TRACE_TEXT, font=_font(14), anchor="lm")
    return tile


def _placeholder_tile(message: str, size: tuple[int, int], bg: str = PANEL) -> Image.Image:
    tile = Image.new("RGBA", size, bg)
    draw = ImageDraw.Draw(tile)
    draw.text((size[0] // 2, size[1] // 2), message, fill=MUTED, font=_font(17), anchor="mm")
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


def _latest_completed_run(conn, experiment_id: int, step: str):
    return conn.execute(
        """
        SELECT * FROM analysis_runs
        WHERE experiment_id=? AND step=? AND is_test=0 AND status='completed'
        ORDER BY id DESC LIMIT 1
        """,
        (experiment_id, step),
    ).fetchone()


def _sta_dir(settings: Settings, run_row) -> Path:
    return run_dir(settings, int(run_row["id"]), bool(run_row["is_test"])) / "sta_algorithm_home" / "sta-analysis"


def _mirror_home(settings: Settings, run_row) -> Path:
    return run_dir(settings, int(run_row["id"]), bool(run_row["is_test"])) / "sta_algorithm_home"


def _cluster_polarity(cluster: dict) -> str:
    sign = str(cluster.get("sign", "")).lower()
    return "depolarizing" if sign == "positive" else "hyperpolarizing"


def _cluster_color(cluster: dict) -> str:
    return HYPER_COLOR if _cluster_polarity(cluster) == "hyperpolarizing" else DEP_COLOR


def _load_cluster_result(path: Path):
    from utils.clusters_utils import cluster_mask2d_from_cluster, extract_clusters

    cache_key = (str(path), int(path.stat().st_mtime_ns))
    cached = _cluster_cache_get(cache_key)
    if cached is not None:
        return cached

    with np.load(path, allow_pickle=True) as d:
        effect = np.asarray(d["observed_effect_movie"], dtype=np.float32)
        aw = np.asarray(d["analysis_window"], dtype=int)
        clusters = extract_clusters(
            effect_movie=effect,
            threshold=float(np.asarray(d["cluster_threshold"]).item()),
            connectivity=int(np.asarray(d["cluster_connectivity"]).item()),
            min_cluster_size=int(np.asarray(d["min_cluster_size"]).item()),
            min_cluster_mass=float(np.asarray(d["min_cluster_mass"]).item()),
            merge_similar=bool(np.asarray(d["merge_similar_clusters"]).item()),
            merge_max_distance_px=float(np.asarray(d["merge_max_distance_px"]).item()),
            merge_min_trace_corr=float(np.asarray(d["merge_min_trace_corr"]).item()),
            null_max=np.asarray(d["null_max_cluster_mass"], dtype=np.float32),
            cluster_alpha=float(np.asarray(d["cluster_alpha"]).item()),
        )
        metadata = {
            "spatial_bin": int(np.asarray(d["spatial_bin"]).item()) if "spatial_bin" in d else 1,
            "behavioral_group": str(np.asarray(d["behavioral_group"]).item()) if "behavioral_group" in d else None,
            "anatomy_crop_bounds": tuple(np.asarray(d["anatomy_crop_bounds"], dtype=int).tolist()) if "anatomy_crop_bounds" in d else None,
        }
    t_rel = np.arange(int(aw[0]), int(aw[1]) + 1)
    sig = []
    for cluster in clusters:
        if not cluster.get("is_significant", False):
            continue
        mask = cluster_mask2d_from_cluster(cluster)
        if mask is not None and mask.any():
            sig.append((cluster, mask))
    result = (effect, t_rel, sig, metadata)
    _cluster_cache_put(cache_key, result)
    return result


def _load_suite2p_stat_cells(suite2p_path: Path) -> np.ndarray:
    iscell = np.load(suite2p_path / "iscell.npy", allow_pickle=True)[:, 0].astype(bool)
    stat = np.load(suite2p_path / "stat.npy", allow_pickle=True)
    return stat[iscell]


def _neuron_mask(cell_id: int, shape: tuple[int, int], spatial_bin: int, stat_cells) -> np.ndarray:
    roi = stat_cells[int(cell_id)]
    yb = np.clip(np.floor_divide(np.asarray(roi["ypix"], dtype=int), spatial_bin), 0, shape[0] - 1)
    xb = np.clip(np.floor_divide(np.asarray(roi["xpix"], dtype=int), spatial_bin), 0, shape[1] - 1)
    mask = np.zeros(shape, dtype=bool)
    mask[yb, xb] = True
    return mask


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
    path = paths.sta_dir / f"astro_photobleaching_corrected_gevi_flipped_behavioral_{behavioral_group}_bin{spatial_bin}.npy" if behavioral_group else paths.pb_dir / f"astro_photobleaching_corrected_gevi_flipped_bin{spatial_bin}.npy"
    if not path.exists():
        return None
    movie = np.load(path, mmap_mode="r")
    step = max(1, int(np.ceil(movie.shape[0] / 1000)))
    cache[key] = np.asarray(movie[::step], dtype=np.float32).mean(axis=0)
    return cache[key]


def render_sta_cell_summary(settings: Settings, conn, neuron_row) -> BytesIO:
    _prepare_imports(settings)
    from utils.db_utils import collect_cluster_result_metadata
    from utils.paths_utils import resolve_sta_paths

    sta_run = _latest_completed_run(conn, int(neuron_row["experiment_id"]), "sta")
    if sta_run is None:
        return _message("STA Maps", "No completed STA maps run is available for this experiment.")
    sta_dir = _sta_dir(settings, sta_run)
    if not sta_dir.exists():
        return _message("STA Maps", f"STA output folder was not found: {sta_dir}")

    cell_id = int(neuron_row["pipeline_neuron_id"])
    result_df = collect_cluster_result_metadata(sta_dir, [cell_id])
    if result_df.empty:
        return _message("STA Maps", f"No STA result files were found for cell {cell_id}.")
    rows = result_df.sort_values("version").groupby("result_type", as_index=False).tail(1)
    rows = rows.assign(_order=rows["result_type"].map(lambda x: RESULT_ORDER.get(str(x), 100))).sort_values(["_order", "result_type"])
    has_anatomy_row = "anatomy" in set(rows["result_type"].astype(str))
    result_paths = [Path(row.path) for row in rows.itertuples()]
    source_path = neuron_row["source_path"]
    suite2p_path = Path(source_path) / "pipeline_results" / "for_suite2p" / "suite2p" / "plane0"
    source_mtimes = []
    for path in result_paths + [suite2p_path / "iscell.npy", suite2p_path / "stat.npy"]:
        source_mtimes.append((str(path), int(path.stat().st_mtime_ns) if path.exists() else 0))
    cache_key = (
        int(sta_run["id"]),
        int(cell_id),
        tuple((str(row.result_type), int(row.version)) for row in rows.itertuples()),
        tuple(source_mtimes),
        4,
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        return _bytes_response(cached)

    paths = resolve_sta_paths(_mirror_home(settings, sta_run))
    try:
        stat_cells = _load_suite2p_stat_cells(suite2p_path)
    except Exception as exc:
        return _message("STA Maps", f"Could not read Suite2p outlines: {exc}")

    show_missing_anatomy = False
    n = max(1, len(rows) + (1 if show_missing_anatomy else 0))
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
        title = str(row.result_type).replace("_", " ").title()
        if mean_img is None:
            image_tile = Image.new("RGBA", (tile_w, image_h), PANEL)
            ImageDraw.Draw(image_tile).text((tile_w // 2, image_h // 2), "missing mean image", fill=MUTED, font=_font(17), anchor="mm")
            traces = []
        else:
            neuron = _neuron_mask(cell_id, mean_img.shape, int(meta["spatial_bin"]), stat_cells)
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
        if mean_img is not None:
            for cluster, mask in clusters:
                if np.asarray(mask).shape != effect.shape[1:]:
                    continue
                trace = effect[:, mask].mean(axis=1)
                traces.append((trace, _cluster_color(cluster)))
        trace_tile = _draw_trace_tile(t_rel, traces, (tile_w, trace_h))
        prepared.append((title, int(row.n_peaks), int(row.n_significant_clusters), image_tile, trace_tile))

    if show_missing_anatomy:
        prepared.append(
            (
                "Anatomy",
                0,
                0,
                _placeholder_tile("Anatomy crop params exist,\nbut this STA run did not include anatomy.\nRerun STA Maps.", (tile_w, image_h)),
                _placeholder_tile("Waiting for anatomy result", (tile_w, trace_h), TRACE_BG),
            )
        )

    for col, (title, n_peaks, n_sig, image_tile, trace_tile) in enumerate(prepared):
        x = margin + col * (tile_w + gap)
        _draw_centered(draw, (x + tile_w // 2, top_h + 18), title, _font(19), TEXT)
        _draw_centered(draw, (x + tile_w // 2, top_h + 43), f"{n_peaks} peaks   {n_sig} significant clusters", _font(14), MUTED)
        canvas.paste(image_tile.convert("RGB"), (x, top_h + title_h))
        canvas.paste(trace_tile, (x, top_h + title_h + image_h + 34))

    response = _image_response(canvas)
    _cache_put(cache_key, response.getvalue())
    response.seek(0)
    return response
