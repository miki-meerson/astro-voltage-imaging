from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion

from app.db import run_repository
from app.services import caches, polarity
from app.services.runs import run_dir
from app.services.suite2p_data import experiment_stat_cells
from app.settings import Settings
from utils import globals_utils as g
from utils.cluster_result import cluster_basic_fields, load_cluster_result
from utils.clusters_utils import mask_mean_trace
from utils.db_utils import collect_cluster_result_metadata
from utils.hypotheses import hypothesis_name
from utils.movie_utils import get_mean_images, read_frame_rate
from utils.paths_utils import PathKind, path_for, resolve_sta_paths, sta_dir_of_run
from utils.suite2p_utils import suite2p_roi_mask

RESULT_ORDER = {
    g.ALL_GROUP: 0,
    g.ANATOMY_GROUP: 1,
    "behavioral_active": 2,
    "behavioral_inactive": 3,
}
NEURON_COLOR = "#f7d154"
PANEL = "#0b0f14"


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


def _overlay_mask_pixels(base: Image.Image, mask: np.ndarray, color: str, alpha: float = 0.9) -> Image.Image:
    if mask is None or not np.asarray(mask).any():
        return base
    if np.asarray(mask).shape != (base.height, base.width):
        return base
    mask = np.asarray(mask, dtype=bool)
    outline = mask & ~binary_erosion(mask)
    if not outline.any():
        outline = mask
    rgba = np.zeros((outline.shape[0], outline.shape[1], 4), dtype=np.uint8)
    color = color.lstrip("#")
    rgba[..., :3] = tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))
    rgba[..., 3] = (outline.astype(np.float32) * 255 * float(alpha)).astype(np.uint8)
    return Image.alpha_composite(base.convert("RGBA"), Image.fromarray(rgba, mode="RGBA"))


def _fit_tile(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    tile = Image.new("RGBA", size, PANEL)
    scale = min(size[0] / max(image.width, 1), size[1] / max(image.height, 1))
    work = image.resize((max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))), Image.Resampling.BICUBIC)
    tile.alpha_composite(work.convert("RGBA"), ((size[0] - work.width) // 2, (size[1] - work.height) // 2))
    return tile


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


def _mean_binned_image(paths, spatial_bin: int, behavioral_group: str | None = None) -> np.ndarray | None:
    """The mean of the binned movie (of a behavioral group when given), or None when that movie is not saved. Kept in memory per file (small)."""
    path = (
        path_for(PathKind.RUN_BEHAVIORAL_BINNED_MOVIE, paths.home_dir, group=behavioral_group, bin=int(spatial_bin))
        if behavioral_group
        else path_for(PathKind.BINNED_ASTRO_MOVIE, paths.home_dir, bin=int(spatial_bin))
    )
    if not path.exists():
        return None
    key = ("binned", str(path), int(path.stat().st_mtime_ns))
    image = caches.MEAN_IMAGE.get(key)
    if image is None:
        movie = np.load(path, mmap_mode="r")
        step = max(1, int(np.ceil(movie.shape[0] / 1000)))
        image = np.asarray(movie[::step], dtype=np.float32).mean(axis=0)
        caches.MEAN_IMAGE.put(key, image)
    return image


def _astro_mean_image(paths, spatial_bin: int) -> np.ndarray | None:
    key = ("astro_mean", str(paths.mc_dir), int(spatial_bin))
    image = caches.MEAN_IMAGE.get(key)
    if image is None:
        try:
            astro_mean, _ = get_mean_images(paths.mc_dir, spatial_bin=int(spatial_bin))
        except (OSError, ValueError):  # the motion-corrected mean image is not saved: the tile says so
            return None
        image = polarity.to_biological(astro_mean)  # astro_mean_image.npy is raw; the binned movies are already flipped, so flip this one to match
        caches.MEAN_IMAGE.put(key, image)
    return image


def _result_rows(settings: Settings, neuron_row, step: str):
    """(newest completed run of the hypothesis `step`, its sta-analysis folder, the newest version of each result file of the cell in display
    order). Raises FileNotFoundError saying what is missing."""
    sta_run = run_repository.latest_run(settings, int(neuron_row["experiment_id"]), step)
    if sta_run is None:
        raise FileNotFoundError(f"No completed run of hypothesis '{hypothesis_name(step)}' is available for this experiment.")
    sta_dir = sta_dir_of_run(run_dir(settings, int(sta_run["id"]), bool(sta_run["is_test"])))
    if not sta_dir.exists():
        raise FileNotFoundError(f"The STA output folder was not found: {sta_dir}")
    cell_id = int(neuron_row["pipeline_neuron_id"])
    result_df = collect_cluster_result_metadata(sta_dir, [cell_id])
    if result_df.empty:
        raise FileNotFoundError(f"No STA result files were found for cell {cell_id}.")
    rows = result_df.sort_values("version").groupby("result_type", as_index=False).tail(1)
    rows = rows.assign(_order=rows["result_type"].map(lambda x: RESULT_ORDER.get(str(x), 100))).sort_values(["_order", "result_type"])
    return sta_run, sta_dir, rows


def _load_result(path):
    """The parsed cluster result file (see load_cluster_result), kept in memory by file and modification time."""
    key = (str(path), int(Path(path).stat().st_mtime_ns))
    result = caches.CLUSTER_RESULT.get(key)
    if result is None:
        result = load_cluster_result(path)
        caches.CLUSTER_RESULT.put(key, result)
    return result


def cell_result_cards(settings: Settings, neuron_row, step: str) -> list[dict]:
    """One card per result type of a cell for the Cells tab: {result_type, title, n_peaks, n_significant, tested_window, fps, clusters: [{polarity, p,
    first_rel, last_rel, n_px, t, trace}]} with plain numbers only (the page draws the plots). Raises FileNotFoundError when there is nothing to show."""
    _, _, rows = _result_rows(settings, neuron_row, step)
    fps = read_frame_rate(neuron_row["source_path"])
    cards = []
    for row in rows.itertuples():
        effect, t_rel, clusters, meta = _load_result(row.path)
        found = []
        for cluster, mask in clusters:
            if np.asarray(mask).shape != effect.shape[1:]:
                continue
            wide = cluster.get("wide_trace")  # displayed over a wider window than the tested one
            t_plot, trace = wide if wide is not None else (t_rel, mask_mean_trace(meta["trace_effect"], mask))
            found.append({
                **cluster_basic_fields(cluster, meta["tested_window"][0], meta["time_bin"]), "polarity": polarity.polarity_from_sign(str(cluster.get("sign", ""))),
                "n_px": int(np.asarray(mask).sum()), "t": [round(float(v), 2) for v in t_plot], "trace": [round(float(v), 5) for v in np.asarray(trace).ravel()],
            })
        cards.append({
            "result_type": str(row.result_type), "title": str(row.result_type).replace("_", " ").title(), "n_peaks": int(row.n_peaks),
            "n_significant": int(row.n_significant_clusters), "tested_window": [int(v) for v in meta["tested_window"]], "fps": fps, "clusters": found,
        })
    return cards


def _map_image(paths, stat_cells, cell_id: int, result) -> Image.Image | None:
    """The mean image of a result (cropped for the anatomy result) with its significant clusters (sign colours) and the cell outline drawn on it."""
    effect, _, clusters, meta = result
    spatial_bin, behavioral_group = int(meta["spatial_bin"]), meta["behavioral_group"]
    mean_img = _mean_binned_image(paths, spatial_bin, behavioral_group)
    if mean_img is None and behavioral_group is not None:
        mean_img = _mean_binned_image(paths, spatial_bin)
    if mean_img is None and behavioral_group is None:
        mean_img = _astro_mean_image(paths, spatial_bin)
    if mean_img is None:
        return None
    crop = meta["anatomy_crop_bounds"] if meta["anatomy_crop_bounds"] is not None and effect.shape[1:] != mean_img.shape else None
    neuron = suite2p_roi_mask(cell_id, stat_cells, mean_img.shape, spatial_bin)
    shown_img, shown_neuron = mean_img, neuron
    if crop is not None:
        y0, y1, x0, x1 = crop
        shown_img, shown_neuron = mean_img[y0:y1, x0:x1], neuron[y0:y1, x0:x1]
    image = _gray_rgb(shown_img)
    for cluster, mask in clusters:
        shown = _display_cluster_mask(mask, mean_img.shape, shown_img.shape, crop)
        if shown is not None:
            image = _overlay_mask_pixels(image, shown, _cluster_color(cluster), 0.95)
    return _overlay_mask_pixels(image, shown_neuron, NEURON_COLOR, 0.95)


MAP_SIZE = 640


def render_result_map(settings: Settings, neuron_row, step: str, result_type: str) -> BytesIO:
    """The square map image of one result type of a cell (no text: the page writes the labels)."""
    try:
        sta_run, _, rows = _result_rows(settings, neuron_row, step)
        match = rows[rows["result_type"].astype(str) == result_type]
        if match.empty:
            return _message("STA map", f"This cell has no '{result_type}' result in this run.")
        row = next(match.itertuples())
        key = ("map", int(sta_run["id"]), int(neuron_row["pipeline_neuron_id"]), result_type, int(row.version), int(Path(row.path).stat().st_mtime_ns), polarity.RAW_TO_BIOLOGICAL)
        cached = caches.STA_SUMMARY_PNG.get(key)
        if cached is not None:
            return BytesIO(cached)
        source_path = neuron_row["source_path"]
        paths = resolve_sta_paths(source_path, path_for(PathKind.STA_ALGORITHM_HOME, run_dir(settings, int(sta_run["id"]), bool(sta_run["is_test"]))))
        try:
            stat_cells = experiment_stat_cells(source_path)
        except (OSError, ValueError, IndexError, KeyError) as exc:
            return _message("STA map", f"Could not read Suite2p outlines: {exc}")
        image = _map_image(paths, stat_cells, int(neuron_row["pipeline_neuron_id"]), _load_result(row.path))
        if image is None:
            return _message("STA map", "The mean image of this result is missing.")
    except FileNotFoundError as exc:
        return _message("STA map", str(exc))
    response = _image_response(_fit_tile(image, (MAP_SIZE, MAP_SIZE)))
    caches.STA_SUMMARY_PNG.put(key, response.getvalue())
    response.seek(0)
    return response


MOTION_SERIES_LABELS = {
    "dy": "Registration shift dy (px)",
    "dx": "Registration shift dx (px)",
    "shift_mag": "Shift magnitude (px)",
    "jerk": "Frame-to-frame jerk (px)",
    "reg_corr": "Registration quality (corr. to mean)",
}


def _motion_verdict(flag: bool, p_family: float, alpha: float, p_floor: float) -> tuple[str, str]:
    """(kind, sentence) of the motion test of a cell: kind is flag, unresolved, none or few_spikes."""
    if flag:
        return "flag", f"The spikes of this cell are time-locked to motion (p = {p_family:.3f}, corrected over the 5 measures, limit {alpha:g}). Treat its clusters with caution."
    if p_floor > alpha:
        return "unresolved", f"Cannot resolve p <= {alpha:g}: only {int(round(1 / p_floor)) - 1} random copies (the smallest possible p is {p_floor:.3f}). Read the curves against the grey band."
    if np.isfinite(p_family):
        return "none", f"No spike-locked motion found (p = {p_family:.3f}, corrected over the 5 measures, limit {alpha:g}). The shifts are the motion that was corrected; motion left after the correction shows only in the registration quality."
    return "few_spikes", "There are not enough spikes for the motion test."


def cell_motion(settings: Settings, neuron_row, step: str) -> dict:
    """The motion test of a cell from the run of the hypothesis `step`, as plain numbers for the page to draw: {n_peaks, p, alpha, verdict, text, fps,
    series: [{name, label, rels, observed, null_mean, null_sd, p, z}]}. Raises FileNotFoundError when the run or the test result is missing."""
    sta_run = run_repository.latest_run(settings, int(neuron_row["experiment_id"]), step)
    if sta_run is None:
        raise FileNotFoundError(f"No completed run of hypothesis '{hypothesis_name(step)}' is available for this experiment.")
    path = path_for(PathKind.MOTION_STA, sta_dir_of_run(run_dir(settings, int(sta_run["id"]), bool(sta_run["is_test"]))), cell=int(neuron_row["pipeline_neuron_id"]))
    if not path.exists():
        raise FileNotFoundError("The motion test was not computed for this cell. The STA Maps step writes it: rerun the analysis for this hypothesis.")

    def numbers(values):
        return [None if not np.isfinite(v) else round(float(v), 6) for v in np.asarray(values, dtype=float).ravel()]

    with np.load(path, allow_pickle=True) as d:
        names = [str(n) for n in d["series_names"]]
        rels = [int(v) for v in np.asarray(d["rels_analysis"]).ravel()]
        flag, p_family, alpha, p_floor = bool(d["flag"]), float(d["p_family"]), float(d["flag_alpha"]), float(d["p_floor"])
        series = [{
            "name": name, "label": MOTION_SERIES_LABELS.get(name, name), "rels": rels, "observed": numbers(d["observed"][i]), "null_mean": numbers(d["null_mean"][i]),
            "null_sd": numbers(d["null_sd"][i]), "p": numbers([d["p"][i]])[0], "z": numbers([d["z"][i]])[0],
        } for i, name in enumerate(names)]
        n_peaks = int(d["n_peaks_used"])
    verdict, text = _motion_verdict(flag, p_family, alpha, p_floor)
    return {"n_peaks": n_peaks, "p": None if not np.isfinite(p_family) else p_family, "alpha": alpha, "verdict": verdict, "text": text, "series": series,
            "fps": read_frame_rate(neuron_row["source_path"])}
