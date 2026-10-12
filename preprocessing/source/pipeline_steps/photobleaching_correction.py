import sys
import os
import json
import numpy as np
import tifffile
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import pipeline_constants as consts
from utils import pipeline_utils as pipe_utils

##### GUI params extraction functions #####

def extract_params(gui_param_path):
    with open(gui_param_path, 'r') as fp:
        gui_params = json.load(fp)

    raw_video_path = gui_params[consts.RAW_VIDEO_PATH_LINUX]

    pipeline_dir = pipe_utils.get_pipeline_results_dir(raw_video_path)
    gui_time = gui_params[consts.GUI_TIME]
    trimmed = gui_params.get(consts.TRIMMED, 0)
    mc_dir = os.path.join(pipeline_dir, consts.MC_DIR)
    split_2ch_flag = gui_params.get(consts.IS_2CH, False)

    is2p = "hyp" in str(raw_video_path).lower()
    fr = pipe_utils.get_frame_rate(raw_video_path, is2p=is2p)

    if split_2ch_flag:
        channels = ["neuron", "astro"]
        mc_video_paths = {
            ch: os.path.join(mc_dir, f"{ch}_{consts.MC_VIDEO_PATH}")
            for ch in channels
        }
    else:
        channels = ["full"]
        mc_video_paths = {
            "full": os.path.join(mc_dir, consts.MC_VIDEO_PATH)
        }

    return raw_video_path, mc_video_paths, channels, gui_time, fr, trimmed


def run_photobleaching_correction(fr, start_frame, stop_frame, mc_path):
    start_frame = int(start_frame) if start_frame is not None else 0
    full_mean_trace = _movie_mean_trace(mc_path)
    n_total_frames = int(full_mean_trace.shape[0])
    if start_frame < 0:
        start_frame = 0
    if start_frame >= n_total_frames:
        raise ValueError(
            f"Trimmed start_frame ({start_frame}) is outside movie length ({n_total_frames})."
        )
    p = full_mean_trace[start_frame:].astype(np.float64, copy=False)
    n_frames = int(p.shape[0])

    stop = n_frames if stop_frame is None else int(stop_frame)

    frame_idx = np.arange(start_frame, start_frame + n_frames, dtype=np.int64)
    t_rel = np.arange(n_frames, dtype=np.float64) / fr
    t_abs = frame_idx / fr

    # Fitting range
    q = p[:stop]
    n_fit = len(q)

    # Initial parameter guesses
    offs = np.mean(q[int(0.75 * n_fit):])
    amp = np.mean(q[:int(0.25 * n_fit)]) - offs
    tau_init = 1000.0

    # Exponential function
    def expf(tt, c, a, tau):
        return c + a * np.exp(-tt / tau)

    # Fit parameters [offset, amplitude, tau] on time relative to trimmed start
    initial_params = np.array([max(offs, 0.0), max(amp, 0.0), max(tau_init, 1.0 / fr)], dtype=np.float64)
    lower_bounds = np.array([0.0, 0.0, 1.0 / fr], dtype=np.float64)
    upper_bounds = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    try:
        fitted_params, _ = curve_fit(
            expf,
            t_rel[:stop],
            p[:stop],
            p0=initial_params,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000,
        )
    except Exception:
        fitted_params = initial_params
    fitted_curve = expf(t_rel, *fitted_params)
    fit_range = p[:stop]
    fit_pred = fitted_curve[:stop]
    fit_residual = fit_range - fit_pred
    fit_denom = np.sum((fit_range - np.mean(fit_range)) ** 2)
    bleach_r2 = 1.0 - (np.sum(fit_residual ** 2) / fit_denom) if fit_denom > 0 else 0.0

    # ---- Extract parameters ----
    pb_correct_dict = {'C': fitted_params[0],
                       'A': fitted_params[1],
                       'tau': fitted_params[2],
                       'fitted_curve': fitted_curve}

    scale = float(np.mean(p))
    fitted_safe = np.where(np.abs(fitted_curve) > np.finfo(np.float64).eps, fitted_curve, np.finfo(np.float64).eps)
    corrected_mean = (p / fitted_safe) * scale
    slope, intercept = np.polyfit(t_rel, corrected_mean, 1)
    eps = np.finfo(np.float32).eps
    f0 = np.percentile(corrected_mean, 20)
    if abs(f0) < eps:
        f0 = eps
    global_dff = (corrected_mean - f0) / f0
    dff_centered = global_dff - np.mean(global_dff)
    dff_std = float(np.std(global_dff))
    dff_skewness = 0.0
    if dff_std > 0:
        dff_skewness = float(np.mean((dff_centered / dff_std) ** 3))

    qc_metrics = {
        consts.TRIMMED: int(start_frame),
        consts.BLEACH_TAU: float(fitted_params[2]),
        consts.BLEACH_C: float(fitted_params[0]),
        consts.BLEACH_R2: float(bleach_r2),
        consts.RESIDUAL_SLOPE_POST_CORRECTION: float(abs(slope)),
        consts.CORRECTED_MEAN_INTENSITY: float(np.mean(corrected_mean)),
        consts.GLOBAL_DFF_STD: dff_std,
        consts.GLOBAL_DFF_MEAN: float(np.mean(global_dff)),
        consts.GLOBAL_DFF_SKENESS: dff_skewness
    }

    pb_correct_dict["original_mean"] = p
    pb_correct_dict["full_original_mean"] = full_mean_trace
    pb_correct_dict["corrected_mean"] = corrected_mean
    pb_correct_dict["frame_idx"] = frame_idx
    pb_correct_dict["time_sec"] = t_abs
    pb_correct_dict["corrected_mean_fit_line"] = slope * t_rel + intercept
    pb_correct_dict["qc_metrics"] = qc_metrics
    pb_correct_dict["mc_path"] = mc_path
    pb_correct_dict["start_frame"] = int(start_frame)
    pb_correct_dict["scale"] = scale

    return pb_correct_dict


def _movie_mean_trace(movie_path):
    with tifffile.TiffFile(movie_path) as tif:
        series = tif.series[0]
        n_frames = int(series.shape[0])
        means = np.empty(n_frames, dtype=np.float64)
        for idx, page in enumerate(series.pages):
            means[idx] = float(np.mean(page.asarray()))
    return means


def _write_corrected_movie_streaming(pb_movie_path, pb_correct_dict, raw_video_path):
    target_dtype = pipe_utils.get_signed_movie_dtype(raw_video_path)
    mc_path = pb_correct_dict["mc_path"]
    start_frame = int(pb_correct_dict["start_frame"])
    fitted_curve = np.asarray(pb_correct_dict["fitted_curve"], dtype=np.float64)
    scale = float(pb_correct_dict["scale"])
    eps = np.finfo(np.float64).eps

    with tifffile.TiffFile(mc_path) as tif, tifffile.TiffWriter(pb_movie_path, bigtiff=True) as writer:
        pages = tif.series[0].pages
        for out_idx, page in enumerate(pages[start_frame:]):
            denom = float(fitted_curve[out_idx])
            if abs(denom) < eps:
                denom = eps
            frame = np.asarray(page.asarray(), dtype=np.float32)
            corrected = (frame / denom) * scale
            writer.write(pipe_utils.cast_movie_for_tiff_save(corrected, target_dtype), contiguous=True)


def save_pb_correct_data(pipeline_dir, pb_correct_dict, raw_video_path, channel_name=None):
    pb_dir = os.path.join(pipeline_dir, consts.PB_DIR)
    pipe_utils.mkdir(pb_dir)

    if channel_name is None:
        pb_movie_path = os.path.join(pb_dir, consts.PB_VIDEO_PATH)
        pb_fit_path = os.path.join(pb_dir, consts.PB_FIT_PATH)
    else:
        pb_movie_path = os.path.join(pb_dir, f"{channel_name}_{consts.PB_VIDEO_PATH}")
        pb_fit_path = os.path.join(pb_dir, f"{channel_name}_{consts.PB_FIT_PATH}")

    _write_corrected_movie_streaming(pb_movie_path, pb_correct_dict, raw_video_path)

    np.savez(
        pb_fit_path,
        fitted_curve=pb_correct_dict["fitted_curve"],
        A=pb_correct_dict["A"],
        tau=pb_correct_dict["tau"],
        C=pb_correct_dict["C"]
    )


def save_pb_qc(pipeline_dir, pb_correct_dict, channel_name=None):
    qc_dir = os.path.join(pipeline_dir, consts.QC_DIR)
    pipe_utils.mkdir(qc_dir)

    suffix = "" if channel_name is None else f"_{channel_name}"
    qc_json_path = os.path.join(qc_dir, f"photobleaching_qc{suffix}.json")
    with open(qc_json_path, "w") as fp:
        json.dump(pb_correct_dict["qc_metrics"], fp, indent=2)

    frame_idx = pb_correct_dict["frame_idx"]
    original_mean = pb_correct_dict["original_mean"]
    full_original_mean = pb_correct_dict.get("full_original_mean")
    corrected_mean = pb_correct_dict["corrected_mean"]
    fitted_curve = pb_correct_dict["fitted_curve"]
    corrected_fit_line = pb_correct_dict["corrected_mean_fit_line"]
    trimmed = int(pb_correct_dict["qc_metrics"].get(consts.TRIMMED, 0))

    trace_path = os.path.join(qc_dir, f"photobleaching_qc_trace{suffix}.npz")
    np.savez_compressed(
        trace_path,
        frame_idx=frame_idx,
        time_sec=pb_correct_dict["time_sec"],
        original_mean=original_mean,
        full_original_mean=full_original_mean if full_original_mean is not None else np.array([], dtype=float),
        fitted_curve=fitted_curve,
        corrected_mean=corrected_mean,
        corrected_mean_fit_line=corrected_fit_line,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    if full_original_mean is not None:
        full_frame_idx = np.arange(len(full_original_mean), dtype=np.int64)
        ax.plot(
            full_frame_idx,
            full_original_mean,
            label="Full mean intensity (pre-trim)",
            linewidth=1.0,
            alpha=0.35,
            color="gray",
        )
        if trimmed > 0:
            ax.axvline(trimmed, color="black", linestyle=":", linewidth=1.0, alpha=0.9, label="Trim start")

    ax.plot(frame_idx, original_mean, label="Original mean intensity", linewidth=1.5)
    ax.plot(frame_idx, fitted_curve, label="Fitted exponential", linewidth=1.5)
    ax.plot(frame_idx, corrected_mean, label="Corrected mean intensity", linewidth=1.5)
    ax.plot(frame_idx, corrected_fit_line, label="Corrected linear fit", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(f"Photobleaching QC (trimmed first {trimmed} frames)")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig_path = os.path.join(qc_dir, f"photobleaching_qc_plot{suffix}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main(args):
    gui_params_path = args[1]
    raw_video_path, mc_video_paths, channels, gui_time, fr, trimmed = extract_params(gui_params_path)
    print("GUI TIME:", gui_time)
    pipeline_dir = pipe_utils.get_pipeline_results_dir(raw_video_path)

    for ch in channels:
        mc_video_path = mc_video_paths[ch]
        print(f"Photobleaching correction on ({ch}):", mc_video_path)

        pb_correct_dict = run_photobleaching_correction(
            fr,
            start_frame=trimmed,
            stop_frame=None,
            mc_path=mc_video_path
        )

        save_pb_correct_data(
            pipeline_dir,
            pb_correct_dict,
            raw_video_path,
            channel_name=None if ch == "full" else ch
        )
        save_pb_qc(
            pipeline_dir,
            pb_correct_dict,
            channel_name=None if ch == "full" else ch
        )

    print(consts.STEP_COMPLETED)
    return


if __name__ == "__main__":
    main(sys.argv)
