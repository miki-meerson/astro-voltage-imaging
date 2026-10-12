"""
based on: @caichangjia
"""
import os
import warnings
import tempfile
import shutil
import hashlib
import gc
from contextlib import contextmanager

warnings.filterwarnings('ignore')
from scipy.io import savemat
import tifffile
import json
import sys
import numpy as np
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.volpy.volparams import volparams
from caiman.summary_images import mean_image
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import pipeline_constants as consts
from utils import pipeline_utils as pipe_utils

##### GUI params extraction functions #####

def extract_params(gui_param_path):
    with open(gui_param_path, 'r') as fp:
        gui_params = json.load(fp)
    video_path = gui_params[consts.RAW_VIDEO_PATH_LINUX]
    gui_time = gui_params[consts.GUI_TIME]
    split_2ch_flag = gui_params.get(consts.IS_2CH, False)

    is2p = "hyp" in str(video_path).lower()
    fr = pipe_utils.get_frame_rate(video_path, is2p=is2p)
    mc_dict = extract_mc_params(gui_params, fr)
    return video_path, gui_time, mc_dict, split_2ch_flag


def extract_mc_params(gui_params, fr):
    mc_params = {consts.PW_RIGID, consts.GSIG_FILT,
                 consts.MAX_SHIFTS_X, consts.MAX_SHIFTS_Y,
                 consts.STRIDES_X, consts.STRIDES_Y, consts.OVERLAPS_X,
                 consts.OVERLAPS_Y, consts.MAX_DEVIATION_RIGID}
    mc_dict = {key: gui_params[key] for key in gui_params.keys() & mc_params}
    for key in mc_dict.keys():
        if key != consts.PW_RIGID:
            mc_dict[key] = int(mc_dict[key])
    if mc_dict[consts.GSIG_FILT] == 0:  # without high pass filtering
        mc_dict[consts.GSIG_FILT] = None
    else:
        mc_dict[consts.GSIG_FILT] = [mc_dict[consts.GSIG_FILT],
                                     0]  # 0 is just placeholder and doesnt really used in the mc code
    mc_dict[consts.STRIDES] = (mc_dict[consts.STRIDES_X], mc_dict[consts.STRIDES_Y])
    mc_dict[consts.OVERLAPS] = (mc_dict[consts.OVERLAPS_X], mc_dict[consts.OVERLAPS_Y])
    mc_dict[consts.MAX_SHIFTS] = (mc_dict[consts.MAX_SHIFTS_X], mc_dict[consts.MAX_SHIFTS_Y])
    to_delete = [consts.MAX_SHIFTS_X, consts.MAX_SHIFTS_Y, consts.STRIDES_X, consts.STRIDES_Y, consts.OVERLAPS_X,
                 consts.OVERLAPS_Y, ]
    for key in to_delete:
        del mc_dict[key]
    mc_dict[consts.FRAME_RATE] = fr
    return mc_dict


##### Caiman motion correction #####

@contextmanager
def isolated_motion_correction_workspace(base_dir):
    """
    Run CaImAn inside a per-job scratch directory so concurrent jobs do not
    reuse intermediate memmaps or relative output paths.
    """
    pipe_utils.mkdir(base_dir)
    workspace = tempfile.mkdtemp(prefix="caiman_mc_", dir=base_dir)
    prev_cwd = os.getcwd()
    try:
        os.chdir(workspace)
        yield workspace
    finally:
        os.chdir(prev_cwd)
        shutil.rmtree(workspace, ignore_errors=True)


def prepare_unique_motion_input(video_path, workspace_dir):
    """
    CaImAn derives memmap/output names from the input filename. The pipeline
    reuses generic names like raw_video.tif and 2ch_split_neuron.tif across
    many runs, so concurrent jobs can collide if CaImAn writes to a shared
    temp location. Give each run a unique local filename first.
    """
    abs_video_path = os.path.abspath(video_path)
    _, ext = os.path.splitext(abs_video_path)
    path_hash = hashlib.sha1(abs_video_path.encode("utf-8")).hexdigest()[:12]
    unique_input_path = os.path.join(workspace_dir, f"mc_input_{path_hash}{ext.lower()}")

    try:
        os.symlink(abs_video_path, unique_input_path)
    except OSError:
        shutil.copy2(abs_video_path, unique_input_path)

    return unique_input_path


def run_motion_correction(video_path, mc_params, workspace_root):
    def set_mc_parameters(video_path, mc_dict):
        opts_dict = mc_dict
        opts_dict["fnames"] = video_path
        opts_dict["border_nan"] = 'copy'
        return volparams(params_dict=opts_dict)

    def cluster_setup():
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
        return dview

    def resolve_mmap_path(mmap_file):
        if isinstance(mmap_file, (list, tuple)):
            candidates = [str(p) for p in mmap_file if p]
        else:
            candidates = [str(mmap_file)] if mmap_file else []

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(f"Motion correction memmap file not found. Candidates: {candidates}")

    with isolated_motion_correction_workspace(workspace_root) as workspace_dir:
        mc_input_path = prepare_unique_motion_input(video_path, workspace_dir)
        opts = set_mc_parameters(mc_input_path, dict(mc_params))
        dview = cluster_setup()

        try:
            try:
                mc = MotionCorrect(mc_input_path, dview=dview, **opts.get_group('motion'))
                mc.motion_correct(save_movie=True)
            except Exception as e:
                print(str(e))
                print("motion correction failed - trying to run another one with reduce shifts")
                retry_params = dict(mc_params)
                retry_params[consts.MAX_SHIFTS] = (10, 10)
                retry_params["fnames"] = mc_input_path
                retry_params["border_nan"] = "copy"
                opts = volparams(params_dict=retry_params)
                mc = MotionCorrect(mc_input_path, dview=dview, **opts.get_group('motion'))
                mc.motion_correct(save_movie=True)

            mmap_path = resolve_mmap_path(mc.mmap_file)
            movie_mmap = cm.load(mmap_path)
            movie = np.asarray(movie_mmap, dtype=np.float32).copy()
            del movie_mmap

            mean_img = mean_image(mmap_path, window=1000, dview=dview)
            return movie, mc.shifts_rig, mean_img
        finally:
            try:
                cm.stop_server(dview=dview)
            except Exception:
                pass


def apply_reg_shifts_and_save_movie(movie_path, shifts_mat, pipeline_dir, target_dtype, channel_name):
    mc_dir = os.path.join(pipeline_dir, consts.MC_DIR)
    pipe_utils.mkdir(mc_dir)
    path = os.path.join(mc_dir, f"{channel_name}_{consts.MC_VIDEO_PATH}")
    mean_path = os.path.join(mc_dir, f"{channel_name}_{consts.MEAN_IMAGE}")

    print(f"Applying shifts to movie and saving: {movie_path}")
    with tifffile.TiffFile(movie_path) as tif:
        series = tif.series[0]
        if len(series.shape) != 3:
            raise ValueError(f"Expected a 3D movie, got shape {series.shape}")
        n_frames, n_row, n_col = (int(v) for v in series.shape)
        mean_acc = np.zeros((n_row, n_col), dtype=np.float64)
        with tifffile.TiffWriter(path, bigtiff=True) as writer:
            for t, page in enumerate(series.pages):
                dy, dx = shifts_mat[t]
                M = np.float32([[1, 0, dx],
                                [0, 1, dy]])
                frame = page.asarray().astype(np.float32, copy=False)
                corrected = cv2.warpAffine(
                    frame,
                    M,
                    (n_col, n_row),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT,
                )
                mean_acc += corrected
                writer.write(pipe_utils.cast_movie_for_tiff_save(corrected, target_dtype), contiguous=True)
                if t and t % 500 == 0:
                    print(f"Saved shifted {channel_name} frame {t}/{n_frames}", flush=True)
    np.save(mean_path, mean_acc / max(1, n_frames))
    return path


##### saving functions #####
def save_corrected_movie(movie, pipeline_dir, mean_image, target_dtype, channel_name=None):
    mc_dir = os.path.join(pipeline_dir, consts.MC_DIR)
    pipe_utils.mkdir(mc_dir)

    if channel_name is None:
        path = os.path.join(mc_dir, consts.MC_VIDEO_PATH)
    else:
        path = os.path.join(mc_dir, f"{channel_name}_{consts.MC_VIDEO_PATH}")
    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        for frame_index in range(int(movie.shape[0])):
            writer.write(pipe_utils.cast_movie_for_tiff_save(movie[frame_index], target_dtype), contiguous=True)
            if frame_index and frame_index % 500 == 0:
                print(f"Saved corrected {channel_name or 'full'} frame {frame_index}/{movie.shape[0]}", flush=True)

    mean_img = np.array(mean_image.tolist())
    if channel_name is None:
        np.save(os.path.join(mc_dir, consts.MEAN_IMAGE), mean_img)
    else:
        np.save(os.path.join(mc_dir, f"{channel_name}_{consts.MEAN_IMAGE}"), mean_img)
    # TODO free memory (memmap)

def save_mc_shifts(pipeline_dir, shifts_mat):
    mc_dir = os.path.join(pipeline_dir, consts.MC_DIR)
    shifts_dir = os.path.join(mc_dir, consts.SHIFTS_DIR)
    pipe_utils.mkdir(shifts_dir)
    shifts_rig = {"motion_corrected_shifts": shifts_mat}
    savemat(os.path.join(shifts_dir, consts.MC_SHIFTS_PATH), shifts_rig)


def save_mc_traces(traces, pipeline_dir):
    mc_dir = os.path.join(pipeline_dir, consts.MC_DIR)
    traces_dir = os.path.join(mc_dir, consts.TRACES_DIR)
    pipe_utils.mkdir(mc_dir)
    pipe_utils.mkdir(traces_dir)
    path = os.path.join(traces_dir, consts.TRACES_PATH)
    traces.to_csv(path, index=False, header=False)
    return


def save_motion_qc_metrics(pipeline_dir, shifts_mat):
    qc_dir = os.path.join(pipeline_dir, consts.QC_DIR)
    pipe_utils.mkdir(qc_dir)

    shifts_arr = np.asarray(shifts_mat, dtype=np.float32)
    shift_mag = np.linalg.norm(shifts_arr, axis=1)
    qc_metrics = {
        consts.MEAN_XY_SHIFT: float(np.mean(shift_mag)),
        consts.MAX_SHIFT: float(np.max(shift_mag)),
        consts.STD_SHIFT: float(np.std(shift_mag))
    }

    qc_path = os.path.join(qc_dir, "motion_correction_qc.json")
    with open(qc_path, "w") as fp:
        json.dump(qc_metrics, fp, indent=2)


def main(args):
    gui_params_path = args[1]
    video_path, gui_time, mc_params, split_2ch_flag = extract_params(gui_params_path)
    print("GUI TIME:", gui_time)
    print("Motion Correcrion on:", video_path)
    pipeline_dir = pipe_utils.get_pipeline_results_dir(video_path)
    mc_workspace_root = os.path.join(pipeline_dir, consts.MC_DIR, "_caiman_workspace")
    video_path_tif = pipe_utils.raw_to_tif(video_path) if video_path.endswith(".raw") else video_path
    target_dtype = pipe_utils.get_signed_movie_dtype(video_path)

    # 2ch movies are 2P movies post splitting (astro and neuron channels)
    # note - not saving traces in 2P movies because there is no SLM
    if split_2ch_flag:
        split_dir = os.path.join(pipeline_dir, consts.SPLIT_DIR)
        split_astro_path = os.path.join(split_dir, consts.SPLIT_ASTRO_VIDEO_PATH)
        split_neuron_path = os.path.join(split_dir, consts.SPLIT_NEURON_VIDEO_PATH)

        mc_neuron, shifts_neuron, mean_image_neuron = run_motion_correction(
            split_neuron_path,
            mc_params,
            mc_workspace_root,
        )
        print("Motion correction applied to neuron movie")
        save_corrected_movie(mc_neuron, pipeline_dir, mean_image_neuron, target_dtype, channel_name="neuron")
        del mc_neuron, mean_image_neuron
        gc.collect()

        apply_reg_shifts_and_save_movie(split_astro_path, shifts_neuron, pipeline_dir, target_dtype, channel_name="astro")
        print("Motion correction applied to astrocyte movie")
        save_mc_shifts(pipeline_dir, shifts_neuron)
        save_motion_qc_metrics(pipeline_dir, shifts_neuron)
    else:
        mc_movie, shifts_mat, mean_image = run_motion_correction(
            video_path_tif,
            mc_params,
            mc_workspace_root,
        )
        save_corrected_movie(mc_movie, pipeline_dir, mean_image, target_dtype)
        save_mc_shifts(pipeline_dir, shifts_mat)
        save_motion_qc_metrics(pipeline_dir, shifts_mat)
        rois = pipe_utils.get_rois_mask(video_path)
        traces = pipe_utils.trace_extraction(mc_movie, rois)
        save_mc_traces(traces, pipeline_dir)

    print(consts.STEP_COMPLETED)

    return


if __name__ == "__main__":
    main(sys.argv)
