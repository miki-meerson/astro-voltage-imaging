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

from utils import globals_utils as g
from utils.file_utils import make_shared_dir
from utils.movie_utils import cast_movie_for_tiff_save, get_signed_movie_dtype, read_frame_rate
from utils.paths_utils import PathKind, channel_file_name, path_for

# keys of the params json (the GUI parameters of motion correction) and of CaImAn's motion-correction parameters; used by this step only
PW_RIGID = "pw_rigid"
GSIG_FILT = "gSig_filt"
MAX_SHIFTS = "max_shifts"
MAX_SHIFTS_X = "max_shifts_x"
MAX_SHIFTS_Y = "max_shifts_y"
STRIDES = "strides"
STRIDES_X = "strides_x"
STRIDES_Y = "strides_y"
OVERLAPS = "overlaps"
OVERLAPS_X = "overlaps_x"
OVERLAPS_Y = "overlaps_y"
MAX_DEVIATION_RIGID = "max_deviation_rigid"
FRAME_RATE = "frame_rate"
# keys of the motion-correction QC json
MEAN_XY_SHIFT = "mean_xy_shift"
MAX_SHIFT = "max_shift"
STD_SHIFT = "std_shift"
CAIMAN_WORKSPACE_DIR = "_caiman_workspace"  # scratch folder inside motion_corrected/, removed after each run

##### GUI params extraction functions #####

def extract_params(gui_param_path):
    with open(gui_param_path, 'r') as fp:
        gui_params = json.load(fp)
    video_path = gui_params[g.PARAM_RAW_VIDEO_PATH_LINUX]
    gui_time = gui_params[g.PARAM_GUI_TIME]

    fr = read_frame_rate(os.path.dirname(video_path))
    mc_dict = extract_mc_params(gui_params, fr)
    return video_path, gui_time, mc_dict


def extract_mc_params(gui_params, fr):
    mc_params = {PW_RIGID, GSIG_FILT,
                 MAX_SHIFTS_X, MAX_SHIFTS_Y,
                 STRIDES_X, STRIDES_Y, OVERLAPS_X,
                 OVERLAPS_Y, MAX_DEVIATION_RIGID}
    mc_dict = {key: gui_params[key] for key in gui_params.keys() & mc_params}
    for key in mc_dict.keys():
        if key != PW_RIGID:
            mc_dict[key] = int(mc_dict[key])
    if mc_dict[GSIG_FILT] == 0:  # without high pass filtering
        mc_dict[GSIG_FILT] = None
    else:
        mc_dict[GSIG_FILT] = [mc_dict[GSIG_FILT],
                                     0]  # 0 is just placeholder and doesnt really used in the mc code
    mc_dict[STRIDES] = (mc_dict[STRIDES_X], mc_dict[STRIDES_Y])
    mc_dict[OVERLAPS] = (mc_dict[OVERLAPS_X], mc_dict[OVERLAPS_Y])
    mc_dict[MAX_SHIFTS] = (mc_dict[MAX_SHIFTS_X], mc_dict[MAX_SHIFTS_Y])
    to_delete = [MAX_SHIFTS_X, MAX_SHIFTS_Y, STRIDES_X, STRIDES_Y, OVERLAPS_X,
                 OVERLAPS_Y, ]
    for key in to_delete:
        del mc_dict[key]
    mc_dict[FRAME_RATE] = fr
    return mc_dict


##### Caiman motion correction #####

@contextmanager
def isolated_motion_correction_workspace(base_dir):
    """
    Run CaImAn inside a per-job scratch directory so concurrent jobs do not
    reuse intermediate memmaps or relative output paths.
    """
    make_shared_dir(base_dir)
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
                retry_params[MAX_SHIFTS] = (10, 10)
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


def _mc_file(home_dir, channel_name, name):
    """Path of a file of one channel in the motion_corrected folder of the experiment (the folder is created)."""
    make_shared_dir(path_for(PathKind.MOTION_CORRECTED_DIR, home_dir))
    return path_for(PathKind.MC_FILE, home_dir, name=channel_file_name(channel_name, name))


def apply_reg_shifts_and_save_movie(movie_path, shifts_mat, home_dir, target_dtype, channel_name):
    path = _mc_file(home_dir, channel_name, g.MC_MOVIE)
    mean_path = _mc_file(home_dir, channel_name, g.MEAN_IMAGE_FILE)

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
                writer.write(cast_movie_for_tiff_save(corrected, target_dtype), contiguous=True)
                if t and t % 500 == 0:
                    print(f"Saved shifted {channel_name} frame {t}/{n_frames}", flush=True)
    np.save(mean_path, mean_acc / max(1, n_frames))
    return path


##### saving functions #####
def save_corrected_movie(movie, home_dir, mean_image, target_dtype, channel_name):
    path = _mc_file(home_dir, channel_name, g.MC_MOVIE)
    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        for frame_index in range(int(movie.shape[0])):
            writer.write(cast_movie_for_tiff_save(movie[frame_index], target_dtype), contiguous=True)
            if frame_index and frame_index % 500 == 0:
                print(f"Saved corrected {channel_name} frame {frame_index}/{movie.shape[0]}", flush=True)

    np.save(_mc_file(home_dir, channel_name, g.MEAN_IMAGE_FILE), np.array(mean_image.tolist()))
    # TODO free memory (memmap)

def save_mc_shifts(home_dir, shifts_mat):
    shifts_path = path_for(PathKind.MC_SHIFTS, home_dir)
    make_shared_dir(shifts_path.parent)
    savemat(str(shifts_path), {g.MOTION_SHIFTS_MAT_KEY: shifts_mat})


def save_motion_qc_metrics(home_dir, shifts_mat):
    make_shared_dir(path_for(PathKind.QC_DIR, home_dir))

    shifts_arr = np.asarray(shifts_mat, dtype=np.float32)
    shift_mag = np.linalg.norm(shifts_arr, axis=1)
    qc_metrics = {
        MEAN_XY_SHIFT: float(np.mean(shift_mag)),
        MAX_SHIFT: float(np.max(shift_mag)),
        STD_SHIFT: float(np.std(shift_mag))
    }

    qc_path = path_for(PathKind.QC_FILE, home_dir, name=g.MC_QC_FILE)
    with open(qc_path, "w") as fp:
        json.dump(qc_metrics, fp, indent=2)


def main(args):
    gui_params_path = args[1]
    video_path, gui_time, mc_params = extract_params(gui_params_path)
    print("GUI TIME:", gui_time)
    print("Motion Correcrion on:", video_path)
    home_dir = os.path.dirname(video_path)
    mc_workspace_root = path_for(PathKind.MOTION_CORRECTED_DIR, home_dir) / CAIMAN_WORKSPACE_DIR
    target_dtype = get_signed_movie_dtype(video_path)

    # the movie is a 2P movie post splitting (astro and neuron channels); no traces are saved because there is no SLM
    split_astro_path = path_for(PathKind.SPLIT_ASTRO_MOVIE, home_dir)
    split_neuron_path = path_for(PathKind.SPLIT_NEURON_MOVIE, home_dir)

    mc_neuron, shifts_neuron, mean_image_neuron = run_motion_correction(
        split_neuron_path,
        mc_params,
        mc_workspace_root,
    )
    print("Motion correction applied to neuron movie")
    save_corrected_movie(mc_neuron, home_dir, mean_image_neuron, target_dtype, channel_name="neuron")
    del mc_neuron, mean_image_neuron
    gc.collect()

    apply_reg_shifts_and_save_movie(split_astro_path, shifts_neuron, home_dir, target_dtype, channel_name="astro")
    print("Motion correction applied to astrocyte movie")
    save_mc_shifts(home_dir, shifts_neuron)
    save_motion_qc_metrics(home_dir, shifts_neuron)

    print(g.STEP_COMPLETED_MESSAGE)

    return


if __name__ == "__main__":
    main(sys.argv)
