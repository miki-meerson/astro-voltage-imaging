import sys
import os
import json
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import pipeline_constants as consts
from utils import pipeline_utils as pipe_utils


##### GUI params extraction functions #####

def extract_params(gui_param_path):
    with open(gui_param_path, "r") as fp:
        gui_params = json.load(fp)

    raw_video_path = gui_params[consts.RAW_VIDEO_PATH_LINUX]
    fr = pipe_utils.get_frame_rate(raw_video_path, is2p=True)
    pipeline_dir = pipe_utils.get_pipeline_results_dir(raw_video_path)
    gui_time = gui_params[consts.GUI_TIME]

    return raw_video_path, gui_time, fr


def _write_tiff_pages(input_path, astro_path, neuron_path, target_dtype):
    with tifffile.TiffFile(input_path) as tif:
        series = tif.series[0]
        if len(series.shape) != 3:
            raise ValueError(f"Expected a 3D movie, got shape {series.shape}")
        n_frames = int(series.shape[0])
        middle_frame = n_frames // 2
        if n_frames % 2:
            print(
                f"Warning: odd frame count {n_frames}; astro gets {middle_frame}, neuron gets {n_frames - middle_frame}.",
                flush=True,
            )
        with tifffile.TiffWriter(astro_path, bigtiff=True) as astro_writer, tifffile.TiffWriter(neuron_path, bigtiff=True) as neuron_writer:
            for frame_index, page in enumerate(series.pages):
                frame = pipe_utils.cast_movie_for_tiff_save(page.asarray(), target_dtype)
                if frame_index < middle_frame:
                    astro_writer.write(frame, contiguous=True)
                else:
                    neuron_writer.write(frame, contiguous=True)
                if frame_index and frame_index % 500 == 0:
                    print(f"Split {frame_index}/{n_frames} frames", flush=True)


def _write_raw_frames(input_path, astro_path, neuron_path, target_dtype):
    n_cols, n_rows = pipe_utils.get_raw_video_dimensions(input_path)
    raw = np.memmap(input_path, dtype=np.uint16, mode="r")
    frame_pixels = int(n_rows) * int(n_cols)
    if raw.size % frame_pixels:
        raise ValueError(f"Raw movie size is not divisible by frame size: {input_path}")
    n_frames = int(raw.size // frame_pixels)
    movie = raw.reshape((n_frames, int(n_rows), int(n_cols)))
    middle_frame = n_frames // 2
    with tifffile.TiffWriter(astro_path, bigtiff=True) as astro_writer, tifffile.TiffWriter(neuron_path, bigtiff=True) as neuron_writer:
        for frame_index in range(n_frames):
            frame = pipe_utils.cast_movie_for_tiff_save(movie[frame_index], target_dtype)
            if frame_index < middle_frame:
                astro_writer.write(frame, contiguous=True)
            else:
                neuron_writer.write(frame, contiguous=True)
            if frame_index and frame_index % 500 == 0:
                print(f"Split {frame_index}/{n_frames} frames", flush=True)


def split_and_save_movies(pipeline_dir, raw_video_path):
    split_dir = os.path.join(pipeline_dir, consts.SPLIT_DIR)
    pipe_utils.mkdir(split_dir)
    target_dtype = pipe_utils.get_signed_movie_dtype(raw_video_path)

    astro_path = os.path.join(split_dir, consts.SPLIT_ASTRO_VIDEO_PATH)
    neuron_path = os.path.join(split_dir, consts.SPLIT_NEURON_VIDEO_PATH)
    ext = os.path.splitext(raw_video_path)[1].lower()
    if ext == ".raw":
        _write_raw_frames(raw_video_path, astro_path, neuron_path, target_dtype)
    else:
        _write_tiff_pages(raw_video_path, astro_path, neuron_path, target_dtype)
    return astro_path, neuron_path


def main(args):
    gui_params_path = args[1]
    raw_video_path, gui_time, fr = extract_params(gui_params_path)
    print("GUI TIME:", gui_time)
    print("Splitting to 2 channels from movie:", raw_video_path)
    pipeline_dir = pipe_utils.get_pipeline_results_dir(raw_video_path)

    astro_path, neuron_path = split_and_save_movies(pipeline_dir, raw_video_path)
    print("Saved astro split:", astro_path, flush=True)
    print("Saved neuron split:", neuron_path, flush=True)

    print(consts.STEP_COMPLETED)
    return


if __name__ == "__main__":
    main(sys.argv)
