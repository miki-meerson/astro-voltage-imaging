import sys
import os
import json
import numpy as np
import tifffile

from utils import globals_utils as g
from utils.file_utils import make_shared_dir
from utils.movie_utils import cast_movie_for_tiff_save, get_signed_movie_dtype, read_experiment_xml
from utils.paths_utils import PathKind, path_for


##### GUI params extraction functions #####

def extract_params(gui_param_path):
    with open(gui_param_path, "r") as fp:
        gui_params = json.load(fp)

    return gui_params[g.PARAM_RAW_VIDEO_PATH_LINUX], gui_params[g.PARAM_GUI_TIME]


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
                frame = cast_movie_for_tiff_save(page.asarray(), target_dtype)
                if frame_index < middle_frame:
                    astro_writer.write(frame, contiguous=True)
                else:
                    neuron_writer.write(frame, contiguous=True)
                if frame_index and frame_index % 500 == 0:
                    print(f"Split {frame_index}/{n_frames} frames", flush=True)


def _write_raw_frames(input_path, astro_path, neuron_path, target_dtype):
    size = read_experiment_xml(os.path.dirname(input_path))
    n_cols, n_rows = size['width'], size['height']
    raw = np.memmap(input_path, dtype=np.uint16, mode="r")
    frame_pixels = int(n_rows) * int(n_cols)
    if raw.size % frame_pixels:
        raise ValueError(f"Raw movie size is not divisible by frame size: {input_path}")
    n_frames = int(raw.size // frame_pixels)
    movie = raw.reshape((n_frames, int(n_rows), int(n_cols)))
    middle_frame = n_frames // 2
    with tifffile.TiffWriter(astro_path, bigtiff=True) as astro_writer, tifffile.TiffWriter(neuron_path, bigtiff=True) as neuron_writer:
        for frame_index in range(n_frames):
            frame = cast_movie_for_tiff_save(movie[frame_index], target_dtype)
            if frame_index < middle_frame:
                astro_writer.write(frame, contiguous=True)
            else:
                neuron_writer.write(frame, contiguous=True)
            if frame_index and frame_index % 500 == 0:
                print(f"Split {frame_index}/{n_frames} frames", flush=True)


def split_and_save_movies(raw_video_path):
    experiment_dir = os.path.dirname(raw_video_path)
    make_shared_dir(path_for(PathKind.SPLIT_DIR, experiment_dir))
    target_dtype = get_signed_movie_dtype(raw_video_path)

    astro_path = path_for(PathKind.SPLIT_ASTRO_MOVIE, experiment_dir)
    neuron_path = path_for(PathKind.SPLIT_NEURON_MOVIE, experiment_dir)
    ext = os.path.splitext(raw_video_path)[1].lower()
    if ext == ".raw":
        _write_raw_frames(raw_video_path, astro_path, neuron_path, target_dtype)
    else:
        _write_tiff_pages(raw_video_path, astro_path, neuron_path, target_dtype)
    return astro_path, neuron_path


def main(args):
    gui_params_path = args[1]
    raw_video_path, gui_time = extract_params(gui_params_path)
    print("GUI TIME:", gui_time)
    print("Splitting to 2 channels from movie:", raw_video_path)

    astro_path, neuron_path = split_and_save_movies(raw_video_path)
    print("Saved astro split:", astro_path, flush=True)
    print("Saved neuron split:", neuron_path, flush=True)

    print(g.STEP_COMPLETED_MESSAGE)
    return


if __name__ == "__main__":
    main(sys.argv)
