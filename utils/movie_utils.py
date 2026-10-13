import os
import numpy as np
import tifffile
import xml.etree.ElementTree as ET

from utils import globals_utils as g
from utils.paths_utils import PathKind, path_for


def _get_xml_camera_shape(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    camera = root.find("Camera")
    if camera is None or "width" not in camera.attrib or "height" not in camera.attrib:
        raise ValueError(f"{xml_path} has no Camera element with width and height.")
    width = int(float(camera.attrib["width"].replace(",", "")))
    height = int(float(camera.attrib["height"].replace(",", "")))
    return width, height


def _reorder_tiff_to_thw(movie, axes):
    if movie.ndim != 3:
        return movie

    if not axes:
        return movie

    axes = axes.upper()

    t_axis = next((axes.find(a) for a in ("T", "I", "Q", "Z") if a in axes), -1)
    y_axis = axes.find("Y")
    x_axis = axes.find("X")

    if min(t_axis, y_axis, x_axis) >= 0:
        return np.transpose(movie, (t_axis, y_axis, x_axis))

    return movie


def get_movie_parameters(path, is_dir=False, verbal=False):
    tree = ET.parse(path_for(PathKind.EXPERIMENT_XML, os.path.split(path)[0]))  # is_dir is accepted but has no effect
    root = tree.getroot()
    # Find the LSM tag and get the frameRate attribute
    lsm = root.find("LSM")
    fps = int(float(lsm.attrib['frameRate'].replace(',', '')))
    camera = root.find("Camera")
    width = int(float(camera.attrib['width'].replace(',', '')))
    height = int(float(camera.attrib['height'].replace(',', '')))

    if verbal:
        print(f"FPS: {fps}, Width: {width}, Height: {height}")

    return {'fps': fps, 'width': width, 'height': height}


def read_frame_rate(source_dir) -> float:
    """Frame rate (fps) of an experiment from its Experiment.xml. There is no default: when it cannot be read this raises a ValueError that names the file."""
    xml_path = path_for(PathKind.EXPERIMENT_XML, source_dir)
    try:
        fps = float(get_movie_parameters(xml_path)['fps'])
        if not fps > 0:
            raise ValueError(f"frameRate is {fps}")
        return fps
    except (OSError, ET.ParseError, AttributeError, KeyError, ValueError) as exc:
        raise ValueError(f"The frame rate (fps) could not be read from {xml_path} ({type(exc).__name__}: {exc}). Check that {g.EXPERIMENT_XML} is there and has the LSM frameRate.") from exc


def _read_raw_movie(path, dtype=np.uint16):
    """ Read movie from a .raw path
        First get size from the Experiment.xml file (assumed to be in the same folder)
        Then use it to retrieve the movie
    """
    raw_movie_1d = np.fromfile(path, dtype)
    xml_path = path_for(PathKind.EXPERIMENT_XML, os.path.split(path)[0])
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the LSM tag and get the frameRate attribute
    lsm = root.find("LSM")
    fs = int(float(lsm.attrib['frameRate'].replace(',', '')))
    print('Frame rate:', fs)

    width, height = _get_xml_camera_shape(xml_path)
    movie_3d = np.reshape(raw_movie_1d, (-1, height, width))
    print(f"Movie shape: {movie_3d.shape}")
    return movie_3d


def read_movie(path, dtype=np.uint16, as_memmap=False, is_negative_gevi=False):
    """ Read a movie from path
        Each file type requires different handling
        If the GEVI used is negatively-going, flip the movie
    """

    print("Loading data...")
    movie_type = path.split('.')[-1]
    if movie_type == 'tif':
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            axes = getattr(series, "axes", "")
            if as_memmap:
                movie = series.asarray(out="memmap")
            else:
                movie = series.asarray()
        movie = _reorder_tiff_to_thw(movie, axes)
    elif movie_type == 'raw':
        movie = _read_raw_movie(path, dtype=dtype)
    else:
        raise ValueError('Unknown movie type: {}'.format(movie_type))

    # numpy.linalg (SVD/eig) does not support float16 inputs.
    if movie.dtype == np.float16:
        movie = movie.astype(np.float32, copy=False)

    if is_negative_gevi:
        # Avoid uint wraparound when negating.
        movie = -movie.astype(np.float32, copy=False)
    print(f"Movie shape: {movie.shape}, dtype: {movie.dtype}")

    return movie


def read_astro_movie(path, dtype=np.uint16, as_memmap=False):
    """Read the astro GEVI movie in app-standard polarity.

    Astro movies use a negatively-going GEVI, so this is the single ingestion
    point that flips them once. Neuron movies and Suite2p traces should not use
    this helper.
    """
    return read_movie(path, dtype=dtype, as_memmap=as_memmap, is_negative_gevi=True)


def spatial_bin_movie(movie: np.ndarray, bin_factor: int) -> np.ndarray:
    movie = np.asarray(movie, dtype=np.float32)
    if bin_factor <= 1:
        return movie
    n_frames, n_rows, n_cols = movie.shape
    n_row_crop = (n_rows // bin_factor) * bin_factor
    n_col_crop = (n_cols // bin_factor) * bin_factor
    movie = movie[:, :n_row_crop, :n_col_crop]

    n_row_binned = n_row_crop // bin_factor
    n_col_binned = n_col_crop // bin_factor

    return movie.reshape(
        n_frames,
        n_row_binned, bin_factor,
        n_col_binned, bin_factor,
    ).mean(axis=(2, 4))


def apply_keep_mask_to_movie(movie3d, keep_mask2d):
    movie3d = np.asarray(movie3d, dtype=np.float32)
    if keep_mask2d is None:
        return movie3d
    if movie3d.shape[1:] != keep_mask2d.shape:
        raise ValueError(
            f"Mask shape {keep_mask2d.shape} does not match movie shape {movie3d.shape[1:]}"
        )
    out = movie3d.copy()
    out[:, ~keep_mask2d] = 0.0
    return out


def _bin_image(img2d: np.ndarray, spatial_bin: int) -> np.ndarray:
    if int(spatial_bin) <= 1:
        return np.asarray(img2d, dtype=np.float32)
    b = spatial_bin_movie(np.asarray(img2d, dtype=np.float32)[None, :, :], int(spatial_bin))[0]
    return np.asarray(b, dtype=np.float32)


def get_mean_images(mc_dir, spatial_bin=1):
    astro_mean_img = np.asarray(np.load(path_for(PathKind.MC_ASTRO_MEAN_IMAGE, mc_dir)), dtype=np.float32)
    neuron_mean_img = np.asarray(np.load(path_for(PathKind.MC_NEURON_MEAN_IMAGE, mc_dir)), dtype=np.float32)

    astro_mean_img_b = _bin_image(astro_mean_img, spatial_bin)
    neuron_mean_img_b = _bin_image(neuron_mean_img, spatial_bin)
    
    return astro_mean_img_b, neuron_mean_img_b
