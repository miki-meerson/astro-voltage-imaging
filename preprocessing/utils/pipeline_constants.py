### pipeline steps names ###

### tab names ###

### field names ###
RAW_VIDEO_PATH_LINUX = "raw_video_path_linux"
IS_2CH = "is_2ch"

### pipeline states ###
FAILED = 3

### pipeline steps states ###

### SLURM states ###

### new SLURM commands ### 4.8.2025

### directories and files names ###
PIPELINE_DIR = "pipeline_results"
RAW_VIDEO_TIF = "raw_video.tif"
SPLIT_DIR = "2ch_split"
SPLIT_NEURON_VIDEO_PATH = "2ch_split_neuron.tif"
SPLIT_ASTRO_VIDEO_PATH = "2ch_split_astro.tif"
MC_DIR = "motion_corrected"
SHIFTS_DIR = "shifts"
MEAN_IMAGE = 'mean_image'
MC_VIDEO_PATH = "motion_corrected.tif"
MC_SHIFTS_PATH = "motion_corrected_shifts.mat"
PB_DIR = "pb_corrected"
PB_VIDEO_PATH = "photobleaching_corrected.tif"
PB_FIT_PATH = "pb_corrected_fit_params.npz"
QC_DIR = "qc"
TRACES_DIR = "traces"
TRACES_PATH = "_traces.csv"

### motion correction ###
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
MEAN_XY_SHIFT = "mean_xy_shift"
MAX_SHIFT = "max_shift"
STD_SHIFT = "std_shift"

### columns names ###
GUI_TIME = "gui_time"
FOV = "FOV"
FRAME_RATE = "frame_rate"

### photobleaching metrics - I(t)=Aâ‹…e^(âˆ’t/Ï„)+C, corrected_mean(t)=mt+b ###
TRIMMED = "trimmed"
BLEACH_TAU = "bleach_tau"
BLEACH_C = "bleach_C"
BLEACH_R2 = "bleach_r2"
RESIDUAL_SLOPE_POST_CORRECTION = "residual_slope_post_correction"  # |m|
CORRECTED_MEAN_INTENSITY = "corrected_mean_intensity"

### global signal metrics (on corrected movie) ###
GLOBAL_DFF_STD = "global_dff_std"
GLOBAL_DFF_MEAN = "global_dff_mean"
GLOBAL_DFF_SKENESS = "global_dff_skewness"

 ### Messages ####
STEP_COMPLETED = "Everything worked well. The Script finished to run."
