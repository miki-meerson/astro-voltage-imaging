"""File and folder names shared across the app, the cluster jobs and the tools. Constants only (no imports, no logic), so the cluster jobs can use it.

A name that is used in more than one place is defined here once; everything else imports it. Paths are BUILT from these names in
utils/paths_utils.py (one registry, see PathKind there); nothing else builds paths.
"""

# ---- experiment folder layout -------------------------------------------------------------------------------------------------------------
PIPELINE_RESULTS_DIR = "pipeline_results"
QC_DIR = "qc"
PB_CORRECTED_DIR = "pb_corrected"
MOTION_CORRECTED_DIR = "motion_corrected"
FOR_SUITE2P_DIR = "for_suite2p"
SUITE2P_DIR = "suite2p"
PLANE0_DIR = "plane0"
ASTRO_PB_MOVIE = "astro_photobleaching_corrected.tif"
NEURON_PB_MOVIE = "neuron_photobleaching_corrected.tif"
BAD_SEGMENTS = "bad_segments.json"
RAW_MOVIE = "Image_scan_1_region_0_0.tif"
STA_MOVIES_TIF_DIR = "sta_movies_tif"  # <experiment>/sta_movies_tif/cell_<N>/sta_<group>_bin<B>.tif (Fiji copies of the STA movies)

# ---- an STA run folder: <run>/sta_algorithm_home/sta-analysis/cell_<N>/... -----------------------------------------------------------------
STA_ALGORITHM_HOME_DIR = "sta_algorithm_home"
STA_ANALYSIS_DIR = "sta-analysis"
CELL_DIR_TEMPLATE = "cell_{cell}"
CELL_DIR_GLOB = "cell_*"
CELL_DIR_PREFIX = "cell_"

# result files of one cell (cell folder)
ANATOMY_RESULT = "astro_sta_cluster_permutation_anatomy.npz"
ALL_RESULT = "astro_sta_cluster_permutation.npz"  # result type 'all'
TYPED_RESULT_TEMPLATE = "astro_sta_cluster_permutation_{result_type}.npz"  # every other result type
RESULT_PREFIX = "astro_sta_cluster_permutation"
RESULT_GLOB = RESULT_PREFIX + "*.npz"  # every cluster result of a cell
STA_MOVIE_NPY_PREFIX = "astro_sta_movie"  # binned STA movie the job saves as .npy only when its .tif could not be written: astro_sta_movie[_<group>]_bin<B>.npy
MOTION_STA = "motion_sta.npz"
# per-cell motion status, decided once by the STA job (column "motion_status" of the STA summary csv)
MOTION_STATUS_COLUMN = "motion_status"
MOTION_OK = "ok"
MOTION_SUSPECTED = "motion_suspected"
MOTION_NOT_TESTED = "not_tested"
ANATOMY_GROUP = "anatomy"  # name of the anatomy spike group / result type
ALL_GROUP = "all"
STA_MOVIE_TIF_TEMPLATE = "sta_{group}_bin{bin}.tif"

# files of the STA analysis folder
STA_SUMMARY_CSV = "sta_maps_batch_summary.csv"
ANATOMY_SUMMARY = "anatomy_summary.json"
GROUP_TEST = "anatomy_group_test.json"
GROUP_TEST_RECORDS = "anatomy_group_records.json.gz"
ANATOMY_CROP_PARAMS = "anatomy_crop_params.npz"  # written by the Anatomy Crop step into <anatomy run>/sta-analysis/
BEHAVIOR_SEGMENTS = "behavioral_segements.npz"  # spelling as saved by the Behavior Split step

# ---- spike detection outputs (<spike run>/cell_<N>/...) ------------------------------------------------------------------------------------
SPIKE_LABELS = "simple_complex_spike_labels.npz"
SPIKE_PEAKS = "simple_complex_peaks_for_analysis.npz"
MANUAL_SPIKE_REVIEW = "manual_spike_review.json"

# ---- preprocessing: photobleaching QC files (<experiment>/pipeline_results/qc/), raw movie names, run-folder files -----------------------------
PB_QC_METRICS_TEMPLATE = "photobleaching_qc_{pb_kind}.json"  # pb_kind: astro / neuron
PB_QC_TRACE_TEMPLATE = "photobleaching_qc_trace_{pb_kind}.npz"
PB_QC_PLOT_TEMPLATE = "photobleaching_qc_plot_{pb_kind}.png"
PREPROCESSING_STATE_FILE = "preprocessing_state.json"  # <preprocessing run>/: which step is queued/running/done
PREPROCESSING_PARAMS_DIR = "pipeline_params"  # <preprocessing run>/
PREPROCESSING_PARAMS_TEMPLATE = "run_{run_id}_preprocessing_params.json"
CLUSTER_JOB_ID_FILE = "cluster_job_id.txt"  # <run>/: Slurm job id of the run
PREPROCESSING_SOURCE_DIR = ("preprocessing", "source", "pipeline_steps")  # under the project folder
PREPROCESSING_RUNNER_DIR = "cluster_runners_scripts"  # inside PREPROCESSING_SOURCE_DIR

# ---- experiment registry (experiments service): files the validation looks for in the experiment folder, runtime folders --------------------
EXPERIMENT_XML = "Experiment.xml"  # the microscope's settings file of an experiment (frame rate, ...)
IMAGING_CSV_GLOB = "imaging_*.csv"  # behavior recorded as a CSV next to the movie
TS_DIR_PREFIX = "ts"  # ThorSync folders of the experiment (compared in lower case)
THOR_SETTINGS_XML = "ThorRealTimeDataSettings.xml"  # inside a ThorSync folder
RUNTIME_DIR = "runtime"  # under the project folder: database, outputs, logs, confirmed-cell data
SCIENTIFIC_DATABASE_DIR = "scientific_database"  # <runtime>/scientific_database/<canonical_id>/ : the confirmed-cell data of one experiment

# ---- area 5 (utils / cluster jobs / tools): Suite2p outputs (<experiment>/pipeline_results/for_suite2p/suite2p/plane0/), motion-correction outputs, STA job files
SUITE2P_ISCELL_FILE = "iscell.npy"
SUITE2P_STAT_FILE = "stat.npy"
SUITE2P_F_FILE = "F.npy"
SUITE2P_FNEU_FILE = "Fneu.npy"
SUITE2P_OPS_FILE = "ops.npy"
ASTRO_MEAN_IMAGE_FILE = "astro_mean_image.npy"  # <experiment>/pipeline_results/motion_corrected/
NEURON_MEAN_IMAGE_FILE = "neuron_mean_image.npy"
MOTION_SHIFTS_DIR = "shifts"  # <experiment>/pipeline_results/motion_corrected/shifts/
MOTION_SHIFTS_FILE = "motion_corrected_shifts.mat"
MOTION_SHIFTS_MAT_KEY = "motion_corrected_shifts"  # the variable inside that .mat file
BINNED_ASTRO_MOVIE_TEMPLATE = "astro_photobleaching_corrected_gevi_flipped_bin{bin}.npy"  # <experiment>/pipeline_results/pb_corrected/ (cache of the STA job)
BEHAVIORAL_BINNED_MOVIE_TEMPLATE = "astro_photobleaching_corrected_gevi_flipped_behavioral_{group}_bin{bin}.npy"  # <run>/sta-analysis/
BEHAVIORAL_BINNED_FRAMES_TEMPLATE = "astro_photobleaching_corrected_gevi_flipped_behavioral_{group}_bin{bin}.frames.npy"
STA_USED_PARAMETERS_JSON = "sta_maps_used_parameters.json"  # STA analysis folder: the parameters the job really used
STA_USED_PARAMETERS_NPZ = "sta_maps_used_parameters.npz"
MOTION_SERIES_FILE = "motion_series.npz"  # STA analysis folder
MEAN_ASTRO_BINNED_FILE = "mean_astro_binned.npy"  # STA analysis folder of an Anatomy Crop run
STA_SKIPPED_BEHAVIORAL_GROUPS = "sta_maps_skipped_behavioral_groups.json"
NEURON_STA_TEMPLATE = "neuron_sta{suffix}.npz"  # cell folder of an STA run
PEAK_DETECTION_PARAMS_FILE = "peak_detection_params.npz"  # cell folder of a spike run and of an STA run

# ---- area 4 (cells / spikes / confirmed cells): <runtime>/scientific_database/<canonical_id>/cell_<N>/{spikes,sta}/ ---------------------------------------
CONFIRMED_SPIKES_SUBDIR = "spikes"  # copy of the spike label file of a confirmed cell
CONFIRMED_STA_SUBDIR = "sta"  # copies of the STA result files and the cluster masks of a confirmed cell
CONFIRMED_CLUSTER_MASKS_TEMPLATE = "{name}_cluster_masks.npz"  # name: name of the STA result file without .npz
STA_MOVIE_NPY_TEMPLATE = "{name}_bin{bin}.npy"  # name = STA_MOVIE_NPY_PREFIX or STA_MOVIE_NPY_PREFIX_<group>; cell folder of an STA run

# ---- area 3 (runs / monitoring / adapters): files of an app run folder, cluster-code scripts ----------------------------------------------
RUN_STDOUT_LOG = "stdout.log"  # <run>/: what the adapter process printed
RUN_STDERR_LOG = "stderr.log"
CLUSTER_STDOUT_LOG = "cluster_stdout.log"  # <run>/: output of the Slurm job of the run
CLUSTER_STDERR_LOG = "cluster_stderr.log"
RUN_METRICS_JSON = "metrics.json"  # <run>/: small summary every adapter writes
BEHAVIOR_SPEED_PNG = "behavior_speed_summary.png"  # <behavior run>/
PREPROCESSING_STEP_LOG_TEMPLATE = "preprocessing_{step}.log"  # <preprocessing run>/: local copy of the Slurm log of a preprocessing step
PREPROCESSING_STEP_LOG_GLOB = "preprocessing_*.log"
SPIKE_REVIEW_PNG = "automatic_spike_classification_review.png"  # <spike run>/cell_<N>/
ANATOMY_CROP_PREVIEW_PNG = "anatomy_crop_preview.png"  # <anatomy run>/
STA_JOB_PARAMS_EFFECTIVE = "sta_maps_job_params.effective.json"  # <STA run>/: the parameters file the cluster job reads
GROUP_TEST_STDOUT_LOG = "group_test_stdout.log"  # <STA run>/: log of the anatomy group test job
GROUP_TEST_JOB_ID_FILE = "cluster_group_test_job_id.txt"  # <STA run>/: Slurm job id of the anatomy group test
DATA_SHARE_NAME = "Adam-Lab-Shared"  # the folder of the lab share that the cluster mounts at cluster.path_map.to_cluster
CLUSTER_CODE_DIR = "cluster-code"  # under the project folder: the scripts the Slurm jobs run
STA_JOB_SCRIPT = "run_sta_maps.sh"  # in CLUSTER_CODE_DIR: the sbatch script of the STA Maps job
STA_ALGORITHM_SCRIPT = "sta_maps_sparse.py"  # in CLUSTER_CODE_DIR: the STA Maps algorithm
GROUP_TEST_JOB_SCRIPT = "run_anatomy_group_test.sh"  # in CLUSTER_CODE_DIR: the sbatch script of the anatomy group test
RUN_DIR_TEMPLATE = "{canonical_id}__cell_{cell}__{step}__run_{run_id}"  # <outputs root>/: the folder of one app run (the fields are made file-name safe by the caller)
RUN_DIR_ATTEMPT_TEMPLATE = RUN_DIR_TEMPLATE + "__attempt_{attempt}"  # when the folder above exists already
