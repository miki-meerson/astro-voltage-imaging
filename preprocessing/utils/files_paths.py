import os
pipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

### BASH runners ###
# Build cluster path from the repo location instead of a brittle hardcoded path.
# This supports the current `.../Miki/astro-pipeline/...` layout and future moves.
_path_parts = pipeline_dir.replace("\\", "/").split("/")
if "Adam-Lab-Shared" in _path_parts:
    _suffix = "/".join(_path_parts[_path_parts.index("Adam-Lab-Shared") + 1:])
    CLUSTER_RUNNERS_DIR = f"/ems/elsc-labs/adam-y/Adam-Lab-Shared/{_suffix}/source/pipeline_steps/cluster_runners_scripts/"
else:
    # Fallback to current behavior if the expected mount marker is unavailable.
    CLUSTER_RUNNERS_DIR = '/ems/elsc-labs/adam-y/Adam-Lab-Shared/FromExperimentToAnalysis/Miki/astro-pipeline/source/pipeline_steps/cluster_runners_scripts/'
MOTION_CORRECTION_BASH = 'run_motion_correction.sh'
PHOTOBLEACHING_BASH = 'run_photobleaching_correction.sh'
SPLIT_2CH_BASH = 'run_split_2ch.sh'

### pipeline dirs and scripts ###
PIPELINE_LOGS_DIR = os.path.join(pipeline_dir, "metadata", "logs")
PIPELINE_RUNNER_SCRIPT = 'step_manager.py'
