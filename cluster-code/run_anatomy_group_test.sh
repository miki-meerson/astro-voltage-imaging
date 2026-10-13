#!/bin/bash
#SBATCH -J ANATGROUP
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 0-12:00:00
#SBATCH --mem=96G

# Usage: sbatch --chdir=<repo>/cluster-code run_anatomy_group_test.sh <sta-dir> <experiment source folder>
# Re-runs the anatomy cluster search on every cell's fakes and compares real vs fake groups (tools/anatomy_cluster_null.py, run as a module from the repo root).

. ${HOME}/.bashrc
. ${HOME}/miniconda3/bin/activate caiman

REPO="$(cd "$PWD/.." && pwd)"  # the job starts in --chdir (= cluster-code)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
python -u -m tools.anatomy_cluster_null --sta-dir "$1" --source-dir "$2" --workers ${SLURM_CPUS_PER_TASK:-8}
