#!/bin/bash
#SBATCH -J SNAPT
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 0-08:00:00
#SBATCH --mem=48G

# Usage: sbatch --chdir=<repo>/cluster-code run_snapt.sh <sta-dir> <decomposition run folder> <snapt run folder> <experiment source folder>
# SNAPT propagation maps of every cell of a finished decomposition (tools/snapt.py, run as a module from the repo root); the cells are fitted in parallel, one per CPU.

. ${HOME}/.bashrc
. ${HOME}/miniconda3/bin/activate caiman

REPO="$(cd "$PWD/.." && pwd)"  # the job starts in --chdir (= cluster-code)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
python -u -m tools.snapt --sta-dir "$1" --decomposition-dir "$2" --output "$3" --source-dir "$4" --workers "${SLURM_CPUS_PER_TASK:-1}"
