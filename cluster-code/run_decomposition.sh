#!/bin/bash
#SBATCH -J STADECOMP
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0-04:00:00
#SBATCH --mem=48G

# Usage: sbatch --chdir=<repo>/cluster-code run_decomposition.sh <sta-dir> <decomposition run folder> <experiment source folder>
# PCA and ICA of the whole-field STA movie of every cell of an STA run that is already done (tools/sta_decomposition.py, run as a module from the repo root).

. ${HOME}/.bashrc
. ${HOME}/miniconda3/bin/activate caiman

REPO="$(cd "$PWD/.." && pwd)"  # the job starts in --chdir (= cluster-code)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
python -u -m tools.sta_decomposition --sta-dir "$1" --output "$2" --source-dir "$3"
