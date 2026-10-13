#!/bin/bash
#SBATCH -J STAMAP
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 0-48:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=END

PARAMS_FILE=${1:-"None"}

. ${HOME}/.bashrc
echo "bashrc sourced"
. ${HOME}/miniconda3/bin/activate caiman
echo "caiman env activated"

# the job starts in --chdir (= cluster-code): the repo root holds the utils package
export PYTHONPATH="$(cd "$PWD/.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
python -u sta_maps_sparse.py "$PARAMS_FILE"
