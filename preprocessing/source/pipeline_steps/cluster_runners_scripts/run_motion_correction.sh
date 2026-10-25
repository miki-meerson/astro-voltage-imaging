#!/bin/bash
#SBATCH -J MC
#SBATCH --constraint=avx
#SBATCH -N 1
#SBATCH -c 16
# a rigid correction takes about 15 minutes, a non-rigid one about an hour: the limit leaves room for the slower one
#SBATCH -t 0-4:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=END

PARAMS_FILE=${1:-"None"}


. ${HOME}/.bashrc
echo "bashrc sourced"
. ${HOME}/miniconda3/bin/activate caiman
echo "caiman env activated"

# the job starts in --chdir (= preprocessing/source/pipeline_steps): the repo root, three folders up, holds the utils package
export PYTHONPATH="$(cd "$PWD/../../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
python -u motion_correction.py "$PARAMS_FILE"
