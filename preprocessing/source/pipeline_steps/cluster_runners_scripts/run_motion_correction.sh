#!/bin/bash
#SBATCH -J MC
#SBATCH --constraint=avx
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 0-1:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=END

PARAMS_FILE=${1:-"None"}


. ${HOME}/.bashrc
echo "bashrc sourced"
. ${HOME}/miniconda3/bin/activate caiman
echo "caiman env activated"

python -u motion_correction.py "$PARAMS_FILE"
