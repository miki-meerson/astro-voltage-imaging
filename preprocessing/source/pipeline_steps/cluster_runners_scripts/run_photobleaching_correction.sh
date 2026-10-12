#!/bin/bash
#SBATCH -J PB
#SBATCH --constraint=avx
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0-1:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=END

PARAMS_FILE=${1:-"None"}


. ${HOME}/.bashrc
echo "bashrc sourced"
. ${HOME}/miniconda3/bin/activate caiman
echo "caiman env activated"

python photobleaching_correction.py "$PARAMS_FILE"
