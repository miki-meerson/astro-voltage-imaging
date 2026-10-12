#!/bin/bash
#SBATCH -J SPLT
#SBATCH --constraint=avx
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0-1:00:00
#SBATCH --mem=256G
#SBATCH --mail-type=END

PARAMS_FILE=${1:-"None"}

. ${HOME}/.bashrc
echo "bashrc sourced"
. ${HOME}/miniconda3/bin/activate caiman
echo "caiman env activated"

python split_2ch.py "$PARAMS_FILE"
