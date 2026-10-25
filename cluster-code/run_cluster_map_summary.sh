#!/bin/bash
#SBATCH -J CLUSTERMAP
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 0-04:00:00
#SBATCH --mem=48G

# Usage: sbatch --chdir=<repo>/cluster-code run_cluster_map_summary.sh <sta-dir> <experiment source folder>
# Makes cluster_map_summary.json of an STA run that is already done (tools/cluster_map_summary.py, run as a module from the repo root).

. ${HOME}/.bashrc
. ${HOME}/miniconda3/bin/activate caiman

REPO="$(cd "$PWD/.." && pwd)"  # the job starts in --chdir (= cluster-code)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
python -u -m tools.cluster_map_summary --sta-dir "$1" --source-dir "$2"
