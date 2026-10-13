"""The shell text that prepares the STA mirror folder on the cluster (links to the experiment's files and to the input runs)."""
from __future__ import annotations

from pathlib import Path

from app.cluster.client import to_cluster_path
from utils import globals_utils as g
from utils.paths_utils import PathKind, path_for


def sh_single(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def preflight_script(source_home: Path, mirror_home: Path, spike_run: Path, behavior_run: Path | None, anatomy_run: Path | None, cluster: dict) -> str:
    source = sh_single(to_cluster_path(source_home, cluster))
    mirror = sh_single(to_cluster_path(mirror_home, cluster))
    spike = sh_single(to_cluster_path(spike_run, cluster))
    behavior = sh_single(to_cluster_path(behavior_run, cluster)) if behavior_run else "''"
    anatomy = sh_single(to_cluster_path(anatomy_run, cluster)) if anatomy_run else "''"

    def at(kind: PathKind, root: str, **fields) -> str:
        return path_for(kind, root, **fields).as_posix()  # a shell path under $SOURCE_HOME, $MIRROR_HOME or $ANATOMY_RUN

    mirror_sta = at(PathKind.RUN_STA_ANALYSIS, "$MIRROR_HOME")
    return f"""
set -euo pipefail
SOURCE_HOME={source}
MIRROR_HOME={mirror}
SPIKE_RUN={spike}
BEHAVIOR_RUN={behavior}
ANATOMY_RUN={anatomy}

echo "Preparing cluster STA mirror at $MIRROR_HOME"
mkdir -p "{at(PathKind.PB_CORRECTED_DIR, "$MIRROR_HOME")}"
mkdir -p "{at(PathKind.SUITE2P_PLANE0, "$MIRROR_HOME")}"
mkdir -p "{at(PathKind.QC_DIR, "$MIRROR_HOME")}"
mkdir -p "{mirror_sta}"

ln -sfn "{at(PathKind.EXPERIMENT_XML, "$SOURCE_HOME")}" "{at(PathKind.EXPERIMENT_XML, "$MIRROR_HOME")}"
ln -sfn "{at(PathKind.RAW_MOVIE, "$SOURCE_HOME")}" "{at(PathKind.RAW_MOVIE, "$MIRROR_HOME")}"
for behavior_csv in "$SOURCE_HOME"/{g.IMAGING_CSV_GLOB}; do
  if [ -e "$behavior_csv" ]; then ln -sfn "$behavior_csv" "$MIRROR_HOME/$(basename "$behavior_csv")"; fi
done
ln -sfn "{at(PathKind.ASTRO_PB_MOVIE, "$SOURCE_HOME")}" "{at(PathKind.ASTRO_PB_MOVIE, "$MIRROR_HOME")}"
if [ -e "{at(PathKind.NEURON_PB_MOVIE, "$SOURCE_HOME")}" ]; then
  ln -sfn "{at(PathKind.NEURON_PB_MOVIE, "$SOURCE_HOME")}" "{at(PathKind.NEURON_PB_MOVIE, "$MIRROR_HOME")}"
fi
if [ -e "{at(PathKind.BAD_SEGMENTS, "$SOURCE_HOME")}" ]; then
  ln -sfn "{at(PathKind.BAD_SEGMENTS, "$SOURCE_HOME")}" "{at(PathKind.BAD_SEGMENTS, "$MIRROR_HOME")}"
fi
rm -rf "{at(PathKind.SUITE2P_PLANE0, "$MIRROR_HOME")}"
ln -sfn "{at(PathKind.SUITE2P_PLANE0, "$SOURCE_HOME")}" "{at(PathKind.SUITE2P_PLANE0, "$MIRROR_HOME")}"

rm -rf "{at(PathKind.SPIKE_RUN_LINK, "$MIRROR_HOME")}"
ln -sfn "$SPIKE_RUN" "{at(PathKind.SPIKE_RUN_LINK, "$MIRROR_HOME")}"
if [ -n "$BEHAVIOR_RUN" ] && [ -e "{at(PathKind.BEHAVIOR_SEGMENTS, "$BEHAVIOR_RUN")}" ]; then
  ln -sfn "{at(PathKind.BEHAVIOR_SEGMENTS, "$BEHAVIOR_RUN")}" "{at(PathKind.STA_BEHAVIOR_SEGMENTS, mirror_sta)}"
fi
if [ -n "$ANATOMY_RUN" ] && [ -e "{at(PathKind.ANATOMY_CROP_PARAMS, "$ANATOMY_RUN")}" ]; then
  ln -sfn "{at(PathKind.ANATOMY_CROP_PARAMS, "$ANATOMY_RUN")}" "{at(PathKind.ANATOMY_CROP_PARAMS, "$MIRROR_HOME")}"
fi
"""
