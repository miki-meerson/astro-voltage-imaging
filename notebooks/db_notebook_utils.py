from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

import numpy as np
import pandas as pd


def project_root() -> Path:
    here = Path.cwd().resolve()
    for path in (here, *here.parents):
        if (path / "app").exists() and (path / "runtime").exists():
            return path
    return Path(__file__).resolve().parents[1]


ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.settings import load_settings
from utils.movie_utils import get_mean_images, get_movie_parameters
from utils.paths_utils import resolve_sta_paths
from utils.suite2p_utils import load_suite2p_context


def _connect() -> sqlite3.Connection:
    settings = load_settings()
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn


def _json_array(value: str | None, dtype=None) -> np.ndarray:
    if not value:
        return np.asarray([], dtype=dtype or float)
    data = json.loads(value)
    return np.asarray(data, dtype=dtype) if dtype is not None else np.asarray(data)


def _read_sql(sql: str, params: tuple = ()) -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def available_experiments() -> pd.DataFrame:
    return _read_sql(
        """
        SELECT e.id AS experiment_id,e.canonical_id,e.source_path,e.recording_date,
               COUNT(cn.id) AS n_confirmed_cells
        FROM experiments e
        LEFT JOIN confirmed_neurons cn ON cn.experiment_id=e.id
        GROUP BY e.id
        ORDER BY e.recording_date,e.canonical_id
        """
    )


def _experiment_filter(experiment: int | str | None, alias: str = "cn") -> tuple[str, tuple]:
    if experiment is None:
        return "", ()
    if isinstance(experiment, int):
        return f"{alias}.experiment_id=?", (experiment,)
    return f"{alias}.canonical_id=?", (str(experiment),)


def _confirmed_scope_sql(experiment: int | str | None, latest_only: bool = True) -> tuple[str, tuple]:
    clauses = []
    params: tuple = ()
    exp_clause, exp_params = _experiment_filter(experiment, "cn")
    if exp_clause:
        clauses.append(exp_clause)
        params += exp_params
    latest_join = ""
    if latest_only:
        latest_join = """
        JOIN (
            SELECT experiment_id,MAX(id) AS latest_spike_run_id
            FROM analysis_runs
            WHERE step='spike' AND is_test=0 AND status='completed'
            GROUP BY experiment_id
        ) latest_spike ON latest_spike.experiment_id=cn.experiment_id
        JOIN (
            SELECT experiment_id,MAX(id) AS latest_sta_run_id
            FROM analysis_runs
            WHERE step='sta' AND is_test=0 AND status='completed'
            GROUP BY experiment_id
        ) latest_sta ON latest_sta.experiment_id=cn.experiment_id
        """
        clauses.append("cn.spike_run_id=latest_spike.latest_spike_run_id")
        clauses.append("cn.sta_run_id=latest_sta.latest_sta_run_id")
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    return latest_join + "\n" + where, params


def load_confirmed_cells(experiment: int | str | None = None, latest_only: bool = True) -> pd.DataFrame:
    scope_sql, params = _confirmed_scope_sql(experiment, latest_only)
    return _read_sql(
        f"""
        SELECT cn.*
        FROM confirmed_neurons cn
        {scope_sql}
        ORDER BY cn.experiment_id, CAST(cn.pipeline_neuron_id AS INTEGER)
        """,
        params,
    )


def load_confirmed_peaks(experiment: int | str | None = None, latest_only: bool = True) -> pd.DataFrame:
    scope_sql, params = _confirmed_scope_sql(experiment, latest_only)
    df = _read_sql(
        f"""
        SELECT cn.experiment_id,cn.canonical_id,cn.pipeline_neuron_id,cn.source_path,
               cnp.*
        FROM confirmed_neuron_peaks cnp
        JOIN confirmed_neurons cn ON cn.id=cnp.confirmed_neuron_id
        {scope_sql}
        ORDER BY cn.experiment_id, CAST(cn.pipeline_neuron_id AS INTEGER)
        """,
        params,
    )
    if df.empty:
        return df
    df["cell_id"] = df["pipeline_neuron_id"].astype(int)
    df["all_peaks"] = df["all_peak_frames_json"].map(lambda x: _json_array(x, np.int32))
    df["simple_peaks"] = df["simple_peak_frames_json"].map(lambda x: _json_array(x, np.int32))
    df["complex_peaks"] = df["complex_peak_frames_json"].map(lambda x: _json_array(x, np.int32))
    return df


def load_confirmed_sta_results(experiment: int | str | None = None, latest_only: bool = True) -> pd.DataFrame:
    scope_sql, params = _confirmed_scope_sql(experiment, latest_only)
    df = _read_sql(
        f"""
        SELECT cn.experiment_id,cn.canonical_id,cn.pipeline_neuron_id,cn.source_path,
               csr.*
        FROM confirmed_sta_results csr
        JOIN confirmed_neurons cn ON cn.id=csr.confirmed_neuron_id
        {scope_sql}
        ORDER BY cn.experiment_id, CAST(cn.pipeline_neuron_id AS INTEGER), csr.result_type
        """,
        params,
    )
    if df.empty:
        return df
    df["cell_id"] = df["pipeline_neuron_id"].astype(int)
    return df


def load_confirmed_clusters(experiment: int | str | None = None, latest_only: bool = True) -> pd.DataFrame:
    scope_sql, params = _confirmed_scope_sql(experiment, latest_only)
    df = _read_sql(
        f"""
        SELECT cn.experiment_id,cn.canonical_id,cn.pipeline_neuron_id,cn.source_path,
               csr.result_type,csr.version,
               csc.*
        FROM confirmed_sta_clusters csc
        JOIN confirmed_sta_results csr ON csr.id=csc.confirmed_sta_result_id
        JOIN confirmed_neurons cn ON cn.id=csr.confirmed_neuron_id
        {scope_sql}
        ORDER BY cn.experiment_id, CAST(cn.pipeline_neuron_id AS INTEGER), csr.result_type, csc.cluster_id
        """,
        params,
    )
    if df.empty:
        return df
    df["cell_id"] = df["pipeline_neuron_id"].astype(int)
    df["mean_trace"] = df["mean_trace_json"].map(lambda x: _json_array(x, np.float32))
    df["t_rel"] = df["t_rel_json"].map(lambda x: _json_array(x, np.int32))
    df["is_significant"] = df["is_significant"].astype(bool)
    return df


@dataclass
class ConfirmedData:
    experiment: int | str | None
    latest_only: bool
    experiments: pd.DataFrame
    cells: pd.DataFrame
    peaks: pd.DataFrame
    sta_results: pd.DataFrame
    clusters: pd.DataFrame


def load_confirmed_data(experiment: int | str | None = None, latest_only: bool = True) -> ConfirmedData:
    return ConfirmedData(
        experiment=experiment,
        latest_only=latest_only,
        experiments=available_experiments(),
        cells=load_confirmed_cells(experiment, latest_only=latest_only),
        peaks=load_confirmed_peaks(experiment, latest_only=latest_only),
        sta_results=load_confirmed_sta_results(experiment, latest_only=latest_only),
        clusters=load_confirmed_clusters(experiment, latest_only=latest_only),
    )


def require_single_experiment(data: ConfirmedData) -> pd.Series:
    cells = data.cells.loc[:, ~data.cells.columns.duplicated()].copy()
    unique = cells[["experiment_id", "canonical_id", "source_path"]].drop_duplicates()
    if len(unique) != 1:
        raise ValueError(
            "This section needs exactly one experiment. Set EXPERIMENT to one canonical_id "
            "from available_experiments(), then rerun the setup cell."
        )
    return unique.iloc[0]


def _scalar_value(value):
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    return value


@dataclass
class ExperimentContext:
    experiment_id: int
    canonical_id: str
    source_path: Path
    fps: float
    paths: Any
    suite2p_ctx: dict
    astro_mean_img: np.ndarray
    neuron_mean_img: np.ndarray
    image_shape: tuple[int, int]


def load_single_experiment_context(data: ConfirmedData) -> ExperimentContext:
    row = require_single_experiment(data)
    source_path = Path(_scalar_value(row["source_path"]))
    fps = float(get_movie_parameters(f"{source_path.as_posix()}/", is_dir=True)["fps"])
    paths = resolve_sta_paths(source_path)
    suite2p_ctx = load_suite2p_context(paths.suite2p_path)
    astro_mean_img, neuron_mean_img = get_mean_images(paths.mc_dir)
    return ExperimentContext(
        experiment_id=int(_scalar_value(row["experiment_id"])),
        canonical_id=str(_scalar_value(row["canonical_id"])),
        source_path=source_path,
        fps=fps,
        paths=paths,
        suite2p_ctx=suite2p_ctx,
        astro_mean_img=astro_mean_img,
        neuron_mean_img=neuron_mean_img,
        image_shape=astro_mean_img.shape,
    )


def significant_cluster_counts(clusters: pd.DataFrame, result_type: str | None = None) -> pd.Series:
    df = clusters[clusters["is_significant"]].copy()
    if result_type is not None:
        df = df[df["result_type"].eq(result_type)]
    if df.empty:
        return pd.Series(dtype=int)
    return df.groupby(["experiment_id", "cell_id"])["cluster_id"].nunique().astype(int)


def load_cluster_mask(row: pd.Series) -> np.ndarray | None:
    path = Path(row["mask_artifact_path"])
    key = f"cluster_{int(row['cluster_id'])}_mask2d"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as d:
        if key not in d:
            return None
        return np.asarray(d[key], dtype=bool)
