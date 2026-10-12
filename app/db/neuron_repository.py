"""Database access for the cell (neuron) pages: the cell with its experiment, the runs of its experiment, its confirmation.

Every function takes plain arguments and returns plain data (dicts, lists); none returns a connection or a cursor.
"""
from __future__ import annotations

from app.db.core import connect
from app.settings import Settings


def neuron_row(settings: Settings, neuron_id: int) -> dict | None:
    """The cell joined with its experiment: the neurons columns plus canonical_id, source_path and experiment_id; None when the cell is unknown."""
    with connect(settings) as conn:
        return _neuron_row(conn, neuron_id)


def _neuron_row(conn, neuron_id: int) -> dict | None:
    row = conn.execute(
        "SELECT n.*,e.canonical_id,e.source_path,e.id AS experiment_id FROM neurons n JOIN experiments e ON e.id=n.experiment_id WHERE n.id=?",
        (neuron_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def _latest_completed_run(conn, experiment_id: int, step: str) -> dict | None:
    row = conn.execute(
        """
        SELECT * FROM analysis_runs
        WHERE experiment_id=? AND step=? AND is_test=0 AND status='completed'
        ORDER BY id DESC LIMIT 1
        """,
        (experiment_id, step),
    ).fetchone()
    return dict(row) if row is not None else None


def latest_completed_run(settings: Settings, experiment_id: int, step: str) -> dict | None:
    """The newest completed real (not test) run of a step of an experiment, all columns; None when there is none."""
    with connect(settings) as conn:
        return _latest_completed_run(conn, experiment_id, step)


def neuron_with_latest_run(settings: Settings, neuron_id: int, step: str) -> tuple[dict | None, dict | None]:
    """(cell with its experiment, latest completed run of `step` of its experiment); (None, None) when the cell is unknown."""
    with connect(settings) as conn:
        neuron = _neuron_row(conn, neuron_id)
        if neuron is None:
            return None, None
        return neuron, _latest_completed_run(conn, int(neuron["experiment_id"]), step)


def neuron_page_data(settings: Settings, neuron_id: int) -> tuple[dict | None, list[dict], dict | None]:
    """(cell with its experiment, all runs of its experiment ordered by step and newest first, the cell's confirmation row or None)."""
    with connect(settings) as conn:
        neuron = _neuron_row(conn, neuron_id)
        if neuron is None:
            return None, [], None
        runs = [dict(r) for r in conn.execute("SELECT * FROM analysis_runs WHERE experiment_id=? ORDER BY step,id DESC", (neuron["experiment_id"],))]
        confirmed = conn.execute(
            "SELECT * FROM confirmed_neurons WHERE experiment_id=? AND pipeline_neuron_id=?",
            (neuron["experiment_id"], neuron["pipeline_neuron_id"]),
        ).fetchone()
    return neuron, runs, (dict(confirmed) if confirmed is not None else None)
