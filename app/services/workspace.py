"""The experiment workspace shared by every experiment page: the tabs, the step bar and the next action.

The step bar is computed from the workflow stage of the experiment and the status of the newest real run of each step; nothing is stored."""
from __future__ import annotations

from app.db import experiment_repository as repo
from app.services.runs import ACTIVE_STATUSES, STEPS, sta_steps
from utils import globals_utils as g

TABS = (("overview", "Overview"), ("preprocessing", "Preprocessing"), ("spikes", "Spikes"), ("hypotheses", "Hypotheses"), ("cells", "Cells"), ("runs", "Runs"))


def _run_state(status: str | None) -> str:
    """The step-bar state of a step with one run: done, active (running or queued), failed, or todo."""
    if status == "completed":
        return "done"
    if status in ACTIVE_STATUSES:
        return "active"
    return "failed" if status == "failed" else "todo"


def _analysis_step(statuses: dict[str, str]) -> tuple[str, str]:
    """(state, detail) of the analysis step from the newest run status of every hypothesis."""
    latest = [statuses.get(step) for step in sta_steps()]
    done = sum(status == "completed" for status in latest)
    detail = f"{done} of {len(latest)} hypotheses"
    if any(status in ACTIVE_STATUSES for status in latest):
        return "active", detail
    if any(status == "failed" for status in latest):
        return "failed", detail
    return ("done" if done == len(latest) else "todo"), detail


def workflow_steps(stage: str, statuses: dict[str, str]) -> list[dict]:
    """The five steps of the pipeline as the step bar shows them: [{n, label, tab, state, detail}]. `statuses` = {step: status of its newest real run}.
    State: done, active, waiting (it is the user's turn), failed, todo."""
    preprocessing = _run_state(statuses.get(g.PREPROCESSING_STEP))
    detail = ""
    if stage in (repo.STAGE_AWAITING_SUITE2P, repo.STAGE_ANALYSIS_READY):
        preprocessing = "done"
    elif stage == repo.STAGE_PREPROCESSING_REVIEW:
        preprocessing, detail = "waiting", "confirm"
    suite2p, suite2p_detail = ("done", "") if stage == repo.STAGE_ANALYSIS_READY else (("waiting", "confirm") if stage == repo.STAGE_AWAITING_SUITE2P else ("todo", ""))
    analysis, analysis_detail = _analysis_step(statuses)
    steps = [
        ("Preprocessing", "preprocessing", preprocessing, detail),
        ("Suite2p", "preprocessing", suite2p, suite2p_detail),
        ("Spikes", "spikes", _run_state(statuses.get(g.SPIKE_STEP)), ""),
        ("Analysis", "hypotheses", analysis, analysis_detail),
        ("Cell review", "cells", "todo", ""),
    ]
    return [{"n": n, "label": label, "tab": tab, "state": state, "detail": detail} for n, (label, tab, state, detail) in enumerate(steps, start=1)]


NEXT_ACTIONS = {
    ("Preprocessing", "todo"): ("Start preprocessing on the cluster.", "preprocessing"),
    ("Preprocessing", "active"): ("Preprocessing is running on the cluster.", "preprocessing"),
    ("Preprocessing", "failed"): ("Preprocessing failed: open the run to see why, then rerun it.", "runs"),
    ("Preprocessing", "waiting"): ("Preprocessing is finished. Look at the corrected movie and plots, then confirm it.", "preprocessing"),
    ("Suite2p", "waiting"): ("Run Suite2p by hand outside the app, then confirm it. Spike detection starts by itself after that.", "preprocessing"),
    ("Spikes", "active"): ("Spikes are being detected.", "runs"),
    ("Spikes", "failed"): ("Spike detection failed: open the run to see why, then re-detect on the Spikes tab.", "runs"),
    ("Spikes", "todo"): ("Spikes have not been detected yet: re-detect them on the Spikes tab.", "spikes"),
    ("Analysis", "todo"): ("Check the spikes if you want to, then run the analysis below.", "spikes"),
    ("Analysis", "active"): ("The analysis is running.", "runs"),
    ("Analysis", "failed"): ("An analysis run failed: open it to see why, then rerun that hypothesis below.", "runs"),
    ("Cell review", "todo"): ("The results are ready: compare the hypotheses or open single cells.", "hypotheses"),
}


def next_action(steps: list[dict]) -> dict:
    """What to do next, from the first step that is not done: {text, tab}."""
    for step in steps:
        if step["state"] != "done":
            text, tab = NEXT_ACTIONS.get((step["label"], step["state"]), ("Open the step bar to see where you are.", "overview"))
            return {"text": text, "tab": tab}
    return {"text": "Everything is done.", "tab": "cells"}


def tab_links(tab: str) -> list[dict]:
    """The tabs of the workspace with their endpoint names; `tab` is the active one."""
    return [{"key": key, "label": label, "endpoint": f"experiment_{key}", "active": key == tab} for key, label in TABS]


def workspace(exp: dict, latest_runs: dict[str, dict], tab: str) -> dict:
    """What the workspace frame of an experiment page needs: the tabs, the step bar and the next action."""
    steps = workflow_steps(exp["workflow_stage"], {step: row["status"] for step, row in latest_runs.items()})
    return {"tabs": tab_links(tab), "steps": steps, "next": next_action(steps)}


def page_data(settings, experiment_id: int, tab: str) -> dict | None:
    """Everything an experiment page needs from the database plus the workspace frame, or None when the experiment is unknown:
    {exp, neurons, runs, latest_real_runs, notes, active (a run is not finished: the page refreshes itself), tabs, steps, next}."""
    data = repo.experiment_page_data(settings, experiment_id, STEPS)
    if data is None:
        return None
    return {**data, **workspace(data["exp"], data["latest_real_runs"], tab), "active": any(run["status"] in ACTIVE_STATUSES for run in data["runs"])}
