from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.db.preprocessing_repository import run_with_experiment
from app.services.preprocessing import initialize_preprocessing_run
from app.settings import load_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--params-json", default="{}")
    parser.add_argument("--run-id", required=True, type=int)
    args = parser.parse_args()

    settings = load_settings()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    run = run_with_experiment(settings, args.run_id)
    if run is None:
        raise ValueError(f"Run {args.run_id} was not found.")
    run["parameters_json"] = args.params_json or run.get("parameters_json") or "{}"
    state = initialize_preprocessing_run(settings, run, out)
    current = state["steps"][state["current_index"]]
    print(f"Submitted preprocessing step {current['label']} as Slurm job {current['job_id']}")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
