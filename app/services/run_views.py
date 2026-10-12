"""Run rows as shown in the pages, run logs, and small DB look-ups shared by several pages."""
from __future__ import annotations

import json
from pathlib import Path

from app.db.core import rows
from app.services.runs import run_dir

STEP_LABELS = {"preprocessing": "Preprocessing", "spike": "Spike Detection", "behavior": "Behavior Split", "anatomy": "Anatomy Crop", "sta": "STA Maps"}
STATUS_PROGRESS = {
    "pending": 8,
    "queued": 18,
    "running": 55,
    "completed": 100,
    "failed": 100,
    "cancelled": 100,
    "blocked": 100,
    "invalid inputs": 100,
}


def run_view(row, settings):
    run = dict(row)
    run["label"] = STEP_LABELS.get(run["step"], run["step"].title())
    run["progress"] = STATUS_PROGRESS.get(run["status"], 0)
    run["output_dir"] = str(run_dir(settings, int(run["id"]), bool(run["is_test"])))
    run["output_name"] = Path(run["output_dir"]).name
    run["failure_reason"] = failure_reason(run)
    try:
        run["artifacts"] = json.loads(run["output_manifest_json"] or "[]")
    except json.JSONDecodeError:
        run["artifacts"] = []
    run["artifact_count"] = len(run["artifacts"])
    return run


def run_view_for_neuron(row, settings, pipeline_neuron_id: str):
    run = run_view(row, settings)
    output_name = run["output_name"]
    cell_prefixes = (
        f"{output_name}/cell_{pipeline_neuron_id}/",
        f"{output_name}/sta_algorithm_home/sta-analysis/cell_{pipeline_neuron_id}/",
    )
    if run["step"] == "behavior":
        filtered = [
            a for a in run["artifacts"]
            if Path(a.get("path", "")).name in {"behavioral_segements.npz", "behavior_speed_summary.png", "metrics.json"}
        ]
    else:
        filtered = [a for a in run["artifacts"] if str(a.get("path", "")).replace("\\", "/").startswith(cell_prefixes)]
    run["artifacts"] = filtered
    run["artifact_count"] = len(filtered)
    run["preview_artifacts"] = [
        a for a in filtered
        if str(a.get("path", "")).lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    return run


def build_validation_rows(report: dict):
    checks = report.get("checks", {}) if isinstance(report, dict) else {}
    result = []
    for name, info in checks.items():
        if isinstance(info, dict):
            if "exists" in info:
                result.append({"name": name.replace("_", " ").title(), "status": "OK" if info["exists"] else "Missing", "detail": Path(info.get("path", "")).name})
            elif "count" in info:
                result.append({"name": name.replace("_", " ").title(), "status": "OK" if info.get("count") == 1 else "Check", "detail": f"{info.get('count', 0)} file(s)"})
    return result


def validation_all_ok(validation_rows: list[dict]) -> bool:
    return bool(validation_rows) and all(row.get("status") == "OK" for row in validation_rows)


def read_tail(path: str | None, limit: int = 12000) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    data = p.read_text(encoding="utf-8", errors="replace")
    return data[-limit:]


def run_stdout(run: dict) -> str:
    chunks = [read_tail(run.get("stdout_path"))]
    out_dir = Path(run["output_dir"]) if run.get("output_dir") else (Path(run["stdout_path"]).parent if run.get("stdout_path") else None)
    if out_dir is not None:
        cluster = read_tail(str(out_dir / "cluster_stdout.log"))
        if cluster:
            chunks.append("\n\n--- Cluster stdout ---\n" + cluster)
        for log_path in sorted(out_dir.glob("preprocessing_*.log")):
            log_text = read_tail(str(log_path))
            if log_text:
                chunks.append(f"\n\n--- {log_path.name} ---\n" + log_text)
    return "".join(chunks)


def run_stderr(run: dict) -> str:
    chunks = [read_tail(run.get("stderr_path"))]
    out_dir = Path(run["output_dir"]) if run.get("output_dir") else (Path(run["stderr_path"]).parent if run.get("stderr_path") else None)
    if out_dir is not None:
        cluster = read_tail(str(out_dir / "cluster_stderr.log"))
        if cluster:
            chunks.append("\n\n--- Cluster stderr ---\n" + cluster)
    return "".join(chunks)


def failure_reason(run: dict, limit: int = 360) -> str:
    if run.get("status") != "failed":
        return ""
    stderr = read_tail(run.get("stderr_path"), limit=2000).strip()
    if stderr:
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        return " | ".join(lines[-4:])[-limit:]
    try:
        command = json.loads(run.get("command_json") or "{}")
    except json.JSONDecodeError:
        command = run.get("command_json") or ""
    if isinstance(command, dict) and command.get("error"):
        return str(command["error"])[:limit]
    return str(command)[:limit] if command else "No error details were captured."


def latest_completed_run(conn, experiment_id: int, step: str):
    return conn.execute(
        """
        SELECT * FROM analysis_runs
        WHERE experiment_id=? AND step=? AND is_test=0 AND status='completed'
        ORDER BY id DESC LIMIT 1
        """,
        (experiment_id, step),
    ).fetchone()


def experiment_notes(conn, experiment_id: int) -> list[dict]:
    return rows(
        conn,
        """
        SELECT id,note_text,created_at,updated_at
        FROM experiment_notes
        WHERE experiment_id=?
        ORDER BY created_at DESC, id DESC
        """,
        (experiment_id,),
    )
