from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import threading
import time

from app.db.core import connect, tx
from app.services.cluster_ssh import run_ssh_command
from app.settings import Settings


FAILED_STATES = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")


def _ssh_command(settings: Settings, command: str) -> tuple[str, str, int]:
    return run_ssh_command(settings.raw.get("cluster", {}), command)


def _job_id(output_dir: str | None) -> str | None:
    if not output_dir:
        return None
    path = Path(output_dir) / "cluster_job_id.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text or None


def _manifest(output_dir: str | None) -> list[dict]:
    if not output_dir:
        return []
    root = Path(output_dir)
    if not root.exists():
        return []
    return [
        {
            "path": str(p.relative_to(root.parent)),
            "type": p.suffix.lower().lstrip(".") or "file",
            "size": p.stat().st_size,
        }
        for p in root.rglob("*")
        if p.is_file()
    ]


def _tail(path: Path, limit: int = 2000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-limit:].strip()


def poll_slurm_once(settings: Settings) -> int:
    with connect(settings) as conn:
        runs = [dict(row) for row in conn.execute(
            """
            SELECT id,status,output_dir,stderr_path
            FROM analysis_runs
            WHERE step='sta' AND backend='cluster' AND status IN ('queued','running')
            ORDER BY id
            """
        )]
    changed = 0
    cluster = settings.raw.get("cluster", {})
    squeue = str(cluster.get("squeue_command") or "/usr/bin/squeue")
    sacct = str(cluster.get("sacct_command") or "/usr/bin/sacct")
    for run in runs:
        job_id = _job_id(run.get("output_dir"))
        if not job_id:
            continue
        sq_out, _, sq_code = _ssh_command(settings, f"{squeue} -h -j {job_id} -o '%T'")
        if sq_code == 0 and sq_out.strip():
            state = sq_out.strip().splitlines()[0].strip()
            status = "running" if state == "RUNNING" else "queued"
            if status != run["status"]:
                with tx(settings) as conn:
                    conn.execute("UPDATE analysis_runs SET status=? WHERE id=?", (status, run["id"]))
                changed += 1
            continue
        acct_out, acct_err, acct_code = _ssh_command(settings, f"{sacct} -n -j {job_id} --format=State,ExitCode -P")
        if acct_code != 0:
            print(f"[slurm monitor] sacct failed for job {job_id}: {(acct_err or acct_out).strip()}")
            continue
        states = [line.split("|", 1)[0].strip() for line in acct_out.splitlines() if line.strip()]
        if not states:
            continue
        manifest = _manifest(run.get("output_dir"))
        status = None
        exit_code = None
        note = ""
        if any(state.startswith("COMPLETED") for state in states):
            status = "completed"
            exit_code = 0
        elif any(state.startswith(FAILED_STATES) for state in states):
            status = "failed"
            exit_code = 1
            stderr_path = Path(run["output_dir"]) / "cluster_stderr.log" if run.get("output_dir") else Path(run.get("stderr_path") or "")
            note = _tail(stderr_path)
        if status:
            with tx(settings) as conn:
                conn.execute("DELETE FROM artifacts WHERE run_id=?", (run["id"],))
                conn.execute(
                    """
                    UPDATE analysis_runs
                    SET status=?,ended_at=?,exit_code=?,output_manifest_json=?,review_note=?
                    WHERE id=?
                    """,
                    (
                        status,
                        datetime.now().isoformat(timespec="seconds"),
                        exit_code,
                        json.dumps(manifest),
                        note[-1000:],
                        run["id"],
                    ),
                )
                for item in manifest:
                    conn.execute(
                        "INSERT INTO artifacts(run_id,artifact_type,relative_path,size_bytes) VALUES(?,?,?,?)",
                        (run["id"], item["type"], item["path"], item["size"]),
                    )
            changed += 1
    return changed


def start_slurm_monitor(settings: Settings) -> threading.Thread | None:
    if not settings.raw.get("cluster", {}).get("enabled", False):
        return None
    interval = max(15, int(settings.raw.get("cluster", {}).get("poll_interval_seconds", 30)))

    def loop() -> None:
        while True:
            try:
                poll_slurm_once(settings)
            except Exception as exc:
                print(f"[slurm monitor warning] {type(exc).__name__}: {exc}")
            time.sleep(interval)

    thread = threading.Thread(target=loop, daemon=True, name="astro-slurm-monitor")
    thread.start()
    return thread
