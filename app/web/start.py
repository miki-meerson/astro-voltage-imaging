from __future__ import annotations
import threading, time, webbrowser
from waitress import serve
from app.db.core import migrate
from app.services.runs import resume_pending_runs
from app.services.slurm_monitor import start_slurm_monitor
from app.settings import load_settings
from app.web.server import create_app


def main() -> None:
    settings = load_settings()
    try:
        if not settings.source_data_root.exists(): raise FileNotFoundError(f"Source data root is inaccessible: {settings.source_data_root}")
        migrate(settings)
        resumed = resume_pending_runs(settings)
        if resumed:
            print(f"[startup] resumed {resumed} pending run(s)")
    except Exception as exc:
        print(f"[startup failure] {type(exc).__name__}: {exc}"); raise
    url = f"http://{settings.bind_host}:{settings.bind_port}"
    if settings.open_browser: threading.Thread(target=lambda: (time.sleep(1), webbrowser.open(url)), daemon=True).start()
    start_slurm_monitor(settings)
    print(f"Astro Analysis Pipeline listening at {url}"); print(f"SQLite database: {settings.sqlite_path}")
    serve(create_app(), host=settings.bind_host, port=settings.bind_port, threads=8)


if __name__ == "__main__": main()
