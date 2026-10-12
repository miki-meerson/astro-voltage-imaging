from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3

from app.settings import Settings


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def connect(settings: Settings) -> sqlite3.Connection:
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False, factory=ClosingConnection)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")  # wait up to 5 s for a lock instead of failing with "database is locked"
    # WAL is a property of the database file: it is set once in migrate() (schema.sql), not on every connection
    return conn


def migrate(settings: Settings) -> None:
    with connect(settings) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(Path(__file__).with_name("schema.sql").read_text(encoding="utf-8"))
        max_run_id = conn.execute("SELECT COALESCE(MAX(id), 0) AS max_id FROM analysis_runs").fetchone()["max_id"]
        conn.execute(
            "UPDATE run_id_sequence SET next_id=MAX(next_id, ?) WHERE id=1",
            (int(max_run_id) + 1,),
        )


@contextmanager
def tx(settings: Settings):
    conn = connect(settings)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def rows(conn: sqlite3.Connection, sql: str, params=()):
    return [dict(r) for r in conn.execute(sql, params).fetchall()]
