from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    raw: dict
    project_root: Path
    source_data_root: Path
    sqlite_path: Path
    output_root: Path
    test_output_root: Path
    scan_parent: Path
    bind_host: str
    bind_port: int
    open_browser: bool


def _resolve(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def load_secrets() -> dict:
    """The cluster credentials, from the one git-ignored secrets.yml (or ASTRO_SECRETS_FILE). Empty dict when the file is absent;
    the ASTRO_CLUSTER_* environment variables are still honoured by app.cluster.client."""
    path = Path(os.environ.get("ASTRO_SECRETS_FILE") or PROJECT_ROOT / "secrets.yml")
    if not path.exists():
        return {}
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("cluster", {}) or {}


def load_settings() -> Settings:
    cfg_path = Path(
        os.environ.get("ASTRO_PIPELINE_CONFIG")
        or PROJECT_ROOT / "config.example.yml"
    )
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing application configuration: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    secrets = load_secrets()
    if secrets:
        data["cluster"] = {**(data.get("cluster") or {}), **{k: v for k, v in secrets.items() if v not in (None, "")}}
    root = _resolve(PROJECT_ROOT, data.get("project_root", str(PROJECT_ROOT)))
    return Settings(
        raw=data,
        project_root=root,
        source_data_root=Path(data.get("source_data_root", r"Z:\Adam-Lab-Shared\Data\Miki_Meerson\Imaging\HYP-JEDI")),
        sqlite_path=_resolve(root, data.get("sqlite_path", "runtime/astro_database.sqlite3")),
        output_root=_resolve(root, data.get("output_root", "runtime/outputs")),
        test_output_root=_resolve(root, data.get("test_output_root", "runtime/test_outputs")),
        scan_parent=Path(data.get("scan_parent", data.get("source_data_root", "."))),
        bind_host=str(data.get("bind_host", "127.0.0.1")),
        bind_port=int(data.get("bind_port", 8765)),
        open_browser=bool(data.get("open_browser", True)),
    )
