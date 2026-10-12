"""Cluster credentials for the preprocessing code.

Values are read from the shared, git-ignored secrets.yml at the repo root (see secrets.example.yml), or from the
ASTRO_CLUSTER_* environment variables. No credentials are stored in this file.
"""
import os
from pathlib import Path

import yaml


def _load() -> dict:
    path = Path(os.environ.get("ASTRO_SECRETS_FILE") or Path(__file__).resolve().parents[2] / "secrets.yml")
    data = {}
    if path.exists():
        data = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("cluster", {}) or {}
    env = {"ssh_host": "ASTRO_CLUSTER_HOST", "ssh_port": "ASTRO_CLUSTER_PORT", "ssh_username": "ASTRO_CLUSTER_USER", "ssh_password": "ASTRO_CLUSTER_PASSWORD"}
    for key, name in env.items():
        if os.environ.get(name):
            data[key] = os.environ[name]
    if not data.get("ssh_username") or not data.get("ssh_host"):
        raise RuntimeError(f"Cluster credentials not found: create {path} from secrets.example.yml")
    return data


_c = _load()
SSH_USERNAME = _c["ssh_username"]
SSH_PASSWORD = _c.get("ssh_password", "")
CLUSTER_HOST = _c["ssh_host"]
CLUSTER_PORT = int(_c.get("ssh_port") or 22)
