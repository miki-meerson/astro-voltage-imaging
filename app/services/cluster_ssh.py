from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ClusterCredentials:
    host: str
    port: int
    username: str
    password: str | None = None
    key_filename: str | None = None


def _config_or_env(cluster: dict, key: str, env_key: str, default: str | None = None) -> str | None:
    value = cluster.get(key)
    if value not in (None, ""):
        return str(value)
    return os.environ.get(env_key, default)


def load_cluster_credentials(cluster: dict) -> ClusterCredentials:
    host = _config_or_env(cluster, "ssh_host", "ASTRO_CLUSTER_HOST")
    username = _config_or_env(cluster, "ssh_username", "ASTRO_CLUSTER_USER")
    password = _config_or_env(cluster, "ssh_password", "ASTRO_CLUSTER_PASSWORD")
    key_filename = _config_or_env(cluster, "ssh_key_filename", "ASTRO_CLUSTER_KEY")
    port = int(_config_or_env(cluster, "ssh_port", "ASTRO_CLUSTER_PORT", "22") or 22)

    missing = []
    if not host:
        missing.append("cluster.ssh_host or ASTRO_CLUSTER_HOST")
    if not username:
        missing.append("cluster.ssh_username or ASTRO_CLUSTER_USER")
    if not password and not key_filename:
        missing.append("cluster.ssh_password/ASTRO_CLUSTER_PASSWORD or cluster.ssh_key_filename/ASTRO_CLUSTER_KEY")
    if missing:
        raise FileNotFoundError("Cluster SSH settings are incomplete: " + ", ".join(missing))

    return ClusterCredentials(
        host=host,
        port=port,
        username=username,
        password=password,
        key_filename=key_filename,
    )


def run_ssh_command(cluster: dict, command: str) -> tuple[str, str, int]:
    import paramiko

    cred = load_cluster_credentials(cluster)
    timeout = int(cluster.get("paramiko_timeout_seconds") or 600)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(
            cred.host,
            cred.port,
            username=cred.username,
            password=cred.password,
            key_filename=cred.key_filename,
            timeout=timeout,
            banner_timeout=timeout,
            auth_timeout=timeout,
        )
        _, stdout, stderr = ssh.exec_command(command)
        out_text = stdout.read().decode("utf-8", errors="replace")
        err_text = stderr.read().decode("utf-8", errors="replace")
        code = int(stdout.channel.recv_exit_status())
        return out_text, err_text, code
    finally:
        ssh.close()
