from __future__ import annotations

import os
import socket
import threading
import time
from dataclasses import dataclass

import paramiko


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


def _close_ssh(ssh, timeout_seconds: float = 2.0) -> None:
    done = threading.Event()

    def close() -> None:
        try:
            ssh.close()
        finally:
            done.set()

    thread = threading.Thread(target=close, daemon=True, name="astro-ssh-close")
    thread.start()
    done.wait(timeout_seconds)


def run_ssh_command(cluster: dict, command: str, command_timeout_seconds: int | None = None) -> tuple[str, str, int]:
    cred = load_cluster_credentials(cluster)
    timeout = int(cluster.get("paramiko_timeout_seconds") or 600)
    command_timeout = int(command_timeout_seconds or cluster.get("command_timeout_seconds") or timeout)
    connect_timeout = int(cluster.get("connect_timeout_seconds") or command_timeout_seconds or timeout)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(
            cred.host,
            cred.port,
            username=cred.username,
            password=cred.password,
            key_filename=cred.key_filename,
            timeout=connect_timeout,
            banner_timeout=connect_timeout,
            auth_timeout=connect_timeout,
        )
        _, stdout, stderr = ssh.exec_command(command, timeout=command_timeout)
        channel = stdout.channel
        channel.settimeout(1.0)
        out_chunks: list[bytes] = []
        err_chunks: list[bytes] = []
        deadline = time.monotonic() + command_timeout
        while True:
            if channel.recv_ready():
                out_chunks.append(channel.recv(65536))
            if channel.recv_stderr_ready():
                err_chunks.append(channel.recv_stderr(65536))
            if channel.exit_status_ready():
                while channel.recv_ready():
                    out_chunks.append(channel.recv(65536))
                while channel.recv_stderr_ready():
                    err_chunks.append(channel.recv_stderr(65536))
                out_text = b"".join(out_chunks).decode("utf-8", errors="replace")
                err_text = b"".join(err_chunks).decode("utf-8", errors="replace")
                code = int(channel.recv_exit_status())
                return out_text, err_text, code
            if time.monotonic() >= deadline:
                channel.close()
                out_text = b"".join(out_chunks).decode("utf-8", errors="replace")
                err_text = b"".join(err_chunks).decode("utf-8", errors="replace")
                message = f"SSH command timed out after {command_timeout} seconds: {command}"
                return out_text, (err_text + "\n" + message).strip(), 124
            time.sleep(0.1)
    except socket.timeout as exc:
        return "", f"SSH command timed out after {command_timeout} seconds: {command}. {exc}", 124
    finally:
        _close_ssh(ssh)
