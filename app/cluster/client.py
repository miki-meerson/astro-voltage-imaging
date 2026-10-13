"""The only place that talks to the Slurm cluster: paths, ssh, sbatch, squeue/sacct. Everything takes the `cluster` settings dict
(the `cluster:` section of the app config; the ssh_* credentials may also come from the ASTRO_CLUSTER_* environment variables)."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import socket
import sys
import threading
import time

import paramiko

from utils import globals_utils as g

DEFAULT_SBATCH = "/usr/bin/sbatch"
DEFAULT_SQUEUE = "/usr/bin/squeue"
DEFAULT_SACCT = "/usr/bin/sacct"
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_ATTEMPTS = 5
# connection problems: the only errors that are worth another attempt; anything else is a bug and propagates
CONNECTION_ERRORS = (OSError, paramiko.SSHException, EOFError)
# what a cluster call can raise: no/incomplete credentials or a broken connection (OSError), an SSH protocol error,
# an unreadable answer (ValueError, KeyError), a refused submission (RuntimeError)
CLUSTER_ERRORS = (OSError, paramiko.SSHException, ValueError, KeyError, RuntimeError)


def to_cluster_path(path: Path | str, cluster: dict) -> str:
    """The path as the cluster sees it: a Windows drive or UNC path under path_map is mapped to path_map.to_cluster."""
    value = str(path)
    mapping = cluster.get("path_map") or {}
    to_root = str(mapping.get("to_cluster") or "").rstrip("/")
    from_roots = [
        str(mapping.get("from_windows") or "").rstrip("\\/"),
        str(mapping.get("from_windows_unc") or "").rstrip("\\/"),
    ]
    for from_root in [root for root in from_roots if root]:
        if to_root and value.lower().startswith(from_root.lower()):
            suffix = value[len(from_root):].lstrip("\\/")
            return to_root + ("/" + suffix.replace("\\", "/") if suffix else "")
    normalized = value.replace("\\", "/")
    if to_root and g.DATA_SHARE_NAME in normalized:
        suffix = normalized.split(g.DATA_SHARE_NAME, 1)[1].lstrip("/")
        return to_root + ("/" + suffix if suffix else "")
    return normalized


def _config_or_env(cluster: dict, key: str, env_key: str, default: str | None = None) -> str | None:
    value = cluster.get(key)
    if value not in (None, ""):
        return str(value)
    return os.environ.get(env_key, default)


def _credentials(cluster: dict) -> dict:
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
    return {"hostname": host, "port": port, "username": username, "password": password, "key_filename": key_filename}


def check_ready(cluster: dict) -> None:
    """Raise a clear error now when a submission could not work: unsupported submission method or incomplete credentials."""
    if str(cluster.get("submission_method") or "paramiko").strip().lower() != "paramiko":
        raise ValueError("Cluster STA submission supports only Paramiko remote sbatch.")
    _credentials(cluster)


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


def _decode(chunks: list[bytes]) -> str:
    return b"".join(chunks).decode("utf-8", errors="replace")


def _read_until_exit(channel, command: str, command_timeout: int) -> tuple[str, str, int]:
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
            return _decode(out_chunks), _decode(err_chunks), int(channel.recv_exit_status())
        if time.monotonic() >= deadline:
            channel.close()
            message = f"SSH command timed out after {command_timeout} seconds: {command}"
            return _decode(out_chunks), (_decode(err_chunks) + "\n" + message).strip(), 124
        time.sleep(0.1)


def _ssh_once(cluster: dict, command: str, timeout: int | None) -> tuple[str, str, int]:
    cred = _credentials(cluster)
    default_timeout = int(cluster.get("paramiko_timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    command_timeout = int(timeout or cluster.get("command_timeout_seconds") or default_timeout)
    connect_timeout = int(cluster.get("connect_timeout_seconds") or timeout or default_timeout)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(**cred, timeout=connect_timeout, banner_timeout=connect_timeout, auth_timeout=connect_timeout)
        _, stdout, _ = ssh.exec_command(command, timeout=command_timeout)
        stdout.channel.settimeout(1.0)
        return _read_until_exit(stdout.channel, command, command_timeout)
    except socket.timeout as exc:
        return "", f"SSH command timed out after {command_timeout} seconds: {command}. {exc}", 124
    finally:
        _close_ssh(ssh)


def run_remote(cluster: dict, command: str, timeout: int | None = None) -> tuple[str, str, int]:
    """Run one shell command on the cluster: (stdout, stderr, exit code). Only connection errors are retried
    (cluster.paramiko_attempts attempts, growing pauses); the last one propagates."""
    attempts = int(cluster.get("paramiko_attempts") or DEFAULT_ATTEMPTS)
    for attempt in range(1, attempts + 1):
        try:
            return _ssh_once(cluster, command, timeout)
        except CONNECTION_ERRORS as exc:
            if attempt == attempts:
                raise
            delay = min(30, 2 * attempt)
            print(f"Cluster command attempt {attempt}/{attempts} failed: {type(exc).__name__}: {exc}. Retrying in {delay}s.", file=sys.stderr, flush=True)
            time.sleep(delay)
    raise RuntimeError(f"Cluster command was not run: cluster.paramiko_attempts is {attempts}, it must be at least 1.")


def _sbatch_parts(cluster: dict, script, args, chdir, log, dependency, kill_on_invalid_dep, error_log) -> list[str]:
    parts = shlex.split(str(cluster.get("submit_command") or DEFAULT_SBATCH))
    if dependency:
        parts.append(f"--dependency=afterok:{dependency}")
    if kill_on_invalid_dep:
        parts.append("--kill-on-invalid-dep=yes")
    parts += [f"--chdir={to_cluster_path(chdir, cluster)}", f"--output={to_cluster_path(log, cluster)}"]
    if error_log is not None:
        parts.append(f"--error={to_cluster_path(error_log, cluster)}")
    return parts + [to_cluster_path(script, cluster)] + [to_cluster_path(arg, cluster) for arg in args]


def _parse_job_id(text: str) -> str | None:
    return next((token for token in text.replace(".", " ").split() if token.isdigit()), None)


def submit_job(cluster: dict, script, args, chdir, log, dependency: str | None = None, kill_on_invalid_dep: bool = False,
               error_log=None) -> str:
    """sbatch the script with the args (local paths, mapped to cluster paths here); returns the Slurm job id.
    error_log is omitted from the command when None (Slurm then writes stderr to the log). Raises RuntimeError with the sbatch output."""
    command = shlex.join(_sbatch_parts(cluster, script, args, chdir, log, dependency, kill_on_invalid_dep, error_log))
    stdout, stderr, code = run_remote(cluster, command)
    if code != 0:
        raise RuntimeError(f"Failed to submit {Path(str(script)).name} (sbatch exit {code}): {(stderr or stdout).strip()}")
    job_id = _parse_job_id(stdout)
    if not job_id:
        raise RuntimeError(f"Could not parse Slurm job id from sbatch output: {stdout!r}")
    return job_id


def job_state(cluster: dict, job_id: str, timeout: int | None = None) -> tuple[list[str], str]:
    """The Slurm states of the job, from squeue (queued/running jobs) else sacct (finished jobs), and a note: the sacct error when it
    failed. No states and an empty note: Slurm does not know the job."""
    squeue = str(cluster.get("squeue_command") or DEFAULT_SQUEUE)
    sacct = str(cluster.get("sacct_command") or DEFAULT_SACCT)
    sq_out, _, sq_code = run_remote(cluster, f"{squeue} -h -j {job_id} -o '%T'", timeout)
    if sq_code == 0 and sq_out.strip():
        return [line.strip() for line in sq_out.splitlines() if line.strip()], ""
    acct_out, acct_err, acct_code = run_remote(cluster, f"{sacct} -n -j {job_id} --format=State,ExitCode -P", timeout)
    if acct_code != 0:
        return [], (acct_err or acct_out).strip()
    return [line.split("|", 1)[0].strip() for line in acct_out.splitlines() if line.strip()], ""


def tail_remote_file(cluster: dict, remote_path: str, lines: int = 4000, timeout: int | None = None) -> str:
    """The last lines of a file on the cluster; the error text when tail failed."""
    posix_path = remote_path.replace("\\", "/")
    out, err, code = run_remote(cluster, f"tail -n {int(lines)} {shlex.quote(posix_path)}", timeout)
    return out if code == 0 else (err or out)
