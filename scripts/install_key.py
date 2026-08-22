#!/usr/bin/env python3
"""One-shot server re-onboarding after rotation: read creds file -> install pubkey (with /data persistent copy) -> rewrite local ssh config.

Usage: python3 scripts/install_key.py [creds_file_path]
Creds file format (default <repo>/local/address_and_password.md):
  line 1: ssh -p <port> <user>@<host>
  line 2: <password>
"""
import os
import re
import sys
import stat

HOME = os.path.expanduser("~")
PUBKEY_PATH = os.path.join(HOME, ".ssh", "id_ed25519.pub")
CONF_PATH = os.path.join(HOME, ".ssh", "config")
ALIAS = "cac-server"
DEFAULT_CREDS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "local", "address_and_password.md")


def parse_creds(path):
    """Tolerant to any paste format: one connection line + one password line; titles/comments/label prefixes OK."""
    lines = open(path).read().splitlines()
    m = re.search(r"ssh\s+\S*\s*-p\s+(\d+)\s+(\S+)@([\w.-]+)", "\n".join(lines))
    if not m:
        sys.exit(f"[install_key] cannot parse connection string from {path} (expected: ssh -p <port> <user>@<host>)")
    port, user, host = int(m.group(1)), m.group(2), m.group(3)
    pw = None
    for l in lines:  # first non-empty non-comment line after the connection string = password
        s = l.strip()
        if not s or s.startswith("#") or "ssh" in s.split()[0:1]:
            continue
        pw = s.split("：")[-1].split(":")[-1].strip()
        break
    if not pw or len(pw) < 6:
        sys.exit("[install_key] password line not found: the next non-empty line after the ssh string should be the password")
    return host, port, user, pw


def main():
    creds = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CREDS
    host, port, user, pw = parse_creds(creds)
    pub = open(PUBKEY_PATH).read().strip()

    import paramiko
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username=user, password=pw, timeout=20)
    cmd = (
        'mkdir -p /root/.ssh /data/.ssh && chmod 700 /root/.ssh /data/.ssh; '
        f'grep -qF "{pub}" /root/.ssh/authorized_keys 2>/dev/null || echo "{pub}" >> /root/.ssh/authorized_keys; '
        f'cp -f /root/.ssh/authorized_keys /data/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys /data/.ssh/authorized_keys; '
        'echo KEY_INSTALLED'
    )
    _, o, e = c.exec_command(cmd)
    out = o.read().decode().strip()
    err = e.read().decode().strip()
    c.close()
    if "KEY_INSTALLED" not in out:
        sys.exit(f"[install_key] remote install failed: {out} {err}")

    block = (
        f"Host {ALIAS}\n"
        f"    HostName {host}\n"
        f"    Port {port}\n"
        f"    User {user}\n"
        f"    IdentityFile ~/.ssh/id_ed25519\n"
        f"    StrictHostKeyChecking no\n"
    )
    os.makedirs(os.path.dirname(CONF_PATH), exist_ok=True)
    existing = open(CONF_PATH).read() if os.path.exists(CONF_PATH) else ""
    if f"Host {ALIAS}\n" in existing:
        existing = re.sub(rf"Host {ALIAS}\n(?:[ \t]+\S.*\n?)+", block, existing)
    else:
        existing = block + ("\n" + existing if existing.strip() else "")
    open(CONF_PATH, "w").write(existing)
    os.chmod(CONF_PATH, stat.S_IRUSR | stat.S_IWUSR)

    print(f"[install_key] pubkey installed on {user}@{host}:{port} (with /data/.ssh persistent copy)")
    print(f'[install_key] local alias updated: ssh {ALIAS}')


if __name__ == "__main__":
    main()
