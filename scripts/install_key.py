#!/usr/bin/env python3
"""服务器轮换后的一键重接：读凭据文件 -> 装公钥(含/data持久副本) -> 重写本地 ssh config。

用法: python3 scripts/install_key.py [凭据文件路径]
凭据文件格式（默认 <仓库>/local/address_and_password.md）:
  第1行: ssh -p <端口> <用户>@<主机>
  第2行: <密码>
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
    lines = open(path).read().splitlines()
    m = re.search(r"-p\s+(\d+)\s+(\S+)@([\w.-]+)", "\n".join(lines[:3]))
    if not m:
        sys.exit(f"[install_key] 无法从 {path} 解析 ssh 连接串")
    port, user, host = int(m.group(1)), m.group(2), m.group(3)
    pw = next((l.strip() for l in lines if l.strip() and l.strip() != lines[0].strip() and not l.startswith("ssh ") and len(l.strip()) > 8), None)
    if not pw:
        pw = lines[1].strip()
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
        sys.exit(f"[install_key] 远程安装失败: {out} {err}")

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

    print(f"[install_key] 公钥已装入 {user}@{host}:{port}（含 /data/.ssh 持久副本）")
    print(f'[install_key] 本地别名已更新: ssh {ALIAS}')


if __name__ == "__main__":
    main()
