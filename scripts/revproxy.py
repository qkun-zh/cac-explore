"""Spare utility: local socks proxy reachable from the server via ssh -R (unused by default; GitHub is directly reachable)."""
import paramiko, socket, select, threading, re, os, urllib.parse

REMOTE_LISTEN = 1081
conf = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "local", "address_and_password.md")).read().splitlines()
m = re.search(r"-p (\d+)\s+\w+@([\w.-]+)", conf[0])
PORT, HOST, USER = int(m.group(1)), m.group(2), "root"
PW = conf[1].strip()
prox = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or os.environ.get("ALL_PROXY") or ""
prox = prox if prox.startswith("http") else "http://" + prox
pu = urllib.parse.urlparse(prox)
HTTP_PROXY = (pu.hostname, pu.port or 80)

def http_connect(host, port, timeout=20):
    up = socket.create_connection(HTTP_PROXY, timeout=timeout)
    up.sendall(f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        d = up.recv(4096)
        if not d: raise ConnectionError("proxy closed")
        resp += d
    if resp.split(b" ", 2)[1] != b"200":
        raise ConnectionError(f"proxy CONNECT failed: {resp[:200]}")
    return up

def handle(channel):
    up = None
    try:
        data = channel.recv(4096)
        if not data or data[0] != 0x05:
            channel.close(); return
        channel.sendall(b"\x05\x00")
        data = channel.recv(4096)
        if not data: channel.close(); return
        atyp = data[3]
        if atyp == 1:
            host = socket.inet_ntoa(data[4:8]); port = int.from_bytes(data[8:10], "big")
        elif atyp == 3:
            ln = data[4]; host = data[5:5+ln].decode(); port = int.from_bytes(data[5+ln:7+ln], "big")
        else:
            channel.close(); return
        up = http_connect(host, port)
        channel.sendall(b"\x05\x00\x00\x01" + socket.inet_aton("0.0.0.0") + b"\x00\x00")
        while True:
            r, _, _ = select.select([channel, up], [], [], 10)
            if channel in r:
                d = channel.recv(65536)
                if not d: break
                up.sendall(d)
            if up in r:
                d = up.recv(65536)
                if not d: break
                channel.sendall(d)
    except Exception as e:
        print("handle err", e, flush=True)
    finally:
        try: channel.close()
        except: pass
        try: up.close()
        except: pass

def main():
    c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30)
    t = c.get_transport()
    t.request_port_forward("", REMOTE_LISTEN)
    print(f"reverse SOCKS on remote {REMOTE_LISTEN} via {HTTP_PROXY}", flush=True)
    while True:
        try:
            chan = t.accept(10)
            if chan is None: continue
            threading.Thread(target=handle, args=(chan,), daemon=True).start()
        except Exception as e:
            print("accept err", e, flush=True); break

main()
