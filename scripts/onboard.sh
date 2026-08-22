#!/usr/bin/env bash
# One-command server onboarding: read local/ creds -> install key/write alias -> ensure repo -> init env -> verify data.
# Prerequisite: paste the two lines from DeepLn verbatim into local/address_and_password.md (line1 ssh cmd, line2 password).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -s local/address_and_password.md ]; then
    echo "[onboard] First paste the two credential lines into local/address_and_password.md:"
    echo "          line 1: ssh -p <port> root@<host>"
    echo "          line 2: <password>"
    exit 1
fi

echo "== step 1/3: install pubkey and update local ssh alias"
python3 scripts/install_key.py

echo "== step 2/3: ensure repo exists and initialize environment (idempotent)"
ssh -o BatchMode=yes -o ConnectTimeout=20 cac-server '
    [ -d /data/repo/.git ] || git clone https://github.com/qkun-zh/cac-explore.git /data/repo
    bash /data/repo/scripts/bootstrap_remote.sh'

echo "== step 3/3: verify"
if ssh -o BatchMode=yes cac-server '/data/miniconda/envs/cac/bin/python /data/repo/scripts/check_data.py'; then
    echo "[onboard] ALL READY - go build"
else
    echo "[onboard] NOT ready: if dataset is missing, upload FSC147.zip to /data/dataset/, unzip inside FSC147/, then re-run this script"
    exit 1
fi
