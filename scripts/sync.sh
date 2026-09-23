#!/bin/bash
# Sync this repo to the canonical server run dir /data/cac.
# First sync archives the pre-v2 server tree at /data/cac_v1 (kept forever).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
tar czf - --exclude .git --exclude neug.db --exclude __pycache__ --exclude local -C "$ROOT" . \
  | ssh cac-server '
      set -e
      rm -rf /data/cac_new
      mkdir -p /data/cac_new
      tar xzf - -C /data/cac_new
      if [ -d /data/cac ]; then
        if [ ! -d /data/cac_v1 ]; then
          mv /data/cac /data/cac_v1
        else
          rm -rf /data/cac
        fi
      fi
      mv /data/cac_new /data/cac
      echo "SYNCED -> /data/cac (archive: /data/cac_v1)"
    '
