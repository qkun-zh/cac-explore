#!/usr/bin/env bash
# Local sync: pull remote changes and push local commits. Any agent may call this when closing out.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
git pull --ff-only --autostash
if ! git diff --quiet HEAD || [ -n "$(git status --porcelain)" ]; then
  git add -A
  git commit -m "sync: agent state update $(date +%F-%H:%M)" || true
fi
git push origin main
echo "[sync] done."
