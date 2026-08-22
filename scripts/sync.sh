#!/usr/bin/env bash
# 本地侧同步：拉取远程变更并推送本地提交。任何智能体收尾时可调用。
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
git pull --ff-only --autostash
if ! git diff --quiet HEAD || [ -n "$(git status --porcelain)" ]; then
  git add -A
  git commit -m "sync: agent state update $(date +%F-%H:%M)" || true
fi
git push origin main
echo "[sync] done."
