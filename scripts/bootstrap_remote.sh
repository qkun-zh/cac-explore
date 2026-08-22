#!/usr/bin/env bash
# 服务器重建脚本（幂等）。在服务器上以 root 运行。
set -euo pipefail
echo "== 1. 基础工具 =="
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux)
mkdir -p /data/repo /data/dataset /data/runs /data/asset

echo "== 2. 克隆仓库 =="
if [ ! -d /data/repo/.git ]; then
  git clone https://github.com/qkun-zh/cac-explore.git /data/repo
else
  git -C /data/repo pull --ff-only || true
fi

echo "== 3. conda 环境 ="
PY=/data/miniconda/envs/cac/bin/python
if [ ! -x "$PY" ]; then
  /data/miniconda/bin/conda create -y -n cac python=3.12
  $PY -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision --no-cache-dir
  $PY -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy opencv-python-headless scipy pillow tqdm pyyaml --no-cache-dir
  rm -rf /root/.cache/pip
fi
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "== 4. FSC147 数据集 =="
[ -f /data/dataset/FSC147/annotation_FSC147_384.json ] && [ -d /data/dataset/FSC147/images_384_VarV2 ] && echo "FSC147(VarV2) 就绪" || echo "!! FSC147 未就位（训练用 --smoke 模式不受影响）"
echo "[bootstrap_remote] 完成。"
