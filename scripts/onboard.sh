#!/usr/bin/env bash
# 新服务器一键接管：读 local/ 凭据 -> 装公钥/写别名 -> 确保仓库 -> 初始化环境 -> 验证数据。
# 前提：用户已把 DeepLn 给的两行原样粘到 local/address_and_password.md（第1行ssh命令，第2行密码）。
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -s local/address_and_password.md ]; then
    echo "[onboard] 请先把两行凭据粘贴到 local/address_and_password.md："
    echo "          第1行: ssh -p <端口> root@<主机>"
    echo "          第2行: <密码>"
    exit 1
fi

echo "== 步骤1/3：安装公钥并更新本地 ssh 别名"
python3 scripts/install_key.py

echo "== 步骤2/3：确保仓库存在并初始化环境（幂等）"
ssh -o BatchMode=yes -o ConnectTimeout=20 cac-server '
    [ -d /data/repo/.git ] || git clone https://github.com/qkun-zh/cac-explore.git /data/repo
    bash /data/repo/scripts/bootstrap_remote.sh'

echo "== 步骤3/3：验证"
if ssh -o BatchMode=yes cac-server '/data/miniconda/envs/cac/bin/python /data/repo/scripts/check_data.py'; then
    echo "[onboard] ✅ 全部就绪，可以开工"
else
    echo "[onboard] ⚠️ 环境或数据未就绪：数据集丢失时请上传 FSC147.zip 到 /data/dataset/ 并在 FSC147/ 内解压后重试本脚本"
    exit 1
fi
