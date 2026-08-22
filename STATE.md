# STATE — 实时交接文档

> 任何智能体启动后第一件事（除本协议外）就是读这里。每完成一个原子步骤必须更新。

- 最后更新：2026-08-22T22:55+08:00
- 更新者：setup（主控）

## 当前阶段

**阶段 0 — 框架搭建**（收尾中）：服务器环境重建 + torch cu128 修复进行中；冒烟节点 S0001_smoke 已推送待执行。

## 活跃节点

`S0001_smoke` — 已创建并推送，等待服务器 torch 就绪后 `run_node.sh` 执行。

## 下一步行动（按优先级）

1. 确认 `/data/asset/torch_fix.log` 出现 TORCH_OK 且 `cuda.is_available()=True`
2. 服务器：`cd /data/repo && bash scripts/run_node.sh S0001_smoke`（tmux 会话 node_S0001_smoke）
3. 本地：`bash scripts/collect_node.sh S0001_smoke` 回传 result.json → 提交
4. 上传 FSC147 数据集到 `/data/dataset/FSC147`（布局见 docs/research_direction.md）
5. 创建正式根节点 N0001，启动第一个完整研究循环

## 阻塞项

- FSC147 数据集未上传（不影响 --smoke 冒烟）。
- torch 修复中：阿里源下载 916MB wheel（会话 fixtorch）。

## 关键事实

- 仓库：<https://github.com/qkun-zh/cac-explore>（公开）
- 服务器：`ssh cac-server`；仅 /data 持久化；轮换恢复流程见 AGENTS.md「服务器轮换演练」
- conda 环境：`cac`（Python 3.12.14）；**torch 必须锁 2.10.x+cu128**（驱动 12.4，最新版 cu130 不兼容——教训见 memory/failure_modes.md）
- 反向代理：本地 revproxy.py 常驻后服务器可用 socks5://127.0.0.1:1081（已验证 200）
