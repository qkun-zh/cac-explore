# STATE — 实时交接文档

> 任何智能体启动后第一件事（除本协议外）就是读这里。每完成一个原子步骤必须更新。

- 最后更新：2026-08-22T21:40+08:00
- 更新者：setup（主控）

## 当前阶段

**阶段 0 — 框架搭建**：骨架已建、服务器已就绪、冒烟测试通过。等待正式研究循环启动。

## 活跃节点

无。

## 下一步行动（按优先级）

1. 将 FSC147 数据集放到服务器 `/data/dataset/FSC147`（期望布局见 `docs/research_direction.md` §数据）
2. 主控创建根节点 `N0001` 的 `idea.md`，发起第一个完整研究循环
   （Idea → Coding → 冗余检查 → Executor → 反馈×4 → Synthesis → 记忆更新）

## 阻塞项

- FSC147 数据集尚未上传到服务器（冒烟模式不受影响，用合成数据）。

## 关键事实

- 仓库：<https://github.com/qkun-zh/cac-explore>（公开）
- 服务器：`ssh cac-server`；仅 `/data` 持久化
- conda 环境：`cac`（新建，含 torch CUDA）
- 冒烟测试记录：`tree/nodes/S0001_smoke/`
