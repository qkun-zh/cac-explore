# cac-explore

复刻 [HypoExplore](https://arxiv.org/abs/2604.12999)（Agentic Discovery with Active Hypothesis Exploration）的多智能体架构发现框架，应用于 CAC 视觉模型研究（FSC147 类无关计数基准）。

## 这是什么

一个 Git 仓库即所有智能体（主控 + 子智能体）共享的文件系统：

- `tree/` — **轨迹树 T**：每个实验节点一个目录（假设、代码、结果、四方反馈、综合结论）
- `memory/` — **假设记忆库 M**：跨节点积累的已验证/被驳斥/存疑设计知识
- `STATE.md` + `journal/` — 交接现场：任何智能体中途接管都能无损续跑
- `tasks/` — 原子认领式任务队列
- `code/engine/train.py` — 所有节点共用的训练框架；节点自带 `model.py` + `config.py`

## 快速上手

```bash
# 新智能体/新会话
git pull --ff-only && cat AGENTS.md && cat STATE.md

# 冒烟测试（无需数据集）
python code/engine/train.py --smoke --epochs 2

# 远程跑某节点（在服务器上）
bash scripts/run_node.sh N0001

# 回传结果并提交（在本地）
bash scripts/collect_node.sh N0001
```

详见 `AGENTS.md`（协议）与 `docs/PROTOCOL.md`（文件契约细节）。
