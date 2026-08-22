# AGENTS.md — cac-explore 多智能体协作协议

复刻 HypoExplore（arXiv:2604.12999）的共享文件系统多智能体框架，用于 CAC 视觉模型研究（FSC147 类无关计数）。

**核心原则：一个 Git 仓库 = 所有智能体共享的文件系统。所有状态都在文件里，不在任何智能体的上下文里。任何智能体随时可以从文件重建完整现场。**

---

## 启动顺序（每个智能体/新会话必做）

1. 读本文档
2. `git pull --ff-only`
3. 读 `STATE.md`（当前阶段、活跃节点、下一步、阻塞项）
4. 需要细节再读：`docs/PROTOCOL.md`（文件契约）、`journal/events.jsonl` 尾部（最近发生了什么）

## 收尾动作（每完成一个原子步骤）

1. 更新 `STATE.md`
2. 追加一行到 `journal/events.jsonl`：`{"ts":"<ISO8601>","agent":"<名字>","event":"<做了什么>","refs":["<涉及文件>"]}`
3. 本地侧提交推送：`git add -A && git commit -m "<简洁描述>" && git push`

## 角色与产出文件（契约详见 docs/PROTOCOL.md）

| 角色 | 产出 |
|---|---|
| Idea Agent | `tree/nodes/<ID>/idea.md` |
| Coding Agent | 同目录 `model.py` + `config.py`（动手前先读 `memory/failure_modes.md`） |
| Executor（远程） | tmux 训练 → `/data/runs/<ID>/` → 回传 `result.json` + `train.log` |
| 反馈 Agent ×4 | `feedback/{quantitative,qualitative,causal,diagnostic}.md` |
| Synthesis | `synthesis.md` + 追加 `memory/hypotheses.jsonl` + 更新 `memory/index.json` |

## 硬性规则

1. **只有本地机器 push**；服务器只 pull。实验产物由 `scripts/collect_node.sh` 经 SSH 回传后本地提交
2. **大文件永不进 git**：数据集、checkpoint、完整日志只存服务器 `/data/dataset`、`/data/runs`
3. 任务认领是原子的：把 `tasks/T####_pending_*.md` 改名为 `*_claimed_<agent>_*.md` 即占有；互斥资源用 `mkdir locks/<name>` 抢锁，用完删除
4. `memory/hypotheses.jsonl` **只追加、永不改写**；`memory/index.json` 是它的可重建快照
5. 远程一切 >1 分钟的任务必须在 tmux 会话里跑，禁止裸 SSH 挂前台
6. 改完代码先跑冒烟：`python code/engine/train.py --smoke --epochs 2`

## 服务器速查

| 项 | 值 |
|---|---|
| 连接 | `ssh cac-server`（配置在本机 `~/.ssh/config`） |
| 持久化 | 仅 `/data`。目录：`/data/repo`（仓库克隆）、`/data/dataset/FSC147`、`/data/runs/<节点>`、`/data/asset` |
| Python | `/data/miniconda/envs/cac/bin/python` |
| GPU | RTX 3060 12GB（单卡） |
| 网络 | GitHub 直连可用；PyPI 用清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| tmux | 会话名约定 `node_<ID>`；查看 `tmux capture-pane -t node_<ID> -p \| tail -30` |

## 假设记录格式（HypoExplore 式）

```
IF [架构选择] IN [作用域], THEN [预测效果], BECAUSE [机制]. DISPROVED IF [证伪条件].
```

置信度更新（η=0.20，c∈[0.01,0.99]，初始 0.5）：
- 支持：`c ← c + 0.20·w·(1−c)`
- 反驳：`c ← c − 0.20·w·c`
- 状态判定：confirmed >0.75 ／ refuted <0.25 ／ uncertain 其间

证据来自反馈智能体（evidence_type ∈ supports/contradicts/neutral，强度 w∈[0,1]），由 Synthesis 统一落账。
