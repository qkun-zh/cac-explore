# PROTOCOL — 文件契约与节点生命周期

## 1. 同步拓扑

```
本地 ~/cac_explore ──push/pull──> GitHub <──pull── 服务器 /data/repo
 （主控+子智能体编辑）                          （执行训练）
        ▲                                              │
        └──────── scripts/collect_node.sh (SSH/SFTP) ──┘
```

**单一写入者规则**：只有本地 push。服务器对仓库只读，实验产物写 `/data/runs/` 后由采集脚本回传。

## 2. 节点目录契约 `tree/nodes/<ID>/`

ID 格式：`N####_<短名>`（正式节点）/ `S0001_smoke`（冒烟）。

| 文件 | 作者 | 内容 |
|---|---|---|
| `idea.md` | Idea Agent | 固定小节：`## 标题`、`## 动机与直觉`、`## 架构规格`（core_ideas/core_blocks/network_structure/tunable_aspects/invariants）、`## 提出的假设`（每条含 falsification）、`## 与父节点的差异`、`## 新颖性声明` |
| `model.py` | Coding Agent | 必须暴露 `build_model(cfg) -> nn.Module`；输入 `[B,3,H,W]`+bbox，输出 dict 含 `density`；**density 形状必须与数据集密度目标一致**（engine 不做隐式插值，形状不符直接报错） |
| `config.py` | Coding Agent | 必须 `cfg = dict(...)`，必含键：`exp_name, epochs, batch_size, lr, input_size, num_classes, smoke(默认False)`。可自由增键 |
| `result.json` | Executor 回传 | `{node, status: success\|failed\|timeout, metrics:{mae,rmse,...}, timing:{train_seconds, epochs_done}, diagnostics:{oom,instability,notes}, run_dir}` |
| `feedback/quantitative.md` 等 ×4 | 反馈 Agent | 每份固定结构：`## reasoning` / `## actionable_feedback` / `## hypothesis_updates`（列表：hypothesis_id, evidence_type∈supports/contradicts/neutral, strength∈[0,1], reasoning）；diagnostic 仅失败/超时时存在 |
| `synthesis.md` | Synthesis | 合并去重后的更新、质量门判定（7 维度）、落账清单 |
| `train.log` | 采集脚本 | 服务器完整日志的截尾副本（≤500 行） |

## 3. 全局状态文件

- `tree/tree.json`：`nodes: {<ID>: {parent, children[], status(proposed|coded|running|done|failed|timeout|synthesized), best_metric, train_seconds, quality, avail, score}}`。由 Synthesis 步骤维护。
- `memory/hypotheses.jsonl`：每行一条事件：`{ts, type(create\|evidence\|revise), hyp_id, text?, evidence_type?, strength?, source_node?}`。
- `memory/index.json`：`{<hyp_id>: {text, confidence, n_tested, status(confirmed|refuted|uncertain), tags[], log[]}}`——jsonl 的物化快照，可用 `scripts/rebuild_index.py` 重建。
- `journal/events.jsonl`：审计流水，只追加。

## 4. 研究循环（每个节点的标准流程）

1. **选父**：`python code/selection/select_next.py parent`
   `score = λ_parent·quality + (1−λ_parent)·avail`，其中
   `quality = λ_acc·Acc_norm + (1−λ_acc)·(1 − min(τ,τ_max)/τ_max)`，λ_acc=0.85，λ_parent=0.60，τ_max=30min
2. **选假设**：`select_next.py hypo --parent <ID>`
   开发组 = Thompson 采样 Beta(α,β)（先验(1,1)，w=1）取 top-2；探索组 = 认知价值 `1−|2c−1|` 取 top-2；并集 ≤4
3. Idea → Coding → 冗余检查（对照既有 idea.md 判机制重复）
4. Executor：服务器 `bash scripts/run_node.sh <ID>`（tmux 会话 `node_<ID>`，墙钟超时 30min，前 5 epoch sanity-check）
5. 本地 `bash scripts/collect_node.sh <ID>` 回传
6. 反馈 ×4（可并行认领 tasks 卡）→ Synthesis（去重、矛盾消解、质量门、η=0.20 更新置信度）
7. 更新 tree.json / index.json / STATE.md / journal，提交推送

## 5. 冒烟模式

任何新代码先过 `--smoke`：合成随机数据、2 epoch、CPU 可跑，验证 model/config 契约后再上真数据。
