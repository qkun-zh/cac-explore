# cac-explore — 轻量 CAC 计数模型的多智能体发现系统

## 使命（Mission）

> **构造一个 ≤32M 参数、在 FSC147 test 集上 MAE < 16 的轻量创新型类无关计数（Class-Agnostic Counting）模型。**

这是超越现有公开方法的硬目标。主控（Lead）不亲自写模型——模型由本仓库定义的**多智能体假设探索循环**自动产出。任何新接手的 agent，读完本文件 + `AGENTS.md` + `STATE.md` 三份文件即可无损接管。

参考基线：`tree/nodes/S0001_smoke/`（0.01M 参数玩具网络，2 epoch val MAE 46.7）——仅证明全链路可用，与目标不可比。

## 系统架构

复刻 [HypoExplore](https://arxiv.org/abs/2604.12999)：**一个 Git 仓库 = 所有智能体共享的文件系统**。所有状态都在文件里，不在任何 agent 的上下文里。

```
┌─────────────────────────────┐
│ 本地 WSL Debian              │  Lead + Idea/Coding/反馈/Synthesis agents
│ ~/cac_explore               │  ★ 唯一有 push 权限的机器
└──────────┬──────────────────┘
           │ git push / pull
┌──────────▼──────────────────┐
│ GitHub: qkun-zh/cac-explore │  共享总线（公共仓库）
└──────────┬──────────────────┘
           │ git pull（服务器只读）
┌──────────▼──────────────────┐
│ DeepLn 租用服务器            │  唯一有 GPU 的地方（RTX 3060 12GB）
│ /data/repo + /data/runs     │  tmux 训练；产物经 SSH 回传本地入库
└─────────────────────────────┘
```

## 目录地图（每个文件的作用）

| 路径 | 作用 |
|---|---|
| `AGENTS.md` | **协议**：启动顺序、各角色的标准工作循环、硬性规则、服务器速查、轮换演练 |
| `STATE.md` | **当前现场快照**：阶段、已验证事实、活跃任务、下一步。第二个必读 |
| `docs/PROTOCOL.md` | 文件契约细节：节点目录里每个文件的必填结构、研究循环公式、状态机 |
| `docs/research_direction.md` | 研究方向备忘：CAC 领域现状、FSC147 数据协议、候选技术路线 |
| `code/engine/train.py` | **唯一训练入口**，所有节点共用。契约：读节点的 model/config，输出 result.json |
| `code/data/fsc147.py` | FSC147 VarV2 数据加载器（预计算密度图、总和守恒缩放、exemplar box 解析） |
| `code/selection/select_next.py` | 轨迹树扩展策略：选父节点（quality×avail 加权）、选假设（Thompson 采样 + 认知价值） |
| `scripts/run_node.sh` | 【服务器】tmux 启动某节点训练（会话名 `node_<ID>`，墙钟 30min 超时） |
| `scripts/collect_node.sh` | 【本地】SSH 回传某节点的 result.json + train.log 尾部并放入节点目录 |
| `scripts/check_data.py` | 数据集健全性检查（划分数量、形状、计数守恒） |
| `scripts/bootstrap_remote.sh` | 【服务器】新实例一键初始化环境（幂等） |
| `scripts/install_key.py` | 【本地】实例轮换后重装 SSH 公钥并更新连接别名 |
| `scripts/rebuild_index.py` | 从 hypotheses.jsonl 重建 index.json（快照损坏时自救） |
| `tasks/_template.md` | 任务卡模板；`T####_pending_*.md` 待领，**改名 `*_claimed_*` 即占有** |
| `journal/events.jsonl` | 全局审计流水（append-only）：谁在何时做了什么 |
| `memory/hypotheses.jsonl` | **假设记忆库**（append-only，永不改写历史行） |
| `memory/index.json` | 假设库的可重建快照：每条假设的置信度/状态/证据日志 |
| `memory/failure_modes.md` | **已踩坑清单**：Coding agent 动手前必读，事故后必须追加 |
| `tree/tree.json` | 轨迹树 T：节点父子关系 + status/best_metric/score，由 Synthesis 维护 |
| `tree/nodes/<ID>/` | 每个实验节点一个自包含目录：idea → code → result → feedback ×4 → synthesis |

节点 ID 约定：`S0001_smoke` 为冒烟节点；正式节点 `N0002_<短名>` 起。

## 新 agent 快速上手

```bash
git pull --ff-only          # 1. 同步到最新
cat AGENTS.md STATE.md      # 2. 协议 + 现场（按需再读 docs/PROTOCOL.md）
tail -5 journal/events.jsonl # 3. 最近发生了什么
```

然后按 `AGENTS.md` 中你的角色循环行事。
