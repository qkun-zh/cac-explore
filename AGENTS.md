# AGENTS.md — 多智能体协作协议

**使命**：≤32M 参数、FSC147 test MAE < 16 的轻量创新 CAC 模型。通过 HypoExplore 式假设探索循环逼近，不靠单次英雄式设计。

---

## 0. 启动顺序（每个 agent / 新会话必做）

1. 读本文档
2. `git pull --ff-only`
3. 读 `STATE.md`（阶段、已验证事实、活跃任务、下一步）
4. 需要细节再读：`docs/PROTOCOL.md`（文件契约）、`journal/events.jsonl` 尾部（最近事件）、`memory/failure_modes.md`（避坑）

---

## 1. 你的工作循环（按角色对号入座）

### Idea Agent
1. 读 `memory/index.json` + `memory/failure_modes.md` + 父节点的 `synthesis.md`
2. 按 `docs/PROTOCOL.md §4` 选父节点、选假设
3. 写 `tree/nodes/<ID>/idea.md`（固定小节，含可证伪条件；对照已有 idea 声明新颖性）
4. 建 `tasks/T####_pending_coding_<ID>.md` 任务卡 → 提交推送

### Coding Agent
1. 认领任务卡（改名），读该节点 `idea.md` + **必读** `memory/failure_modes.md`
2. 写 `model.py`（暴露 `build_model(cfg)`）+ `config.py`（cfg 必含键见 PROTOCOL §2）
3. 冒烟自检（无需数据集）：
   ```bash
   python code/engine/train.py --node_dir tree/nodes/<ID> --smoke --epochs 2
   ```
   本地无 torch 则在服务器跑同样命令。**契约不过不许 push**
4. 任务卡改名 `_done_`，推送；建 executor 任务卡

### Executor（服务器侧，由 Lead 触发）
1. 本地：`git push` 后在服务器拉取并启动：
   ```bash
   ssh cac-server 'cd /data/repo && git pull && bash scripts/run_node.sh <ID>'
   ```
2. 盯进度：`ssh cac-server 'tmux capture-pane -t node_<ID> -p | tail -30'`
3. 完成后本地回传入库：
   ```bash
   bash scripts/collect_node.sh <ID>
   git add -A && git commit -m "result: <ID> status=..." && git push
   ```

### 反馈 Agent ×4（quantitative / qualitative / causal / diagnostic）
1. 认领对应任务卡；读节点 `idea.md` + `model.py` + `config.py` + `result.json` + `train.log`
2. 写 `tree/nodes/<ID>/feedback/<维度>.md`（固定结构见 PROTOCOL §2，含 hypothesis_updates 列表）

### Synthesis Agent
1. 四份反馈齐后合并去重、消解矛盾、质量门判定
2. 写 `synthesis.md`；按 η=0.20 规则更新每条相关假设置信度
3. 落账：追加 `memory/hypotheses.jsonl` → `python scripts/rebuild_index.py`
4. 更新 `tree/tree.json` 节点状态与评分 → 更新 `STATE.md` → journal 追加 → 提交推送

### 每个角色收尾三件事（不可省）
1. 更新 `STATE.md`　2. journal 追加一行　3. `git add -A && git commit && git push`

---

## 2. 硬性规则

1. **只有本地机器 push**；服务器只 pull。实验产物经 `scripts/collect_node.sh` 回传后随本地提交入库
2. **大文件永不进 git**：数据集、checkpoint、完整日志只存服务器 `/data/dataset`、`/data/runs`
3. 任务认领原子化：改任务卡文件名即占有；互斥资源用 `mkdir locks/<name>` 抢锁，用完删除
4. `memory/hypotheses.jsonl` **只追加、永不改写**
5. 远程一切 >1 分钟的任务必须在 tmux 会话里，禁止裸 SSH 挂前台
6. 新代码必须先过 `--smoke` 再上真数据
7. 写代码前先读 `memory/failure_modes.md`；新踩坑后必须追加进去

## 3. 服务器速查

| 项 | 值 |
|---|---|
| 连接 | `ssh cac-server`（别名映射在本地 `~/.ssh/config`，轮换后自动更新） |
| 持久化 | 仅 `/data`：`repo/`(仓库)、`dataset/FSC147/`(VarV2 全套)、`runs/<ID>/`、`asset/`(杂物) |
| Python | `/data/miniconda/envs/cac/bin/python`（torch 2.10.0+cu128，CUDA 可用，RTX 3060 12GB） |
| 网络 | GitHub 直连可用；pip 必带 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 数据验证 | `/data/miniconda/envs/cac/bin/python scripts/check_data.py` 应全过 |

## 4. 服务器轮换演练（实例重租后地址密码全变时照此恢复）

1. 用户把新连接串+密码更新到本地 `~/cv_study/address_and_password.md`
2. 本地：`python3 ~/cac_explore/scripts/install_key.py`（装公钥、重写 ssh 别名）
3. 服务器：`ssh cac-server 'bash /data/repo/scripts/bootstrap_remote.sh'`（幂等；若 /data 被清则先重新 clone）
4. 数据集若丢：请用户重新上传 zip 至 `/data/dataset/` 后在 FSC147 目录内解压，跑 check_data 验证

## 5. 假设记录格式

```
IF [架构选择] IN [作用域], THEN [预测效果], BECAUSE [机制]. DISPROVED IF [证伪条件].
```

置信度更新（η=0.20，c∈[0.01,0.99]，初始 0.5）：支持 `c←c+0.20·w·(1−c)`；反驳 `c←c−0.20·w·c`。判定：confirmed >0.75 ／ refuted <0.25 ／ uncertain 其间。
