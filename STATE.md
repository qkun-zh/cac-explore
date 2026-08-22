# STATE — 当前现场

**阶段**: M2 首个真实节点完成 → 进入 M3 批量假设生成
**阻塞**: 无。服务器约 2 小时后回收，优先跑短任务。

## 已验证事实（勿再踩坑）
- torch==2.10.0+cu128 / torchvision 0.25.0 已装于 `cac` env，CUDA 可用（RTX 3060）
- FSC147 VarV2 就位于 `/data/dataset/FSC147`，check_data 全过（3659/1286/1190）
- 引擎契约：模型可输出低分辨率 density，engine 自动上采样+总和守恒；评估按密度和
- S0001_smoke: status=success, val MAE 46.69 @2ep/27s（真数据端到端验证通过）

## 下一步（按序）
1. Idea Agent 批量产出 S0002–S0005 假设节点（写 tree/nodes/*/idea.md + tasks/T*_pending_*.md）
2. Coding Agent 实现 model/config → 本地 --smoke 自检 → push
3. Executor tmux 跑真实 epoch（τ_max=30min 内）→ collect 回传 → 反馈四件套
4. Synthesis 落账 hypotheses.jsonl + 置信度更新

## 活跃任务
- T0001 S0001_smoke → **done**（result.json 已回传入库）
