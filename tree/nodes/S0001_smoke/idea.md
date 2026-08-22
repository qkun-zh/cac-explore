# 标题：Tiny Exemplar Density Baseline（冒烟测试）

## 动机与直觉
验证框架全链路（契约、tmux、采集、提交）用的最小节点。一个纯卷积密度头，不追求精度。

## 架构规格
- core_ideas: 4 层 conv 下采样到 1/8 分辨率，输出单通道密度图，softplus 保证非负
- core_blocks: Conv3x3-BN-ReLU ×2 → Conv3x3-s1 → 1x1 输出
- network_structure: 输入 [B,3,S,S] + bbox（本节点忽略 exemplar 内容，仅占位）→ 密度 [B,1,S/8,S/8]
- tunable_aspects: 通道数(16)、深度
- invariants: 参数 <0.5M；CPU 可跑；无外部权重

## 提出的假设
- IF 用 softplus 而非 relu 做密度输出 IN 小型密度网络, THEN 训练更平稳无死区, BECAUSE 零梯度死单元在极小网络上影响占比更大. DISPROVED IF 冒烟中 loss 出现 NaN 或不下降。

## 与父节点的差异
根节点（无父）。结果仅用于框架验收，不计入轨迹树评分。

## 新颖性声明
零新颖性——刻意保守。
