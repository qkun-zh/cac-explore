# Idea — N0030_sizenorm (parent: N0027_norm_flip_swa, 20.403 @23.11M)

## 与 N0029_loghead 的区别（novelty stage-1 0.625 相似度澄清）
两者都动 head 输出端，但机制不同：N0029 是 **单调重参数化**（log/exp，无新信息源，
只改数值几何）；本节点是 **乘性尺寸条件化**（head 多出一条 `log_scale` 通道，
把 exemplar 框信息从 prompt 隐式注入升级为输出端显式调制——网络必须学会
"这块密度该整体放大还是缩小"的逐图决策）。N0029 无新参数语义，N0030 有。

## 单一改动（GOD v4 §6 种子2，V1 Spearman 0.465 支持）
用 3 个 exemplar 框的**平均尺寸**对 head 输出做逐图归一化：head 多输出一个标量通道
`log_size`，推理时 `dens_final = dens_raw * exp(log_size_pred)`，其中训练目标是
`log(mean_box_area)`。等价于把"物体尺寸"从隐式（网络猜）变成显式条件。

## 机制
H-A 说质量 ∝ 物体面积，但网络只能从特征隐式推尺寸——密集小物体时推不准。
显式喂尺寸（Fourier prompt 已有 area token，这里改成乘性输出端）让 head 幅度只负责
"单位面积的密度"，尺寸由框白拿，不再挤占幅度带宽。

## 预注册假设
**H0044**: IF head output is size-conditioned multiplicatively IN champion recipe THEN
val best MAE ≤19.6 BECAUSE amplitude bandwidth freed from encoding object scale.
DISPROVED IF MAE ≥20.4 OR exemplar-box mass variance across size-terciles doesn't shrink ≥30%.

## 数值细节
- `size_token = Fourier(w,h,area)` 已在 PromptEncoderV2；新增 head 分支 1ch → `log_scale`
- forward: `mass = dens_raw * torch.exp(torch.clamp(log_scale, -5, 5))`
- loss 不变（MSE on final density + count L1）

## Kill/confirm ladder
R0 smoke（scale 输出分布合理、无 nan）→ R1 40ep 主跑 → R2 归因臂（仅 scale vs 仅 base）

## 风险
模型可能把 log_scale 学成常数（退化回 parent）；若 mass-size 方差没缩，说明网络没用上它。
