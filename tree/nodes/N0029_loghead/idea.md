# Idea — N0029_loghead (parent: N0027_norm_flip_swa, 20.403 @23.11M)

## 与 N0030_sizenorm 的区别（novelty stage-1 0.625 相似度澄清）
两者都动 head 输出端，但机制不同：本节点是 **单调重参数化**（log/exp，无新信息源，
只改数值几何——把"学绝对幅度"变成"学数量级"，天花板从输出上限变为 float 上限）；
N0030 是 **乘性尺寸条件化**（新增 log_scale 通道，注入框尺寸这一外部信息源）。
N0029 改变的是损失的等高线形状，N0030 增加的是条件变量。

## 单一改动（GOD v4 §6 种子1，V1 实证支持）
把密度头输出从线性幅度改成 **log 空间**：`density = exp(head_out)`，训练目标同步
`log(1+dens)`。其余与 parent 完全一致（partial-FT blocks10-11@lr×0.1、ImageNet norm、flip、40ep、bs8、lr1e-3 cosine）。

## 机制（H-A，V1 Spearman 0.465 已证）
冠军的 head 输出幅度在线性空间被 MSE 拉向"平均物体尺寸"的常数（a*≈0.055h），
遇到密集小物体就系统性低估（压缩比 0.96→2.22 单调恶化）。log 参数化让网络预测的是
数量级而非绝对值：密集区要学的只是 log 增量，天花板从"输出上限"变成"float 上限"。

## 预注册假设
**H0043**: IF density head outputs in log-space IN champion recipe THEN val best MAE ≤19.4
BECAUSE dense-region amplitude no longer saturates (log removes multiplicative ceiling).
DISPROVED IF MAE ≥20.4 OR tail[500+) bucket error worsens >+50 vs parent.

## 数值稳定性
- forward: `dens = torch.exp(torch.clamp(raw, -10, 10))` 防 inf/nan
- target: `log1p(dens)`，loss = MSE(log-space) + count L1（count 从 exp 后积分）
- eval: 与现引擎一致，直接对 dens 积分

## Kill/confirm ladder
R0 smoke（数值范围 + 反传不 nan）→ R1 40ep 主跑 → R2 仅当赢：log-only vs linear-only 归因臂

## 风险
log 放大小密度误差 → bulk 可能劣化；exp 溢出已 clamp；若 bulk 劣化>0.5 且尾部无改善即杀。
