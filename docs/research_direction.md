# 研究方向（人类指定）

## 方向声明

在 FSC147 类无关计数基准上，通过假设驱动的多智能体进化搜索，发现轻量、准确、可迁移的 CAC 计数架构。

## 任务定义

- 输入：RGB 图像 + 示例框（exemplar box）标注目标类别
- 输出：密度图，其积分即预测计数
- 指标：MAE（主）、RMSE（次）；训练/验证/测试划分沿用 FSC147 官方 pkl

## 数据集布局（服务器 `/data/dataset/FSC147`，VarV2 协议）

```
FSC147/
├── images_384_VarV2/<im_id>.jpg            # 长边~384 可变宽高比图像
├── gt_density_map_adaptive_384_VarV2/<im_id>.npy  # 预计算自适应密度图（与图同尺寸）
├── annotation_FSC147_384.json              # exemplar boxes: {"<id>": {"box":[x1,y1,x2,y2]}}
└── Train_Test_Val_FSC_147.json             # {"train":[...],"val":[...],"test":[...]} 6135 图/147 类
```

加载器 `code/data/fsc147.py` 已按此布局实现；密度重采样采用总和守恒（计数严格不变）。
来源：用户本地官方打包上传（scp），内容同 HF 镜像 isentropic/FSC147。

## 约束

- 参数量预算与训练时长上限写入各节点 `config.py`；默认墙钟 ≤30 分钟
- 单卡 RTX 3060 12GB，AMP 混合精度
- 架构必须暴露 `build_model(cfg)`，输入 `[B,3,H,W]` + exemplar

## 基线参考（来自前期 cv_study 工作）

DViT-Light：392×392 输入、grid 48 密度图、MSE+0.3·L1 计数损失、AdamW 1e-3、cosine 150ep。
根节点可从类似配置出发，也可完全另起炉灶——由 Idea Agent 自主判断。
