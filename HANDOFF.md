# HANDOFF — 交接文档 (2026-09-22)

目标:把 FSC147 冻结头 counting 的 val EMA 从 **19.3431 (N0015_h0014)** 推到 **< 18**,全程:
冻结 backbone、头 ≤32M、seed 20260830、不算作弊、真实前沿背书、可发表创新
(test 数据已有 19.066,仅在收官论文时用;val 是对照货币)。

新服务器已在跑,GPU 空着,可直接开工。这文档是唯一交接物——先读 STATE.md + journal/events.jsonl + memory/hypotheses.jsonl 再动手。

---

## 1. 服务器(新,已配好 ssh)

```
ssh cac-server        # -> root@hzpcqeuyl8w9sljhsnow.deepln.com:52662
                      # 密码见 local/address_and_password.md (gitignored)
```
- RTX 3060 12GB;torch 2.10.0+cu128;python 用 `/data/miniconda/envs/cac/bin/python`
- `/data` 完整保留:工作副本 `/data/cac`(源码,用 tar-over-ssh 同步,无远程 git)、`/data/dataset/FSC147`、`/data/repro`(all 历史重训)、`/data/cdino_run`(CountingDINO 复现)、`/data/asset/hf`(DINOv3 ViT-S/16 缓存)
- HF 环境流:`export HF_HUB_OFFLINE=1 HF_HOME=/data/asset/hf`
- 同步命令范式:本地 `tar czf - --exclude 各类 | ssh cac-server 'tar xzf - -C /data/cac'`
- 跑训练:`setsid nohup ... > log 2>&1 </dev/null &`(无 tmux 可用)

## 2. 当前冠军与规程

- Live = `tree/N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014`(val 19.3431 / test 19.066)
- 全部机制在 `src/cac/expt/mechanisms.py`(单源):cellcal(w_c=+0.148,CONFIRMED 首个突破)、temp-pin 卫生(peakcal 已 dp 删除,机制 null)
- 运行范式:`scripts/run_node.py --parent <parent> --hyp <idea_FINAL.md> --set <overrides> --set futility_bar=<bar>`
- futility-stop 是 AGENTS 硬规则 16:`ep16 仅 WARN、ep24 可 HALT`;
- **AGENTS IRON RULE #0:GPU 在跑时绝不空等**——单次 ssh 轮询查看,禁 sleep 循环;有洞就立刻续新卡,否则算事故。
- v2 协议 = `configs/protocol_augment.toml`(augment=true, seed 20260830, 32ep/1800s, EMA eval)

## 3. 重要事实(接手前必须内化)

1. **确定性危机(最硬的事实)**:同 seed 同代码重跑结果不同,噪声地板 **±1–2 MAE**(CUDA atomic 非确定性 + fp 非结合率在混沌训练放大)。AGENTS 里"同 seed ±0.02"是错的,已作废。
   → 任何 ±1.5 以内的"改进"都不可信;大效应(崩/NaN/+2 以上)才可信。
   → **3-seed 多次复现迄今未做**(用户未批),是 precision 危机的唯一正解,也是挡在"继续增量"前的根。
2. **Hybrid 路线已判死(最新结论,勿再回头)**:本会话全 val 精确配对(N0015 vs CountingDINO ViT-S/16)后,
   所有确定性推理混合 router 全部变差:max +14.6, collapse-T 网格 +4~+15, ratio 网格最优 +4.1, blend +3~+12;oracle **−5.5**(上限)。
   根因:ViT-S aux 太弱(sparse 22.0),错误与 base **高度相关**(err-err pearson 0.566 / signed 0.553)——两个方法撞同一堵墙,无互补信号可白捡。
   router 脚本:`/data/cdino_run/hybrid_router.py`。
3. **共因诊断(定位根源)**:base 与 cdino 在同一 bin 同向失败——
   - `[0,50)` 都**过数**(base +3.5~-0.9 / cdino +14~+19 bias)
   - `[50,500)` 都**亏数**(base −10.6/−73.7 / cdino +14.9/−72.6)
   - `[500,∞)` 都**大幅亏数**(base −168/−721 / cdino −247/−663)
   → 根源 = **高密度区域的系统性欠数(collapse/normalization 失能)**,稀疏区过数来自共享朴实 bias。
   → 真正能推 <18 的不是第二个计数法,而是**提高 exemplar 到 dense-region 的信令**(解码端)。
4. **CountingDINO(reall 背书,WACV 2026, arXiv:2504.16570)已复现**:val MAE 39.70(用缓存 ViT-S/16,非论文 ViT-L;对灾难尾 8 图反杀 base 170–370)。原版 ViT-L/14 权重 1.2GB **在此网络下不 可 下载**,别在上面耗时间。复现目录 `/data/cdino_run`,已打 ids 补丁,val ids 已跑完(ids.npy 在)。
5. **H0010–H0021 除 cellcal 外全部 REFUTED**:verdict 细节在 memory/hypotheses.jsonl。readout 侧(exemplar→density 融合)已判**接近穷尽**;remaining 残血 idea = exemplar-distinctness(期望为负)。分支树:`local/research/h0014_branches.md`、`h0018_branches.md`。

## 4. 本地工件(会话后唯一权威副本)

- `/home/qkun/cac_backup/`:N0015_best.pth(125MB)、N0015 val/test perimage(有 `ids` 字段)、baseline_v2_val_perimage.json、`cdino_results/`(predictions/targets)、`cdino_ids.npy`(与其配对)
- `local/research/`:paper_spine/methods(论文骨架)、各 h00XX_idea_FINAL(下一卡设计模板)、survey_mechanisms/runaway_meta(first-principles 级分析)
- 很干净:所有 `__pycache__` 已清;git 在远端 `main`,最新 commit 含本会话 journal

## 5. 自主执行的建议次序(别停下来问)

G与空闲 GPU 在手时,按此推进(此基线已获用户授意"治本、不必汇报、自主行动"):

1. **治本 #1 = precision**:解 `futility`/seed 工期约束跑 **N0015 3 组复现**(同 cfg 不同 seed)标定噪声 ≥3 次独立估 —— 这是下面一切判断的乘数。产出:阈值"真实改进 vs 噪声"从猜测变实测。
2. **定位 dense 欠数的机械来源**(诊断,不烧卡):在 N0015 上对 dense 尾(gt>300)做逐成分归因——v1 的 perimage 工具 + `local/research/perimage_diagnostic.md`。查三点:(a)回归头饱和/残差 block 容量;(b)exemplar 全局池对 dense 局部密度的信令稀释;(c)boosting 归一化(求和→密度→回乘)在 dense 的静默崩。
3. 依据归因,出 **H0022 卡**(或直写新 idea_FINAL,沿用现有 authoring 流程):e.g. dense-aware readout(局域密度提示 + 计数感知归一化)、或 exemplar-distinctness 变体在 dense 的靶向版。过 novelty_check + conformance,进队列。
4. 每张卡必带 futility_bar;verdict 用 `eval_test.py --set` 全切片出,记录比对 baseline 各 bin(sparse/mid/dense)。
5. 若 3 张卡仍无 >1.5 净降,**转论文收官**(numbers + 语料已是现货)。paper 各骨架文件在 local/research/ 下,直接续写。

## 6. 戒律

- 别做 test 优先实验;val 是对照货币
- 别碰"第二个计数法/大模型换权"的路子(hybrid 已判死,ViT-L 下不了)
- 每个 train 必须带 futility_bar,ep24 无净降就 HALT 省时
- 所有新文档:继承现模板(header 写 id/父/机理/制约),journal 记 booking→verdict
- 沟通:用中文,结论先行;自主推进,不必每步汇报