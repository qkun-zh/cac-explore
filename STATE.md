# STATE — Session 2026-08-29 (Lead=qkun-local)

**Mode**: Free-Research (autonomous; user: "你是自主模式，不必请求我的意见")
**Preflight**: creds ROTATED (port 44387, new host gxkkqyad0izmmwnlsnow.deepln.com) → reran `python3 scripts/install_key.py` → SERVER_OK. git up to date. Server python/HF/tmux gotchas below unchanged.

## Session directive (new generation)
Combine the two papers' pluggable attention/module designs into the frozen N0054 champion:
- **PoM** (arXiv:2604.06129, CVPR-F'26): polynomial mixer — `H=[Σ_p α_p ⊙ h(W_h X)^p]·1` + gate `σ(W_s X)⊙H`; linear, permutation-equivariant, contextual-mapping. D=2d,k=2 enough; **hybrids (attention+PoM) best**.
- **ParTY** (arXiv:2603.09611, CVPR'26): part-guidance — separate part tokens → fused Part-Guidance conditions the holistic branch; part-aware gating; holistic-part cross-attn fusion.

**Lead integration decision (3 parallel lens idea agents)**: counter-intuitive (a)-KV-hybrid rejected (semantics collapse to global mean — fails the exemplar interface), (b) bias rejected (query-side modulation = SALF/FILM family), (d) gap² into GCA weak. Champion-lineage PMOM accepted: change ONLY the exemplar **aggregation operator** (→**not** the info-add axis N0055/N0056, **not** the consumer-swap axis N0057).

## Champion (frozen regime, UNCHANGED)
**N0054_xscale_exemplar** (GCA + XScale) val MAE **19.647** / RMSE 74.05 · 31.32M · LOCKED DELIVERABLE. Recipe: AdamW 1e-3, wd0.05, cosine, bs16, AMP, 30ep @384, MSE+SmoothL1, augment. use_gca=True, use_ddca=False, use_xscale=True, xscale_size=3.

## Active node
**N0058_pompart_exemplar** (parent N0054, launched): PMOM = 2×2 part-pool → 2nd-order polynomial moments `m=[h;h²]` → learned part-gate softmax → moment_proj → single fused prototype (B,K,256) preserved; condenser MHA, XScale, GCA untouched. Novelty GATE passed (0.264, stage-2 NOVEL — swap-producer vs N0057 swap-consumer). H0080. smoke GREEN (real backbone): use_pmom=True 29.86M / trainable 2.04M; use_pmom=False 31.32M/3.50M (exact N0054 → single-switch proven). Train E1 32.484 @101s/ep.

**H0080 barrier**: CONFIRM best val MAE < 19.647 · WEAK-KEEP ≤20.0 · KILL ≥20.4 or ep16+ train ≥ +1.5 over 19.647 (i.e. ≥21.147).

## Server gotchas
- tmux at `/data/miniconda/bin/tmux` (libtinfo warning non-fatal) — export PATH inside command.
- python `/data/miniconda/envs/cac/bin/python`; HF cache `/data/asset/hf` + `HF_ENDPOINT=https://hf-mirror.com`.
- Remote sync via `scp` (git pull flaky). `/data/runs` hygiene: keep lineage, archive stale.

## Queue
1. ✅ N0058 novelty gate + smoke green (29.86M/2.04M, single-switch exact).
2. 🔄 N0058 training running (~50min @30ep). Poll single ssh; early-stop bar 21.147 @ep16+.
3. ⏸ On done: feedback ×3 + synthesis + calibration table + journal + commit.