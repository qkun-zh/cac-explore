# STATE — Session 2026-08-28 (Lead=qkun-local)

**Mode**: 用户指导模式 (User-Guided)
**Preflight**: git init-sync ok · creds ROTATED (port 52331, new host) → re-ran `python3 scripts/install_key.py` · SERVER_OK
**Cleanup (session start)**: purged junk (OIR log, PROBE_dual_topo, G001, N0050, N0039_upcount_lite) out of `tree/nodes/` → archives; remote `tree/nodes` shrunk 59→30 (matches local); `/data/runs` pruned to active lineage + archive_2026-08-28.

## Champion (frozen regime)

**N0054_xscale_exemplar** (GCA + **XScale**, frozen DINOv3-ConvNeXt-Tiny) — **LOCKED DELIVERABLE**
- val MAE **19.647** / RMSE 74.05 · 31.32M total / ~3.4M trainable · 30ep @384 · fixed recipe (AdamW 1e-3, wd0.05, cosine, bs16, AMP)
- XScale = multi-scale (coarse+fine) ROI pooling → per-exemplar global summary additively fused into exemplar token pre-Condenser; ~0.1M. First pluggable part to BEAT GCA-only.
- Config: use_gca=True · use_ddca=False · use_xscale=True · xscale_size=3
- Confirmed optimum of the exemplar interface (see lineage): GCA-only, XScale, nothing else.

## Frozen-lineage record (what beat what, 30ep @384)

| Node | Config | MAE | Verdict |
|---|---|---|---|
| N0054 | GCA+XScale | **19.647** | ✅ CHAMPION (LOCKED) |
| N0055 | GCA+XScale+XKey (separate 2nd key) | 20.835 | ❌ exemplar-enrich hurts (+1.19) |
| N0051 | GCA-only | 20.599 | GCA genuine (~1.6) |
| N0053 | GCA+RGA (reg count aux) | 21.450 | ❌ density-bias aux hurts |
| N0056 | GCA+XScale+XFine (fused h2) | 24.313 (ES@17) | ❌ exemplar-enrich hurts (+3.06) |
| N0057 | GCA+XScale+Matcher (REPLACEMENT) | 21.076 | ❌ condenser cross-attn load-bearing (+1.43) |
| N0052 | GCA+DDCA (refactor) | 22.410 | ❌ DDCA harmful (+1.8, drop) |
| N0036 | GCA+DDCA (orig) | 20.49 | non-reproducible seed |

**Lesson (locked, fully triangulated)**: N0054 is the frozen-backbone optimum. All add-ons (DDCA, RGA, SALF, FILM, cross-attn, MoE, bg-token, XScale-Key, XFine) degrade; the one cleaner replacement (cosine-sim matcher) ALSO degrades — the learned cross-attn condenser is load-bearing. The exemplar-embedding interface + condenser attention form a tightly-matched pair under frozen 30ep. Structural head innovation exhausted in this regime.

## Server gotchas

- tmux at `/data/miniconda/bin/tmux` (libtinfo warning non-fatal) — launch via tmux not run_node.sh (bare tmux not on PATH)
- python `/data/miniconda/envs/cac/bin/python`; HF cache `/data/asset/hf` + `HF_ENDPOINT=https://hf-mirror.com`
- Remote sync via `scp` (never `git pull`) — proxy fails / untracked conflict
- `/data/runs` hygiene: keep active lineage, archive stale (see AGENTS §7)

## Queue (today)

1. ✅ **N0054 locked as deliverable (19.647).** Both exemplar-enrichment challengers refuted:
   N0055 XScale-Key (separate 2nd key) 20.835 (+1.19), N0056 XFine (fused h2) 24.313 ES@17 (+3.06).
   No further exemplar-side step is justified by evidence; the interface is exhausted at N0054.
2. ✅ Re-verified N0054 best.pth intact on server at `/data/runs/N0054_xscale_exemplar/best.pth`.
3. Remaining: final calibration table in latest synthesis mark; commit & push; session close.
