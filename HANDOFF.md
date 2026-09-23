# HANDOFF — Handoff document

> **Current focus (from 2026-09-22) = training-free `tree/N0029_tf_champion`**  
> **Latest handoff (progress / scores / eval gate / cleanup): `tree/N0029_tf_champion/HANDOFF_TF.md`**  
> Authoritative constraints: same directory `constraints_and_goal.md` · scoreboard: `baseline_board.md`  
> Old N0001 training tree sealed → `archive_local/README_ARCHIVED.md`  
> Everything below = historical training track (N0001/N0015), archive only, **not the current default task**.

---

## [Historical] Goal (pre-2026-09-22 training track)

Goal: push FSC147 frozen-head counting val EMA from **19.3431 (N0015_h0014)** to **< 18**, throughout:
frozen backbone, head ≤32M, seed 20260830, no cheating, real frontier backing, publishable novelty
(test already at 19.066, use only for the final paper; val is the comparison currency).

New server is already running, GPU idle, ready to start. This document is the only handoff artifact —
read STATE.md + journal/events.jsonl + memory/hypotheses.jsonl before acting.

---

## 1. Server (new, ssh configured)

```
ssh cac-server        # -> root@hzpcqeuyl8w9sljhsnow.deepln.com:52662
                      # password: see local/address_and_password.md (gitignored)
```
- RTX 3060 12GB; torch 2.10.0+cu128; python = `/data/miniconda/envs/cac/bin/python`
- `/data` fully preserved: working copy `/data/cac` (source, synced via tar-over-ssh, no remote git),
  `/data/dataset/FSC147`, `/data/repro` (all historical retrains), `/data/cdino_run` (CountingDINO repro),
  `/data/asset/hf` (DINOv3 ViT-S/16 cache)
- HF env: `export HF_HUB_OFFLINE=1 HF_HOME=/data/asset/hf`
- Sync pattern: local `tar czf - --exclude <various> | ssh cac-server 'tar xzf - -C /data/cac'`
- Launch training: `setsid nohup ... > log 2>&1 </dev/null &` (tmux unavailable)

## 2. Current champion and protocol

- Live = `tree/N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014` (val 19.3431 / test 19.066)
- All mechanisms in `src/cac/expt/mechanisms.py` (single source): cellcal(w_c=+0.148, first CONFIRMED breakthrough),
  temp-pin hygiene (peakcal already deleted via dp; mechanism null)
- Run pattern: `scripts/run_node.py --parent <parent> --hyp <idea_FINAL.md> --set <overrides> --set futility_bar=<bar>`
- futility-stop is AGENTS hard rule 16: `ep16 WARN only, ep24 may HALT`
- **AGENTS IRON RULE #0: never idle while the GPU runs** — single ssh grep poll, no sleep loops;
  if there is a gap, start the next card immediately, otherwise it is an incident.
- v2 protocol = `configs/protocol_augment.toml` (augment=true, seed 20260830, 32ep/1800s, EMA eval)

## 3. Critical facts (internalize before taking over)

1. **Determinism crisis (hardest fact):** same seed + same code reruns differ; noise floor **±1–2 MAE**
   (CUDA atomic nondeterminism + fp non-associativity amplified in chaotic training). The AGENTS claim
   "same seed ±0.02" is wrong and void.
   → any "improvement" within ±1.5 is untrustworthy; only large effects (crash/NaN/+2 or more) are credible.
   → **3-seed multi-replication never done** (user not approved); only real fix for the precision crisis
   and the root blocker before further incremental gains.
2. **Hybrid route dead (latest conclusion, do not revisit):** full-val paired comparison this session
   (N0015 vs CountingDINO ViT-S/16): all deterministic inference hybrid routers worse — max +14.6,
   collapse-T grid +4~+15, ratio grid best +4.1, blend +3~+12; oracle **−5.5** (upper bound).
   Root cause: ViT-S aux too weak (sparse 22.0), errors highly **correlated** with base
   (err-err pearson 0.566 / signed 0.553) — both methods hit the same wall, no complementary signal to free-lunch.
   Router script: `/data/cdino_run/hybrid_router.py`.
3. **Common-cause diagnosis (root located):** base and cdino fail in the same direction in the same bin —
   - `[0,50)` both **overcount** (base +3.5~-0.9 / cdino +14~+19 bias)
   - `[50,500)` both **undercount** (base −10.6/−73.7 / cdino +14.9/−72.6)
   - `[500,∞)` both **severely undercount** (base −168/−721 / cdino −247/−663)
   → root = **systematic undercount in high-density regions (collapse/normalization failure)**;
   sparse overcount comes from shared naive bias.
   → what can actually push <18 is not a second counting method but **stronger exemplar→dense-region signaling** (decoder side).
4. **CountingDINO (peer-endorsed, WACV 2026, arXiv:2504.16570) reproduced:** val MAE 39.70 (cached ViT-S/16, not paper ViT-L;
   beats base by 170–370 on the 8 disaster-tail images). Original ViT-L/14 weights 1.2GB **cannot be downloaded on this network** — do not waste time.
   Repro dir `/data/cdino_run`, ids patch applied, full val ids already run (ids.npy present).
5. **H0010–H0021 all REFUTED except cellcal:** verdicts in memory/hypotheses.jsonl. Readout side
   (exemplar→density fusion) judged **near-exhausted**; remaining residual idea = exemplar-distinctness (expected negative).
   Branch trees: `local/research/h0014_branches.md`, `h0018_branches.md`.

## 4. Local artifacts (sole authoritative copies after the session)

- `/home/qkun/cac_backup/`: N0015_best.pth (125MB), N0015 val/test perimage (with `ids` field),
  baseline_v2_val_perimage.json, `cdino_results/` (predictions/targets), `cdino_ids.npy` (paired)
- `local/research/`: paper_spine/methods (paper skeleton), each h00XX_idea_FINAL (next-card design templates),
  survey_mechanisms/runaway_meta (first-principles-level analysis)
- Clean: all `__pycache__` cleared; git on remote `main`, latest commit includes this session's journal

## 5. Suggested autonomous execution order (do not stop to ask)

With a free GPU in hand, proceed in this order (this baseline was authorized by the user as
"fix the root cause, no need to report, act autonomously"):

1. **Root fix #1 = precision:** under the `futility`/seed schedule constraint, run **N0015 3-group replication**
   (same cfg, different seeds) to calibrate noise with ≥3 independent estimates — this is the multiplier
   for every judgment below. Output: threshold "true improvement vs noise" goes from guess to measurement.
2. **Locate mechanical source of dense undercount** (diagnostic, no GPU burn): on N0015, component-wise
   attribution on the dense tail (gt>300) — v1 perimage tools + `local/research/perimage_diagnostic.md`.
   Check three points: (a) regression head saturation / residual block capacity;
   (b) exemplar global pooling diluting dense local density signaling;
   (c) boosting normalization (sum→density→multiply-back) silent collapse on dense.
3. Based on attribution, issue **H0022 card** (or write a new idea_FINAL directly, using the existing authoring flow):
   e.g. dense-aware readout (local density hint + count-aware normalization), or dense-targeted exemplar-distinctness variant.
   Pass novelty_check + conformance, enter queue.
4. Every card must carry futility_bar; verdict via `eval_test.py --set` full slices, compare against baseline bins (sparse/mid/dense).
5. If 3 cards still show no >1.5 net drop, **switch to paper wrap-up** (numbers + corpus already in hand).
   Paper skeleton files are under local/research/ — continue writing directly.

## 6. Commandments

- No test-first experiments; val is the comparison currency
- Do not touch "second counting method / large-model weight swap" (hybrid dead, ViT-L cannot download)
- Every train must carry futility_bar; HALT at ep24 if no net drop, to save time
- All new documents: inherit the existing template (header with id/parent/mechanism/constraints); journal records booking→verdict
- Communication: conclusion first; advance autonomously, no need to report every step

## 2026-09-22 cleanup
- Active tree: `tree/N0029_tf_champion` (TF, subset286 locked).
- Old N0001 sealed; see `archive_local/README_ARCHIVED.md`.

## 2026-09-23 update
- Rapid success bar (user): **subset286 MAE ≤ 20.0** for fast iteration; champion/board rows remain full val 1286 only.
- Documentation language (user): **English only** — no Chinese in any project doc (`constraints_and_goal.md` §5b).
- Current server: see `local/address_and_password.md` (host/port rotated 2026-09-23).
