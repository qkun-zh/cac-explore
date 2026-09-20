# N0007_h0006 — idea (Idea Agent, H0006 `use_h2pool`)

- **Node:** N0007_h0006 · **Parent:** N0002_h0001 (live parent 22.5641 @ep30) · **Booked hypothesis:** H0006 · **Switch:** `use_h2pool` (child `true`; parent default `false`) · **Composition:** SOLO single-switch.
- **Falsifier bar:** final EMA val MAE ≤ 22.26 at seed 20260830 under the canonical 32ep/1800s protocol (live parent 22.5641; confirm iff ≥0.30 lower) AND gt>500 dense-image mean |Δ| below the parent 511.7 baseline.
- **Booked hypothesis text (verbatim — do NOT edit):** see the single runner-parseable line in §5 (kept exactly once per STATE gotcha 2).

---

## 1. Web-survey note (AGENTS §4.3 / rule 15)

Live fetch 2026-09-20 re-verified the load-bearing Direction-1 precedent; the other two refs are inherited from the completed Researcher report `local/research/fallback_verify_next.md` Direction 1 (no new fetches needed for settled rows):

- **LOCA, iterative prototype adaptation (OPE) — pooling neglects shape, adaptation fixes it** · https://github.com/djukicn/loca (ICCV 2023, live-verified 2026-09-20: repo + paper PDF reachable, FSC147 few-shot, no-OPE ablation val 10.24→16.24) · **confidence H.** Core quote for this card: *"Existing methods extract queries by feature pooling which neglects the shape information … and leads to a reduced object localization accuracy"* — LOCA's OPE fuses exemplar shape/appearance with image features instead of widening the key set blindly. Lesson booked for H0006: adapt/pool at the KEY side with finer spatial support (per-exemplar h2-pooled key, 3 keys), never a wider token-max bank.
- **CountingDINO, training-free normalized similarity on frozen DINO** · https://arxiv.org/abs/2504.16570 · repo https://github.com/lorebianchi98/CountingDINO · **confidence H** (from fallback report §Direction 1 Ref B). Normalized cosine maps over frozen DINO geometry carry signal with zero training (val 25.48 / test 20.93); its stated risk ("small FSC147 objects → 1/16 ROI prototype noisy") is exactly the pooled-bank failure H0006 attacks, and its fix (multiscale/filter, not more keys) bounds our bank at 3 keys.
- **SAFECount, score-map normalization across exemplar AND spatial dims** · https://arxiv.org/abs/2201.08959 · repo https://github.com/zhiyuanyou/SAFECount · **confidence H** (fallback report Ref C). Only the normalization half is reused (softmax-over-K consensus); the FE similarity-weighted-enhancement half is the banned pre-Condenser gating family and is explicitly NOT reused.
- **Contrast (what H0006 is not):** H0004 `use_simbank` (75-key token-max, full-grid einsum — timeout autopsy below); N0004 `use_padapt` (unmodulated cross-attention re-encode of K/V — refuted, corrupts Condenser geometry); any second full-res qproj (the exact cost that killed H0004).

## 2. Multi-angle reasoning

### (a) Pure-mathematics lens — why a pooled h3 key is a background average, and what the h2 pool computes instead

For a small exemplar (e.g. ~16×16 px at 384 input), the h3@1/16 ROI is ~1–2 distinct feature cells. `ExemplarEncoder` bilinearly re-samples that patch to 7×7 = 49 tokens and softmax-pools them to one vector (`model.py:97-103`): `e_k = Σ_j α_j t_j` where ~47 of the 49 `t_j` are interpolants of the same 1–2 cells plus surrounding background. The pooled key is therefore a convex combination dominated by background direction; `cos(q, e_k)` compresses toward the background baseline and the `[top1, consensus]` evidence pair goes flat exactly on the small instances the dense tail is made of. No temperature or calibration can recover a margin that pooling destroyed — the signal is rank-collapsed before the readout.

The h2 pool computes a different object at the key side while leaving the trusted readout untouched: at 1/8 res the same exemplar covers ~2×2 = 4 real cells, and a 3×3 ROI-align + attention-pool (`a = softmax(Linear(tok))`, `key = Σ a_j t_j` over 9 tokens) is a learnable selector that can place nearly all mass on object cells — the same functional form as the lineage-proven ExemplarEncoder pooling (lines 102–103), one octave finer. The readout stays a 3-key softmax-over-K comparison with a learned scalar temperature, i.e. the identical comparison dynamics the optimizer already tuned on the parent (simprior temp 0.07→0.0393, out-proj norm 0.5874 — N0002 confirmed). Attribution is therefore clean by construction: readout fixed in kind, bank changed in resolution, so any same-seed delta vs the parent isolates bank resolution. Computing similarity at 48×48 then bilinearly upsampling the 64-ch residual is valid because the evidence is conditioning, not the density field itself — the decoder still emits at 96×96.

### (b) Champion-lineage lens — file:line facts, N0006 failure, H0004 timeout autopsy

- **What h2 is and where it is available (`model.py`):** `Backbone.forward_feature_map` returns `[hs(2), hs(3)]` under `torch.no_grad` (lines 40–43) — frozen, read-only tensors. `CountingHead.forward(self, h2, h3, bboxes_in)` (line 161) already receives h2 in scope: `FineFuser` consumes it (`self.lat(h2)`, line 69) at 48×48×192 (`backbone_dims=(192,384)`, `d_fine=128`, fine emitted at 96×96 after two ×2 lifts, lines 67–74). `ExemplarEncoder` reads **only h3** (line 166: `self.exemplar(h3, …)`). So h2@1/8 (B,192,48,48) is available read-only at the exact attachment point with zero signature changes to backbone/encoder/fuser.
- **Simprior is confirmed load-bearing (H0001, N0002 22.5641 @ep30, margin 0.729 = 2.43× the bar):** the 3-key K-normalized readout works — its bank is what is coarse. H0006 reuses that readout verbatim (same qproj weights, same `[top1, consensus]` pair, same zero-init add) and changes only the keys.
- **N0006 `use_verify` REFUTED used-and-harmful (synthesis §1–2: 23.5802 @ep31, +1.016 vs parent, prop2/ver2 norms off init, temp ÷3.3):** the gate engaged and steered `cond_map` with corrupt evidence. Synthesis §4 causal ordering binds this card: *"pooled-`e` degenerate for small exemplars (h3@1/16 ROI 1–2 cells pooled to one vector — the evidence the gate consumed was coarse before temperature ever acted) → finer bank first."* H0006 is that first branch, booked as NEW `use_h2pool` (Booking 1), hires deferred, no contradicts-retry.
- **H0004 `use_simbank` timeout autopsy (fallback report + STATE):** r=5 75-key token-max bank, best MAE 48.12, 1815.2 s vs parent 1755 s (~45 s headroom gone). Failure decomposition: (i) a **second qproj at 96×96** (~75M MACs, doubling simprior's dominant cost) plus a 75-key full-grid einsum and a (B,75,96,96) activation (~11M elems, ~44 MB fp32 @bs16) blew the time/memory budget so `_BudgetStop` froze learning at an unrecovered point — uninterpretable, not a verdict; (ii) max-over-75 over noisy 1/8 part tokens with one shared temperature is a high-variance readout (early-gradient thrash). H0006 is NOT a retry: 3 keys not 75, no second qproj (reuses simprior's), similarity at 48×48 (¼ the einsum/activation), r=3 pooled readout — distinct mechanism with the new dual falsifier, booked as a fresh id per synthesis §3 dedup.
- **N0004 `use_padapt` boundary (refuted +3.00):** unmodulated cross-attention over h2 tokens corrupted Condenser K/V. H0006 never touches `e`, `fine`, or the Condenser path — it reads h2 through its own ROI op and adds a residual post-Condenser. No mechanism overlap.

### (c) Counter-intuitive / low-cost lens

- **Counter-intuitive:** the head already has a matcher (Condenser) plus a confirmed prior (SimPrior), so a *second* similarity bank looks redundant. It is not a second matcher — it is a second BANK for the same matcher form: the query projection, the K-normalization, the evidence pair, and the zero-init trust mechanism are all reused/shared, and only the key-side spatial support changes (h3-pooled → h2-pooled). Two banks × one readout form is precisely what isolates "resolution of the prototype" as the single variable under test.
- **Low-cost:** Δ = 12,674 params (~0.04% of the 32M cap, ~2% of the ~630k headroom), ≈0.02 GFLOPs/forward (≈19.6M MACs, quarter of H0004's fatal cost, <0.5 s/epoch vs ~45 s headroom — no `_BudgetStop` risk), one extra (B,3,48,48) similarity volume (~2.8 MB fp32 @bs16 vs H0004's 44 MB). Zero-init `out` makes step-0 forward bit-identical to the parent; flag-off reproduces N0002 byte-for-byte (safe ablation).
- **Cheap falsification with a directed redirect:** one canonical run decides it. Clean null (`out` norm ≈ 0, or miss with the h2 branch engaged) means pooled resolution was not the bottleneck → redirect to Direction 2 (`use_simcal`, anisotropy calibration), never back to 75-key max (§11: no silent retries).

## 3. Exact implementation spec for the Coding Agent (ONE targeted change: `use_h2pool`)

**New module class** (name fixed): `H2Pool(nn.Module)` — per-exemplar h2 attention-pooled 3-key bank with simprior-style readout at 48×48.

```python
class H2Pool(nn.Module):
    def __init__(self, d_h2=192, d_proj=64, cond_dim=64, roi=3, n_ev=2):
        super().__init__()
        self.r = roi
        self.pool_attn = nn.Linear(d_h2, 1)          # 192 -> 1, attention-pool over 9 tokens
        self.kproj = nn.Linear(d_h2, d_proj, bias=False)  # 192 -> 64, NEW (only new projection)
        self.temp = nn.Parameter(torch.tensor(0.07))      # own learnable temperature (divisor)
        self.out = nn.Conv2d(n_ev, cond_dim, 1)           # 2 -> 64, ZERO-INIT (see below)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, h2, fine_down, bboxes_in, S, qproj):
        # h2: (B,192,48,48) frozen read-only; fine_down: (B,128,48,48); qproj: simprior.qproj (shared, no copy)
        B, _, H2, W2 = h2.shape
        K = bboxes_in.shape[1]
        s2 = W2 / float(S)
        idx = torch.arange(B, device=bboxes_in.device, dtype=bboxes_in.dtype).view(B, 1, 1).expand(B, K, 1)
        rois = torch.cat([idx, bboxes_in * s2], -1).reshape(B * K, 5)
        patch = roi_align(h2, rois, output_size=(self.r, self.r))        # (B*K,192,3,3)
        tok = patch.flatten(2).transpose(1, 2)                          # (B*K,9,192)
        a = self.pool_attn(tok).softmax(1)                              # (B*K,9,1)
        key = (tok * a).sum(1).view(B, K, -1)                           # (B,K,192)
        q = F.normalize(qproj(fine_down), dim=1)                        # (B,64,48,48), SHARED weights
        p = F.normalize(self.kproj(key), dim=-1)                        # (B,K,64)
        Sv = torch.einsum('bchw,bkc->bkhw', q, p)                       # (B,K,48,48)
        W = (Sv / self.temp.clamp_min(1e-3)).softmax(dim=1)             # softmax over K=3
        top1 = Sv.max(dim=1, keepdim=True).values                       # (B,1,48,48)
        cons = (W * Sv).sum(dim=1, keepdim=True)                        # (B,1,48,48)
        ev = torch.cat([top1, cons], dim=1)                             # (B,2,48,48)
        return self.out(ev)                                             # (B,64,48,48), all-zeros at init
```

**Constructor order (append-only RNG, AGENTS §5 rule 13):** construct `H2Pool` in `Counter.__init__` **after** the `self.head.simprior = SimPrior(…)` assignment (i.e., after every parent module AND after SimPrior), attached as `self.head.h2pool`. It must be the last `nn.Module` construction in `__init__` so all parent + SimPrior init draws keep their exact RNG positions. Only `H2Pool`'s own init draws (`pool_attn`/`kproj` default init) are appended.

**Exact attachment point and tensor ops** (in `CountingHead.forward`, where `h2` is already in scope; `S = self.S`):
- Non-module flag on the head next to `use_simprior` (line 159 style): `self.use_h2pool = _get(cfg, "use_h2pool", False)` — no RNG consumption.
- After the existing simprior block (lines 169–172), add:
  `if self.use_h2pool:`
  `    fine_down = F.adaptive_avg_pool2d(fine, (48, 48))`
  `    d48 = self.h2pool(h2, fine_down, bboxes_in, self.S, self.simprior.qproj)`
  `    cond_map = cond_map + F.interpolate(d48, size=(Hf, Wf), mode="bilinear", align_corners=False)`
- `h2` is read-only: it arrives from `forward_feature_map` under `torch.no_grad` (lines 40–43) and is never written, gated, or rescaled — ROI-align reads only. `fine`, `e`, Condenser, SimPrior, GCA, XScale paths untouched. Flag off → the three added lines are skipped and forward is the parent's byte-for-byte.
- **No second full-res qproj:** the query side calls the EXISTING `self.simprior.qproj` on the 48×48-downsampled fine (shared weights, no new Conv2d) — this is the "reuse simprior q-projection style" requirement and the exact cost H0004 died on. The only new projection is `kproj` (192→64) on the key side.
- **Explicit non-violation statement:** decoder `in_ch` stays 192 (pure additive residual into `cond_map`); no parent module shape/init touched; `e`/`fine` never gated/rescaled → NOT pre-condenser exemplar gating at any granularity (§11); backbone stays frozen (`requires_grad_(False)`, `net.eval()`); contract `build_model(cfg) → forward → {"density",…}` unchanged.

**Zero-init plan:** `self.out` weight+bias exact zeros → at step 0 `d48 == 0` regardless of `pool_attn`/`kproj`/`temp`, so child forward output is numerically identical to the parent at init (same pattern as SimPrior `out`, lines 208–210, and GCA, lines 183–184). `pool_attn`/`kproj` keep default init (gated by the zero proj).

**Config key and default:** `use_h2pool = false` in `config.toml` (mirroring `use_simprior` style, read via `_get(cfg, "use_h2pool", False)`). Child sets `use_h2pool = true`; parent config untouched.

**Param delta arithmetic** (all exact):
- `pool_attn` Linear(192→1, with bias): 192·1 + 1 = 193
- `kproj` Linear(192→64, no bias): 192·64 = 12,288
- `temp`: 1
- `out` Conv2d(2→64, k1, with bias): 2·64 + 64 = 192
- **Total Δ = 193 + 12,288 + 1 + 192 = 12,674 (≈12.7k; ≈12.5k booked nominal).** Node total ≈ 31.37M + 0.0127M ≈ **31.38M ≤ 32M** (`max_params_M = 32`), ~620k headroom remaining. No other module changes size.

**Expected FLOPs change** (per 384px image, forward): shared-qproj at 48² 128·64·2304 ≈ 18.9M MACs; ROI-align 3×3×3 + 9-token pool negligible; einsum 3·64·2304 ≈ 0.44M; softmax-over-K negligible; out-proj at 48² 2·64·2304 ≈ 0.29M; bilinear 48→96 negligible. **Total ≈ 19.6M MACs (≈0.02 GFLOPs)** — ~¼ of H0004's fatal full-grid cost; expected epoch impact <0.5 s of the ~45 s headroom (parent 1755 s vs τ_max 1800 s). Activation +1 (B,3,48,48) volume ≈ 0.7M elems (~2.8 MB fp32 @bs16) vs H0004's ~44 MB. No `_BudgetStop` risk.

**RNG-order statement:** the only new RNG consumption is `H2Pool` parameter init, appended strictly after all parent + SimPrior init draws; forward has no stochastic ops (no dropout; pool-softmax/einsum/norm/interpolate deterministic); data seeding (`seed=20260830`, `augment=false`) unchanged. Parent-vs-child divergence comes only from the learned h2 bank (§5 rule 13 paired-contrast semantics, precision ~0.02).

**Register compliance (AGENTS §11):** no DDCA branch (`use_ddca=false` stays); no extra spatial summaries/RGA; no final-layer readout (hs taps unchanged); no unfreeze; no pre-condenser exemplar gating in any granularity (no channel/per-exemplar scalar gate on `e`/`fine` — additive post-Condenser residual only). GCA/XScale/hs(2,3)/SimPrior intact; only `out["density"]` feeds the loss.

## 4. Falsification reading

- **Confirm/refute rule:** canonical run at seed 20260830, 32ep/1800s, EMA eval, same-seed paired contrast vs live parent 22.5641. **Confirm** H0006 iff final EMA val MAE ≤ 22.26 AND gt>500 mean |Δ| < 511.7. Otherwise **refute** — including flag-engaged misses and clean nulls. No cross-seed comparison, no re-run shopping (§5 rules 1/5/6).
- **What to inspect in the TB curve:** (i) best-epoch EMA val MAE + epoch (must clear 22.26, not merely touch parent); (ii) val RMSE/MAE ratio (parent ≈3.9 — joint MAE+RMSE drop implicates small-instance rescue, the claimed mechanism); (iii) `train/loss_epoch` vs parent 2.618 (train-worse + val-worse = N0006-style corruption, not resolution gain); (iv) `h2pool.out` norm off zero + `temp` trajectory away from 0.07 + pool-attention entropy (peaked = selector working; uniform = h2 pool adds nothing over mean).
- **What a negative means:** a clean null (`out` norm ≈ 0 or engaged-but-missing) falsifies "h2-pooled keys restore small-exemplar evidence the h3 pool drops" and redirects to Direction 2 (`use_simcal`, anisotropy calibration) — never to a 75-key retry. An engaged-but-worse outcome (N0006 pattern) falsifies the bank *form* (pooled keys at any single octave insufficient) and points at token-level or multiscale banks under the same budget caps.

## 5. Booked hypothesis set (machine-readable — exactly one runner-parseable line)

1. **H0006** — IF h2-pooled 3-key similarity bank via use_h2pool IN a solo single-switch child of live parent N0002_h0001 trained under the canonical protocol at seed 20260830 for 32ep/1800s with the module constructed AFTER all parent modules with append-only RNG and decoder in_ch untouched at 12.5k added params, THEN final EMA val MAE falls at least 0.30 below live parent 22.5641 to 22.26 or lower AND gt>500 dense-image mean |Δ| falls below the parent 511.7 baseline, BECAUSE per-exemplar h2-at-1/8 3x3 attention-pooled keys retain small-object part geometry that h3-pooled keys average into background while the proven 3-key K-normalized readout at 48x48 upsampled to 96x96 preserves optimizer-trusted comparison dynamics within the time budget, DISPROVED IF final EMA val MAE exceeds 22.26 under the canonical protocol at seed 20260830 AND gt>500 dense-image mean |Δ| is not reduced below the parent 511.7 baseline.
