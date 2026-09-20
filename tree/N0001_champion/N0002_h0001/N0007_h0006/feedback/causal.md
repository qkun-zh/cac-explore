# N0007_h0006 — causal feedback (H0006 `use_h2pool`)

- **Node:** N0007_h0006 · **Parent:** N0002_h0001 (22.5641 @ep30) · **Result:** 47.8439 @ep32, 32/32ep, `budget_hit=false`, elapsed 1803.7 s · **Verdict: REFUTE** (misses 22.26 bar by +25.58).

## 1. Checkpoint probe (server `run/latest/best.pth`, epoch 32, best_mae 47.8439)

Probed via `ssh cac-server` (`/data/miniconda/envs/cac/bin/python`, `cac.hub.setup_hf_env()` first, `PYTHONPATH=/data/cac/src`). Weight norms (L2):

| Param | N0007 (47.84) | N0002 parent (22.56) |
|---|---|---|
| `head.h2pool.out.weight` (64,2,1,1) | 0.1966 | — |
| `head.h2pool.out.bias` (64,) | 0.3128 | — |
| `head.h2pool.pool_attn.weight` (1,192) | 0.4384 | — |
| `head.h2pool.kproj.weight` (64,192) | 3.5461 | — |
| `head.h2pool.temp` | **0.0524** (init 0.07) | — |
| `head.simprior.qproj.weight` | **3.5155** | **5.7314** |
| `head.simprior.kproj.weight` | **3.5428** | **7.3547** |
| `head.simprior.out.weight` | 0.3611 | 0.5874 |
| `head.simprior.temp` | **0.0547** | **0.0393** |
| `head.decoder.head.weight` | 2.5853 | 1.2862 |
| `head.cond.out.weight` | 5.6698 | 3.2677 |

## 2. Forward probe (best.pth weights, real val batches, `torch.no_grad`)

- `pool_attn` entropy **2.1971 ≈ log(9) = 2.1972**, mean max-attention 0.114 ≈ 1/9 → the "learnable selector" learned nothing; it is a plain 9-token mean.
- `Sv` (h2 cosine volume) range **[-0.36, -0.08]**, std 0.051, across-K std **0.039** → all-negative mush with zero K-discriminability (no margin for any temperature to sharpen).
- `W` entropy 0.949 / max 1.099 → near-uniform over K, as forced by flat `Sv`.
- `d48` RMS 0.045, evidence |mean| 0.21 → the residual itself is small.
- Count probe on dense val images: pred 105.7 vs target 429.0 → severe under-count, consistent with the 47.84 MAE.

## 3. Training-curve causality (TB `val/mae`, `train/loss_epoch`)

- Parent N0002: ep1 val 153.3 → ep3 48.8 → ep8 27.3 → 22.56; train 38.0 → 8.7 → **2.62**.
- Child N0007: ep1 val 219.6 → ep3 76.0 → ep8 48.5 → **stuck 47.84**; train 19.6 → flat **~13.9–14.3 forever**.
- Both start catastrophic at ep1 (fresh-readout disruption under zero-init is normal — step-0 equality does not survive the first epoch's updates). The child differs in that it **never recovers**: train loss 5.3× parent's (13.92 vs 2.62) means this is an **optimization failure, not a generalization gap** — the model cannot even fit train.

## 4. Mechanism verdict (attribute, then rule out)

- **Primary: shared-query corruption of the confirmed simprior readout.** `H2Pool.forward` reuses `simprior.qproj` on `fine_down`. The h2 key-side never produced discriminative geometry (uniform pool, collapsed all-negative `Sv`), so as `h2pool.out` grew off zero it injected a growing garbage-gradient stream into the shared `qproj` — the exact projection the confirmed H0001 readout needs. Result: `qproj`/`kproj` froze at ~half the parent's norms (3.5 vs 5.7/7.35), `simprior.temp` never sharpened (0.055 vs 0.039), `simprior.out` stayed immature (0.36 vs 0.59). The load-bearing readout was starved, not supplemented.
- **Secondary, compensatory (not causal): decoder growth.** `decoder.head` 2.59 vs 1.29 and `cond.out` 5.67 vs 3.27 are the optimizer pushing the only free gain knobs on a corrupted `cond_map` — effect, not cause.
- **Ruled out — temp collapse:** temps 0.052/0.055, mild drift from 0.07; nothing like N0006's ÷3.3 collapse to 0.021.
- **Ruled out — exploding residual:** `h2pool.out` norms 0.20/0.31, live `d48` RMS 0.045. The branch is engaged-but-impotent, not explosive.
- **Ruled out — diverged decoder readout as prime mover:** decoder growth follows the corrupted conditioning (train stuck proves the features, not just the readout, are broken).
- **Ruled out — harness:** server config has `seed = 20260830`, `augment = false`; server `result.json` checksums (`4ba6d04d77ab6a88` / `e709b1c83c4c0055`) re-verified byte-identical against server `sha256sum` of `config.toml`/`model.py` AND local `sha256sum` (first-16 match both files); 32/32 epochs, `budget_hit=false`. Same-seed paired contrast vs 22.5641 is valid; the +25.28 catastrophe is model, not harness.

## 5. Causal ordering for synthesis

`h2 stage-2 geometry mismatched to h3-tuned query space (kproj-from-scratch 192→64 + shared qproj) → flat all-negative Sv → pool_attn gets no discriminative gradient, stays uniform → garbage-gradient stream into shared qproj grows with out.weight → confirmed simprior readout never re-matures (q/k under-grown, temp unsharpened) → train AND val stuck (13.9/47.8) → decoder compensatory doubling on corrupted cond_map.` Zero-init guaranteed step-0 equality but could not protect the shared projection once training started — the "reuse qproj" cost-saving was the load-bearing mistake (same family as H0004's qproj lesson, opposite direction: H0004 paid forward-cost for a second qproj; H0006 paid gradient-cost for sharing it).

## 6. Redirect (per idea.md §4: engaged-but-worse ⇒ bank *form* falsified)

- Do NOT retry pooled keys at any single octave, and do NOT share a live readout's query projection with an unproven key-side — any future bank needs its own query path or frozen-query discipline.
- Redirect to Direction 2 (`use_simcal`, anisotropy calibration of the proven h3 bank), never to wider/token-max banks.
