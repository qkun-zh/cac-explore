# Causal feedback — N0006_h0005 (H0005 `use_verify`)

- **Node:** N0006_h0005 · **Parent:** N0002_h0001 (22.5641 @ep30) · **Switch:** `use_verify` SOLO.
- **Outcome:** 23.5802 @ep31 (`result.json`: `best_mae 23.58022117614746`, `n_epochs_done 32`, `budget_hit true`, `elapsed_s 1815.4`; `info.json` status `timeout`). Margin **+1.016 vs parent** — falsifier bar (≤22.26) missed by 1.32. Verdict: **DISPROVED**.
- **Method:** direct weight forensics on the server checkpoint `run/latest/best.pth` (`epoch 31`, `best_mae` matches `result.json`), plus harness-artifact exclusion. Distinct angle from curve/log reads: did the gate learn anything at all, and did *it* cause the regression?

## 1. Checkpoint forensics (server `best.pth`, `model` state_dict)

| Param | Shape | ‖w‖₂ | mean | max|·| | Init | Moved? |
|---|---|---|---|---|---|
| `head.verify.prop2.weight` | (64,32,1,1) | **2.2408** | −0.0009 | 0.1845 | zeros | **yes, far** |
| `head.verify.prop2.bias` | (64,) | **0.0612** | 0.0004 | 0.0201 | zeros | yes |
| `head.verify.ver2.weight` | (64,32,1,1) | **3.8631** | 0.0068 | 0.2252 | zeros | **yes, far** |
| `head.verify.ver2.bias` | (64,) | **0.4864** | 0.0234 | 0.1329 | zeros (V=0.5) | yes |
| `head.verify.ver1.weight` | (32,34,1,1) | 4.2355 | — | 0.3675 | Kaiming | yes |
| `head.verify.prop1.weight` | (32,64,1,1) | 3.6896 | — | 0.2937 | Kaiming | yes |
| `head.verify.qproj.weight` | (32,128,1,1) | 4.2724 | — | 0.2245 | Kaiming | yes |
| `head.verify.kproj.weight` | (32,256) | 6.2618 | — | 0.2661 | Kaiming | yes |
| `head.verify.temp` | () | **0.0214** | — | — | 0.07 | **yes, ÷3.3 sharpened** |
| `head.simprior.out.weight` (ref) | (64,2,1,1) | 0.9591 | −0.0111 | 0.1791 | zeros | yes |
| `head.simprior.out.bias` (ref) | (64,) | 0.0568 | — | 0.0188 | zeros | yes |
| `head.simprior.temp` (ref) | () | 0.0546 | — | — | 0.07 | yes, mild |

- **Channel liveness (dead iff ch-norm < 1e-4):** prop2 0/64 dead (ch mean 0.2686, min 0.1162, max 0.4764); ver2 0/64 dead (mean 0.4538, min 0.1462, max 0.8262). No collapsed/pruned channels — the whole gate is alive.
- **Functional sanity on synthetic N(0,1) inputs:** V branch → mean 0.5435, std 0.1784, range [0.0009, 0.9998], only 29.9% of cells in [0.45, 0.55] — V is **polarized, not stuck at 0.5**. P branch → std 0.2042, max|·| 1.76 — residual is **non-zero with teeth**, and its out-norm (2.24) exceeds the confirmed simprior readout (0.96) by 2.3×. The gate out-shouts the mechanism that actually helped.

## 2. Mechanism-fate verdict: **used-and-harmful** (active corruption)

- **Not ignored:** zero-init is fully escaped — prop2 (the "gate used iff off zero-init" diagnostic from `idea.md` §4) is 2.24 away from zero on every channel; V deviates strongly from 0.5. The optimizer *took* the gate.
- **Not partially-learned / dead:** all 64 P-channels and 64 V-channels live; temp learned aggressively (0.07→0.021). This is a fully-engaged module, not a passenger.
- **Harmful:** with the gate fully engaged, final MAE is +1.016 worse than the same-seed parent. The residual the decoder reads (`cond_map + simprior + verify`) is dominated by the new, unproven term.

## 3. Causal path (why an engaged gate hurts)

1. **Self-echo proposal:** P is computed *from* `cond_map` itself (`prop1→GELU→prop2`), not from independent evidence. At step 0 this is safe (all-zero), but once prop2 leaves zero the loop `cond → P → cond+P` amplifies whatever MSE-smeared structure `cond` already has. A sharpener built from blur inherits the blur — then amplifies it.
2. **Over-sharpened verification:** `verify.temp` 0.07→0.0214 makes the K=3 softmax over frozen-geometry cosine similarities ~3× peakier than designed (and 2.5× peakier than the working simprior temp 0.0546). `top1`/`consensus` evidence on 32-dim q/k (vs simprior's 64-dim) is lower-fidelity to begin with; sharpening it turns small cosine noise into hard per-cell V decisions — exactly the wrong texture for separating merged dots, and consistent with a polarized V histogram coexisting with worse counts.
3. **Magnitude dominance:** ‖prop2‖ (2.24) + ‖ver2‖ (3.86) drive a residual larger than the confirmed simprior term (0.96). The decoder's `cat([fine, cond_map])` interface (in_ch still 192) now sees cond mass steered mostly by the corrupt gate. Regression follows without any other suspect: no backbone/config/schedule change (see §4).

## 4. Harness-artifact exclusion (all checkable)

- **Seed:** `config.toml:9` `seed = 20260830` — canonical fixed seed, no override.
- **Checksums (runner-recorded, locally re-verified):** `result.json` `config_sha256 88b8f37902f414f1` == `sha256sum config.toml` prefix `88b8f37902f414f1…` ✓; `model_sha256 d239d6413086b852` == `sha256sum model.py` prefix `d239d6413086b852…` ✓. The run trained exactly the filed code+config — no stale-file artifact.
- **RNG order:** `Verify` constructed after all parent modules (`model.py:299-300`), flags are non-module booleans; forward has no stochastic ops; augment=false, deterministic loaders. Child-vs-parent divergence is the learned gate only (AGENTS §5.13).
- **`timeout` semantics:** `budget_hit true` is the 1800 s wall-clock ceiling, but all 32 epochs completed (`n_epochs_done 32`, best @ep31, parent best @ep30 of 32). Nothing was truncated mid-schedule; the comparison is full-schedule vs full-schedule.
- **Precision:** margin +1.016 ≫ same-seed paired-contrast precision ~0.02 (AGENTS §6). Not noise.

## 5. What this closes

- H0005's core claim — "repetition-gated P×V sharpening of post-Condenser cond mass restores dense recall" — is refuted **with the strongest form of negative**: the gate was learned, engaged, and moved the metric the wrong way. "Gate ignored" and "under-trained" are both excluded by the norms above.
- Per AGENTS §11, no silent retry of `use_verify`. Any revisit (e.g. P from `fine` instead of self-echo `cond_map`; temperature floor ≥0.05; V supervised by objectness) books as a *new* hypothesis with a *new* falsifier, and must explain why it avoids the echo+oversharpen path documented here.
