# Qualitative feedback — N0010_h0016 (H0016 SOLO, DAVE-lite verify-and-suppress)

- Node: `tree/N0001_champion/N0010_h0016`, H0016 solo (`use_verify_mask`), seed 20260830.
- Result (from `result.json:5,6`): best_mae **22.8831 @ep32** vs canonical parent **23.2932 @ep26** → **−0.4101**, clears the anchor-rule line 22.9932 by 0.11; ledger `supports w=1.0`, H0016 conf 0.600.
- Sources read: child `model.py`, `config.toml`, `idea.md`, `result.json`, `info.json`; parent `tree/N0001_champion/model.py`; siblings N0008/N0009 `model.py`; H0016 rows in `memory/hypotheses.jsonl`; server `best.pth` weights via cac-server python (see §2). No `run_node.log` exists locally (`run/latest/` holds only TB events + hparams); training dynamics below reuse the TB-derived epoch table in `feedback/quant.md:36-53`.
- Scope: mechanism fidelity + behavior reading. No verdict.

## 1. Fidelity: the code is the pre-registered mechanism, with zero appearance parameters

The registered H0016 text (`idea.md:7`, identical to the ledger `create` row) asks for: pool fused fine features at the top-scoring peaks of the preliminary density, score each by cosine against an exemplar prototype "pooled in the same fine-feature space", and multiply the density by a learned-threshold gate before the loss. `VerifyMask` (`model.py:155-212`) implements exactly that:

- **Identity-at-init, confirmed bit-exact by construction**: `raw_gain` and `b` are explicit-zero scalars (`model.py:190-191`, no RNG draw — so shared-module init draws stay the champion's per AGENTS rule 13). At `a=0`, `gate = 1 − 0·sigmoid(·) = 1` for every cos, `supp = 1 − gate = 0`, and `dens * (1 − 0)` returns `dens` unchanged (`model.py:209-212`). The clamp headroom `max=1.0+1e-6` (`model.py:210`) is strictly inactive at init yet keeps `dL/da = −Σ dL/dgate·sigmoid(b−cos) ≠ 0` on step 0 (sigmoid > 0 everywhere), so the gate starts learning immediately; `b` gets gradient once `a ≠ 0`.
- **Peak selection**: `K = min(max(8, round(0.01·N)), N)` on the *detached* density (`model.py:197-198`) — indexing only, no gradient through selection. At S=384, Hf=Wf=96, N=9216 → K=**92**, matching the card spec. Peak features are gathered from `fine` (`model.py:199-200`).
- **Prototype**: `roi_align(fine, rois, 1×1)` at the exemplar boxes, mean over Ke (`model.py:201-205`) — same fine space as the peaks, as registered. Cosine is computed in fp32 after explicit normalize (`model.py:206-208`); learned offset `b` sets the gate half-point at cos ≈ b (`model.py:169-175`).
- **Simplification note**: as coded the module holds **exactly 2 parameters** (server ckpt shows 260 keys, of which only `head.verify_mask.raw_gain` and `head.verify_mask.b` match; there is no Linear/conv/projection anywhere in `VerifyMask`). Appearance matching reuses the 128 fused-fine channels directly. The registered text (`idea.md:7`) likewise specifies same-space cosine with no projection head, so code matches registration — the simplification is relative to a full DAVE-style verifier with a learned appearance projector: there is none here, the similarity lives or dies on the frozen-backbone-derived fine space.

## 2. Learned behavior: suppression turned ON, gently — (a, b) = (0.1663, 0.1885)

Read server-side from the evaluated weights (`/data/cac/tree/N0001_champion/N0010_h0016/run/latest/best.pth`, `ckpt["model"]`):

- `head.verify_mask.raw_gain` = **0.16633445**, `head.verify_mask.b` = **0.18852192** (float32 scalars).
- Sign check: `a > 0` → the gate suppresses in the intended direction (had AdamW driven `a` negative it would have become an enhancer; it did not). `b ≈ +0.19` puts the half-point at weak-positive cosine — the gate's steepest discrimination sits just above zero similarity.
- Operating points (`gate = 1 − a·sigmoid(b − cos)`):

| cos | +1.0 (perfect match) | +0.5 | +0.19 (= b) | 0.0 | −0.5 | −1.0 (anti-match) |
|---|---|---|---|---|---|---|
| gate | 0.9488 | 0.9297 | 0.9168 | 0.9090 | 0.8893 | 0.8725 |
| suppression | 5.12% | 7.03% | 8.32% | 9.10% | 11.07% | 12.75% |

- Reading: this is a **mild, nearly-uniform damper with a small exemplar-selective tilt** — even a perfectly matching peak loses ~5%, the worst mismatch loses ~13%, and the full match-vs-mismatch differential is only **~7.6pp**. The gate never approaches full suppression (floor ≈ 0.87 given learned `a`), and it cannot enhance (clamp cap is moot: with `a > 0`, gate < 1 strictly). Training dynamics corroborate gradual turn-on: the child trails the parent for 12 straight epochs (ep4–15, quant.md:56) while `a` grows from 0, crosses over at ep16 on the parent's +0.056 uptick, then leads monotonically to ep32 — i.e. the mask starts as the champion (identity) and only pays once the (a, b) operating point is found.

## 3. Does the mask suppress false positives as claimed? Only indirectly observable

What is observable: (i) the learned sign/magnitude above — suppression is real, ON, and monotone non-decreasing in exemplar-similarity, exactly the claimed direction; (ii) the MAE/RMSE divergence in the child tail (quant.md:84 — RMSE bottoms @ep23 then +0.689 while MAE falls every epoch) is the signature of a suppressor trimming many small overcounts while leaving a few large misses; (iii) the widening 17-epoch lead (quant.md:56-58) arrives without any train-fit penalty (child train loss 92.74→2.603 fits slightly *better* than parent, quant.md:82), consistent with removing spurious mass rather than underfitting.

What is NOT observable from scalars alone: per-peak suppression (no gate/cos histograms or density-map dumps exist — AGENTS §9 records no stored feature dumps at runtime), so we cannot say what fraction of the −0.41 comes from distractor peaks vs the ~5% uniform damping of all 92 peaks, nor where the cosine distribution sits relative to the b≈0.19 threshold. A follow-up val-time dump of (cos, gate) histograms would settle how selective the mask actually is; on current evidence "suppresses false positives" is *directionally supported but mechanistically unresolved* — the uniform-damping component alone could explain part of the gain if the parent systematically overcounts.

## 4. Pluggability: OFF restores the champion; suppression-only by construction

- OFF path: `use_verify_mask` defaults False (`model.py:230`); module not built (`model.py:231-232`), forward skips it (`model.py:242-243`). Parent `tree/N0001_champion/model.py:143-168` has no `VerifyMask` at all, and the config diff vs parent is exactly one line (`use_verify_mask = true`, `config.toml:27`). Diff surface: docstring + `VerifyMask` class + 2 init lines + 2 forward lines + 1 config line — clean single-switch ablation.
- One ordering subtlety: the mask applies to the decoder density *before* the GCA uniform bias is added (`model.py:243` then `model.py:292` `out["density"] = dens + bias`), so ~2% of count mass (the `0.02·n_aux/N` bias) bypasses verification — negligible, but a purist "verify everything" variant would mask after.
- Suppression-only: clamp `max=1.0+1e-6` (`model.py:210`) caps any enhancement at 1e-6, and with learned `a>0` the gate is strictly < 1 anyway.

## 5. Why post-decoder intervention may succeed where pre-condenser gates failed

- N0008 `SafeEnhance` rebuilds every fmap token as a similarity-weighted mix of exemplar tokens *before* the condenser (~33k new params, `N0008_h0014/model.py:136-179`); N0009 `DynExemplarGate` rescales exemplar channels fed as the condenser's keys/values (`N0009_h0015/model.py:120-139,199-200`). Both perform representational surgery upstream of the decoder nonlinearity: any miscalibration is re-mixed by cross-attention and the dilated decoder before reaching the loss, and both add thousands of degrees of freedom that must survive a 32-epoch budget.
- `VerifyMask` instead multiplies the loss-adjacent output at 92 indexed peaks with 2 parameters and a one-step gradient path (`dL/da ≠ 0` at init). Its worst case is bounded (gates ∈ [0, 1]) and monotone in a quantity — exemplar cosine — that the frozen fine space already provides. The plausible lesson: under a frozen backbone + short budget, *short-path, low-capacity, output-side corrections with exact identity-at-init* beat *high-capacity, input-side representation reshaping* — the former can only nudge mass, the latter must re-teach the condenser/decoder how to read altered tokens.

---

- Learned gate is ON but gentle: (a, b) = (0.1663, 0.1885) → 5.1% suppression even at cos=1, 12.8% at cos=−1, 7.6pp of exemplar-selectivity — a mild damper with a tilt, not a hard verifier.
- Fidelity is exact (identity-at-init, detached top-92, fp32 same-space cosine, 2 scalars, one-switch ablation), but per-peak selectivity is unobservable without gate/cos dumps — the uniform-damping component may carry part of the −0.41.
- Post-decoder, 2-parameter, loss-adjacent suppression succeeded where pre-condenser representation surgery failed; carry forward: keep interventions output-side, identity-at-init, and capacity-minimal.
