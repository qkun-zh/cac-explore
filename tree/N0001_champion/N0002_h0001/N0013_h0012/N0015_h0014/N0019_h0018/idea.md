# H0018 — LinGain: linear wide-swing per-cell gain replacing compressed log gain (SOLO card)

- Parent: `tree/N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014` (LIVE parent N0015_h0014, val MAE 19.3431 @ep32 v2 protocol, seed 20260830).
- SOLO card: exactly one mechanism switch `use_lingain` (default false), shipped as an explicit swap: `use_lingain=true` AND `use_cellcal=false` (two-line config delta; parent file keeps `use_simprior=true`, `use_peakcal=true` untouched). `elif` structure in `SimPrior.forward` — when `use_lingain` is on, the LINEAR gain REPLACES cellcal's log-gain; both never active together. Single testable change: the gain FORM.
- Regime: frozen backbone, head-only training. Decoder `in_ch` untouched (still `D + cond_dim` = 192); optimizer/loss/schedule invariant (AdamW, MSE+w_cnt·L1, cosine, AMP).
- Protocol: v2 (augment=true, EMA eval, seeded loaders), seed FIXED 20260830, 32ep/1800s. Same-seed paired contrast vs live parent N0015 only.
- Triple conjunctive bars (§6 semantics, bound to live parent 19.3431 under v2): THEN final EMA val MAE ≤ 19.0431 (≥0.30 below 19.3431) AND dense-tail below 330.81 AND sparse gt<50 at or below 5.45. Missing any one bar refutes the card. Launch carries `--set futility_bar=19.0431` (see §5).

## §0 — Bookable one-liner (for `discovery hypo --new`)

IF a linear wide-swing per-cell gain `gain = 1 + w_lin * mhat_clamped` replacing the log-compressed cellcal gain is applied to the SimPrior evidence IN the live N0015 head on FSC147 v2, THEN final EMA val MAE drops at least 0.30 below 19.3431 with dense-tail below 330.81 and sparse gt<50 at or below 5.45, BECAUSE the log1p compression caps cellcal's swing at 1.2–1.3x while the mid-collapse block needs 5–20x local elevation, and a linear-in-mhat gain restores the expressible dynamic range so collapsed mid-count cells receive sufficient mass without rescaling the exemplar embedding or the decoder. DISPROVED IF final EMA val MAE is not at least 0.30 below 19.3431, or dense-tail is not below 330.81, or sparse gt<50 exceeds 5.45, or the gain-spread read shows no wide swing (p99/p1 ≈ cellcal's ~1.3x) with w_lin ≈ 0.

## §1 — Survey: gain-schedule / dynamic-range calibration in density estimation (verified arXiv IDs; no matching literature cited)

All three IDs below were verified to resolve (abstract + full text checked 2026-09-21). The point of this survey is the FORM of calibration, not exemplar matching: density estimators repeatedly need a per-location dynamic-range control that lifts high-magnitude regions by multiples, and the choice of nonlinearity decides what swing is expressible.

1. CACViT magnitude embedding — arXiv:2305.04440 (Wang et al., "Vision Transformer Off-the-Shelf: A Surprising Baseline for Few-Shot Class-Agnostic Counting"). Verified: arXiv abs + alphaXiv mirror both resolve to the same title/abstract; core claim confirmed — resizing/normalization in a plain ViT destroys scale and order-of-magnitude information, so the authors add explicit scale and magnitude embeddings to restore dynamic range. Relevance: independent confirmation that counting pipelines lose magnitude information by construction and must re-inject it with a dedicated calibration path; H0018 is that path at the per-cell gain level. Difference: CACViT embeds magnitude as an input feature; LinGain applies it as a multiplicative gain on the similarity evidence, which directly scales predicted mass.
2. L2HCount low→high density generalization — arXiv:2503.12935 (Xu et al., "L2HCount: Generalizing Crowd Counting from Low to High Crowd Density via Density Simulation"). Verified: arXiv abs page + PDF both resolve; contributions confirmed — High-Density Simulation Module + Ground-Truth Generation Module synthesize high-density patterns from low-density images so the model learns high-magnitude behavior it never saw. Relevance: direct evidence that the hard failure mode in counting is HIGH-magnitude under-prediction (models trained on low/mid density collapse on dense scenes), matching our measured 3481-class mid-collapse (pred 22–100 on gt 260–460). Difference: L2HCount fixes the regime with simulated data; LinGain fixes it with gain-form expressivity — complementary, not overlapping.
3. DecideNet adaptive density-mode weights — arXiv:1712.06679 (Liu et al., "DecideNet: Counting Varying Density Crowds Through Attention Guided Detection and Density Estimation", CVPR 2018). Verified: arXiv abs (v1 2017-12-18, v2 2018-03-07) + CVPR open-access page both resolve; mechanism confirmed — QualityNet outputs a per-pixel attention map K(p) blending detection-based and regression-based density maps: D_final = K·D_det + (1−K)·D_reg. Relevance: the canonical precedent for a learned per-location calibration weight driven by local density conditions; per-cell gains are an established, legitimate actuator in density estimation. Difference: DecideNet blends two estimators with a bounded [0,1] weight; LinGain scales one evidence stream with an UNBOUNDED linear gain — the unboundedness is the entire bet (and the entire risk; see §5).

What the survey decides: none of the three gives a per-cell multiplicative gain whose FORM is tested against log-compression; the wide-swing linear form is unclaimed in this lineage. First-principles lens (pure math): a gain of the form exp(w·log1p(m)) is provably bounded in swing for bounded w — it cannot express 5–20x no matter the optimization trajectory, so the mid-block collapse is a FORM ceiling, not a training failure. Champion-lineage lens: cellcal (H0012, w_c=+0.148 CONFIRMED) proved the surrogate (per-cell self-normalized energy mhat) and the actuator (multiplicative gain on ev) both work; H0018 keeps both and changes only the nonlinearity. Counter-intuitive lens: the fix is SIMPLER than the winner — delete the exp/log1p pair, keep a straight line — and simplicity is falsifiable fast (w_lin ≈ 0 kills it in one run).

## §2 — Exact mechanism: LinGain v2 of the confirmed cellcal winner

### 2.1 The measured compression flaw (the raison d'être, with numbers)

Parent N0015 `SimPrior.forward` (model.py:225–239) computes:

```
m      = fine.detach().mean(dim=1, keepdim=True).clamp_min(0)   # (B,1,96,96)
mbar   = m.mean(dim=(2,3), keepdim=True)                        # (B,1,1,1)
mhat   = m / (mbar + 1e-6)                                      # mean 1.0 by construction
gain_c = exp(w_c * log1p(mhat))                                 # cellcal log-gain, model.py:238
ev'    = ev * gain_c                                            # broadcast over both channels, model.py:239
```

with confirmed w_c = +0.148. The compression is MEASURED, not hypothesized: mhat over val cells concentrates at 2–5x in the mid-collapse block, and log1p maps that interval to [log1p(2), log1p(5)] = [1.0986, 1.7918]. The gain swing across the whole block is therefore:

```
gain(2) = exp(0.148 × 1.0986) = exp(0.1626) ≈ 1.177
gain(5) = exp(0.148 × 1.7918) = exp(0.2652) ≈ 1.304
```

i.e. 1.2–1.3x total swing. The 3481-class mid-collapse block needs 5–20x local gains (pred 22 on gt 260 → 11.8x; pred 100 on gt 460 → 4.6x; 56% of total error). No value of w_c reachable by gradient descent fixes this within the log form: doubling w_c to 0.30 still gives only exp(0.30×1.79) ≈ 1.71x at the top end, while pushing the bottom end up uniformly (broad 1.2x shift, already harvested). The ceiling is structural. LinGain removes it.

### 2.2 The replacement form (exact math)

When `use_lingain` is on (and `use_cellcal` off — elif, never both):

```
mhat_c = (fine.detach().mean(dim=1, keepdim=True).clamp_min(0)
          / (mbar + 1e-6)).clamp(0, 10)     # recomputed decoupled from fine.detach(); guarded ops
gain   = (1 + w_lin * mhat_c).clamp_min(0.0) # (B,1,96,96); w_lin single zero-init scalar
ev'    = ev * gain                            # broadcast over both channels, same site as model.py:239
```

Swing comparison at the same operating point (mhat 2–5): with w_lin = 0.5, gain runs 2.0–3.5x; with w_lin = 1.0, gain runs 3.0–6.0x; with mhat reaching the clamp 10 and w_lin = 1.0, gain reaches 11x — inside the 5–20x band the mid-block needs. The linear form is unbounded in w_lin (up to the clamp), where the log form is provably capped near ~1.3x at the learned w_c. That asymmetry — same surrogate, same actuator, strictly larger expressible range — is the single testable change.

### 2.3 Twin pre-emption (dedicated paragraph; the gate will probe this)

Same actuator family and same surrogate BY DESIGN: LinGain is a v2 of the confirmed H0012 winner, not a new family — it reads the identical per-cell self-normalized energy mhat from `fine.detach()` and applies it as a multiplicative gain on the identical evidence tensor `ev` at the identical broadcast site. A novelty gate that bins by "per-cell gain on similarity evidence" will flag this as a twin of cellcal, and that flag is factually correct at the family level. The card survives the twin probe at the nonlinearity level with a MEASURED consequence: log-gain swing 1.2–1.3x (computed above from the confirmed w_c = 0.148) versus linear-gain swing unbounded up to 1+w·10 — a full order of magnitude of additional dynamic range that no log-gain parameterization reaches. The claim differs (mid-block 5–20x recovery vs broad 1.2x lift), the operating regime differs (collapsed high-magnitude cells vs whole-image energy field), and the falsifier differs (gain-spread read p99/p1 ≫ 1.3x + mid recall + sparse guard ≤ 5.45 — a triple no log-gain run satisfies, because its spread is provably narrower). If the gate still rules twin, the card dies honestly: the §5 fallback is exemplar-distinctness — re-book only with a NEW surrogate (not mhat) and a NEW falsifier, never a re-encoded log/linear swap.

### 2.4 Attach points (file:line refs against live parent model.py)

- `CountingHead.__init__` (model.py:147–163): add non-module flag `self.use_lingain = _get(cfg, "use_lingain", False)` beside `use_cellcal` (model.py:161) / `use_peakcal` (model.py:163). Non-module flag: no RNG consumption.
- `CountingHead.forward` (model.py:175–185): extend the gating cascade with an `elif` so the linear gain REPLACES the log gain — `if use_cellcal … elif use_lingain …`, passing `w_lin=self.lingain.w_lin` through new kwargs `use_lingain, w_lin` on `SimPrior.forward` (current signature model.py:225). Config launches with `use_cellcal=false`, so only the linear branch executes; both-on is structurally unreachable in this card.
- `SimPrior.forward` gain site (model.py:234–239): the linear math lives exactly where `gain = torch.exp(w_c * torch.log1p(mhat))` (model.py:238) and `ev = ev * gain` (model.py:239) sit today; mhat is recomputed decoupled (own detach/clamp chain, guarded ops) rather than reusing cellcal's tensor.
- New scalar module `LinGain` (mirrors `CellCal`, model.py:255–266): ONE zero-init scalar `w_lin = nn.Parameter(torch.zeros(()))`; no norms, no dropout, no shape changes.
- `Counter.__init__` attach order (model.py:301–316): `self.head.lingain = LinGain()` constructed LAST, after `peakcal` (model.py:312) — append-only RNG order per AGENTS rule 13; zero-init draws no RNG so every parent init draw keeps its exact position.
- Untouched: `DensityDecoder` (model.py:128–140, `in_ch = D + cond_dim` at model.py:156); exemplar embedding `e` (read-only, never gated/rescaled); `fine` read via `detach()` only; `Condenser` (model.py:111–125); `GCA` (model.py:191–203); no match/prototypes/queries/attention/temperature touched.

### 2.5 Step-0 identity proof + guards + temp-pin

At w_lin = 0 (init): gain = (1 + 0·mhat_c).clamp_min(0.0) ≡ 1 everywhere, so ev' ≡ ev and forward is numerically identical to the parent with `use_cellcal=false` — clean ablation by removing one switch. The `clamp_min(0.0)` is an anti-collapse guard, NOT an identity-changer: at w = 0 gain is exactly 1 with or without it; for w_lin < 0 and large mhat it prevents sign-flip suppression (negative density evidence), keeping the card mass-adding by construction. Finite-guards: mhat clamped to [0, 10] (kills runaway cells); all ops guarded (clamp_min on means, no bare divisions — same hygiene as model.py:235–237). Temp-pin hygiene REQUIRED: `SimPrior.temp` stays detached in the softmax (model.py:230, `self.temp.detach().clamp_min(1e-3)`) and pinned (`requires_grad_(False)`, cf. model.py:315–316) — the linear gain must not reopen a free-temperature path; the Coding Agent keeps the exact parent temp handling byte-for-byte.

## §3 — Why not (alternatives considered and rejected with measurements)

- Keep cellcal and raise w_c (status-quo scaling): REFUTED by the §2.1 arithmetic — the log1p form caps swing at ~1.3x at w_c = 0.148 and ~1.7x even at w_c = 0.30; the mid-block needs 5–20x. More of the same form cannot reach the target band; only a form change does.
- H0009 no-learned-gate lesson: H0009's fixed-form gate failed because its form was frozen with gradient from step 0 through a hand-set shape — LinGain inverts that lesson (learned scalar, zero-init, gradient-driven from step 0 through the live loss, form chosen for expressivity rather than fixed). Same pluggable discipline, opposite failure mode.
- New surrogate (replace mhat with a different field): rejected — cellcal CONFIRMED the mhat surrogate (w_c = +0.148 learned positive, dense 330.81 improved); swapping the surrogate AND the form in one card breaks attribution (two changes, one hypothesis — §4 violation). Surrogate stays, form moves.
- Resolution lever (512 input): REFUTED by probe (23.56 vs 21.44 — overfits; representation-bound, not quantization-bound). Resolution is dead; dynamic-range form is live.
- §11 compliance: no DDCA, no extra spatial summaries/RGA, no final-layer readout change, no backbone unfreeze, no pre-condenser exemplar gating; frozen hs(2,3) readout + cross-attn condenser untouched; decoder in_ch untouched. The card is a decoder-side-adjacent gain at the SimPrior→cond interface — a live direction, booked fresh with a new falsifier.

## §4 — Predictions, failure modes, and the 4 mechanism reads

Predictions (all four must hold; triple bars + spread proof):

- R1 — LEARNED POSITIVE GAIN WITH WIDE SWING (the wide-swing proof): w_lin converges decisively positive (≥ +0.2, i.e. clearly off zero relative to training noise ~0.02), AND the val-cell gain spread p99/p1 ≫ cellcal's ~1.3x (expect ≥ 3x). A positive w_lin with narrow spread refutes the mechanism (it re-encoded cellcal, not a v2).
- R2 — MID-BLOCK RECALL: 3481-class mid-collapse images (gt 260–460, currently pred 22–100) rise 3–10x toward gt; the block's share of total error drops materially below 56%. No mid movement with bars met = wrong mechanism (gains went elsewhere).
- R3 — SPARSE GUARD (the deciding read): sparse slice gt<50 stays at or below 5.45. Linear swing risks sparse overshoot by construction — this read, not the MAE bars, is the bet's judge.
- R4 — BARS: final EMA val MAE ≤ 19.0431 AND dense-tail below 330.81, same-seed v2 vs live N0015 (19.3431 / 330.81).

Failure modes (each maps to an honest refutation):

- F1 — QUIET NULL: w_lin ≈ 0 (|w| < 0.05), spread ≈ 1.0, numbers level with parent → the form was unneeded; REFUTE, no re-book.
- F2 — SLIPPAGE DOMINATES (the predicted way to die): mid-block improves (R2 holds) but sparse blows past 6.0 — linear gain cannot discriminate collapsed-mid from healthy-sparse → the bet fails; REFUTE, and any successor must gate the gain by peakiness/density regime, not rescale globally.
- F3 — DENSE COLLAPSE: dense-tail ≥ 330.81 (gains saturated the clamp or destabilized the decoder interface) → REFUTE; successor direction is clamp/schedule surgery, a different card.

## §5 — Risks, v2 pair protocol, futility bar

- Dominant risk — sparse overshoot, quantified: at w_lin = 0.5 a healthy sparse cell with mhat = 3 already takes gain 2.5x; the sparse slice (gt<50, currently 5.814 → bar 5.45) has no headroom for broad 2x lifts. The card bets that gradient descent concentrates gain where loss-mass lives (the 56%-error mid-block) rather than spreading it — a bet on loss geometry, not on the form alone. F2 prices this bet: mid-up + sparse-past-6.0 = honest death, and the loss-mass argument is recorded as wrong.
- Secondary risks: (a) clamp-10 saturation — if val mhat piles at the clamp, the "linear" gain is effectively a step function; read the mhat histogram and report it; (b) twin-gate ruling — fallback is exemplar-distinctness (new surrogate + new falsifier, §2.3), never a silent re-encode of this swap; (c) temp-path reopening — guarded by the §2.5 pin requirement; the smoke diff must show temp handling byte-identical to parent.
- v2 pair protocol: single child of live N0015, same seed 20260830, 32ep/1800s, EMA eval; config delta is exactly two lines (`use_lingain=true`, `use_cellcal=false`); every other key byte-identical to the parent config; verdict is the same-seed paired contrast on the triple bars (level noise ±1 MAE never enters — only the paired delta at fixed seed counts).
- Futility: launch with `--set futility_bar=19.0431` (ref slope 0.06/ep default; ep16 WARN-only / ep24 HALT; margin 2.0). A futility-halted run keeps best.pth: verdict = futility refutation + full eval_test (dense tail + all four mechanism reads still valid and still reported).
