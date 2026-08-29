# Idea — N0059_pom_morph (parent: N0054_xscale_exemplar, LOCKED MAE 19.647)

**PRE-REGISTERED FALSIFIER of H0081** (the load-bearing attention law from N0058). N0058's
PMOM failed (+2.17 same-epoch, +3.50 floor) but with a −42% capacity cut AND a 2×2 part-pool that
destroyed within-part layout — capacity and expressivity are confounded, so the operator class was
never cleanly tested. N0059 removes both confounds and runs the paper's full PoM polynomial mixer
block at matched capacity over all 49 ROI tokens.

## Change (structural REPLACEMENT, FROZEN, optimizer unchanged)
Single switch `use_pom` in ExemplarEncoder (N0054 model.py:64-111). Replace {proj 384→256 +
2×TransformerEncoderLayer(256,4hd,FF1024,norm_first,residual) + attn-pool} with {proj 384→256 +
2× PoM-PolyMorpher blocks + attn-pool}. Condenser/GCA/XScale/FineFuser/decoder UNTOUCHED. Recipe
identical to N0054 (AdamW 1e-3, wd0.05, cosine, bs16, AMP, 30ep @384, MSE+SmoothL1, augment).

Forward (B*K,49,256)→(B,K,256): roi_align(feat,7)→flatten→proj→tok (B*K,n=49,256) + shape_mlp(wh);
2× block P(X)=X+PoM(X)+FF(X+PoM(X)); attn-pool a=tok·(Linear 256→1).softmax(1), out=Σa·tok→(B,K,256);
XScale coarse add on fused prototype (unchanged). Over n=49, dim 256, per block:
- W_h: Linear(256→D); h=GELU(W_h X). α: nn.Parameter(D,k=2), init ±0.001.
- H(X)=[Σ_{p=1..k} α_p ⊙ h^p].mean(dim=1) → (B*K,D)   [mean over 49 DETERMINISTIC; do NOT broadcast-sum]
- gate σ(W_s X) → (B*K,49,D), W_s: Linear(256→D); PoM(X)=W_o[σ(W_s X) ⊙ H(X).unsqueeze(1)] → (B*K,49,256)
- FF: Linear(256→1024)→GELU→Linear(1024→256); LayerNorm before blocks + residual, as Eq.3.

**Param table** (D=352, k=2): proj 98,560 · per block: W_h 90,464 + W_s 90,464 + W_o 90,368 +
α 704 + FF 525,568 + 2×LN 1,024 = 798,592 · ×2 blocks 1,597,184 · attn-pool 257 ⇒ aggregation
**1,696,001** · rest-of-head 1,823,711 ⇒ head **3,519,712 ≈ 3.52M** · frozen backbone ~27.81M ⇒
**TOTAL 31.32M ≤32M ✓** (exact match to N0054 3.50M, +0.6%; trainable ≥1.5M ✓).
**D=352 rationale**: D=512 overshoots 3.5M; D=352 lands head 3.52M / total 31.32M, exactly matching
N0054 — capacity confound removed (D is the sanctioned dial, FF held at 1024).
**Activation deviation (doc'd)**: paper uses h=clamp(LeakyReLU(x,0.01),−0.1,6); we use GELU (project-
internal, minor). Noted, not a design change.

## Why (grounding)
- PoM (arXiv:2604.06129, CVPR-F 2026, §3.1 Eq.3) — degree-k polynomial mixer with a token-averaged
  moment state H and per-token nonlinear gate σ(W_s X), linear-complexity, attention-matching quality.
  Applied EXACTLY as the paper's sequence-to-sequence block, replacing the exemplar self-attention.
- Isolates N0058's confounds: (a) all 49 tokens kept (no part-pool info destruction); (b) capacity
  matched to 3.52M (no −42% cut); (c) residual path matched (Eq.3 residual kept, as N0054's
  norm_first+residual); (d) condenser untouched (consumer not re-swapped, unlike N0057); (e) ParTY
  part-pool explicitly EXCLUDED as the N0058 confound — this is a pure full-token operator morph.
- H0081's boundary: aggregation swap only, ≥1.5M trainable, 49 tokens, no part-pool. N0059 is exactly
  that control; whether it confirms, ties, or kills resolves the operator-class question H0081 leaves open.
- Paradigm: permutation-equivariant token mixing (moments sum-invariant + per-token gate + W_o mixing)
  at full capacity — a genuinely non-attention producer, the absent arm of the H0081 falsification.

## Hypothesis (verbatim)
**H0082** IF [param-matched non-attention operator (PoM-PolyMorpher, k=2, D=352, residual-kept,
49 tokens, no part-pool) replaces the exemplar producer self-attention in frozen N0054] THEN [val MAE
< 19.647] BECAUSE [full-token 2nd-order moments + nonlinear gating supply cross-token aggregation at
matched capacity → producer SELF-ATTENTION is not load-bearing, cross-token order is].
**DISPROVED IF best val MAE ≥ 20.40.**

## Gates (vs N0054 19.647; noise ±0.25; N=1 defensible for kill/ties)
- **R1 smoke**: stub backbone; use_gca/use_xscale/use_pom on; params ≤32M; density (B,1,96,96); loss
  finite & drops. Verify use_pom=False restores the exact N0054 attn path (single-switch).
- **R2 CONFIRM (H0081 DISPROVED)** if best val MAE **< 19.40** (non-marginal) — non-attention ≥
  attention at matched capacity ⇒ the attention-law capacity/expressivity claim is refuted;
  N0058's +2.17 was capacity+part-pool, NOT the operator class.
- **R3 TIE (operator exonerated; N0058 failure = capacity cut + 2×2 pool info destruction)** if
  **19.40 ≤ MAE ≤ 19.90** — require late-drop (E18-27) parity check vs N0054.
- **R4 KILL (non-attention under-expressive at n=49 is regime-law)** if **MAE ≥ 19.90** OR early-stop
  (ep16+ same-epoch ≥ N0054_same_epoch+1.5).
- Second-seed rule: a 2nd seed is MANDATORY only if first-seed best < 19.40 (before any CONFIRM
  booking); kill/ties are defensible at N=1.
