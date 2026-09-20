# N0004_h0003 — idea (Idea Agent, H0003 `use_padapt`)

- **Node:** N0004_h0003 · **Parent:** N0002_h0001 · **Booked hypothesis:** H0003 · **Switch:** `use_padapt` (child `true`; parent default `false`) · **Composition:** SOLO single-switch.
- **Falsifier bar:** final val MAE ≤ 22.26 at seed 20260830 under the canonical 32ep/1800s protocol (live parent N0002_h0001 = 22.5641 @ep30; confirm iff ≥0.30 lower).
- **Booked hypothesis text (verbatim — do NOT edit):**
- IF image-conditioned iterative prototype adaptation via cross-attention (use_padapt) IN a solo single-switch child of the simprior-bearing live parent with use_padapt=true, THEN final val MAE falls to 22.26 or lower versus live parent 22.5641 at seed 20260830 under the canonical 32ep/1800s protocol, BECAUSE two cross-attention refinement steps re-condition each pooled K=3 exemplar prototype on the specific image's frozen h2 feature tokens before the Condenser matches pixels to it, moving the class definition from a dataset-generic pooled average to the target instance's local appearance so matching happens against the right region of feature space, DISPROVED IF final val MAE under the canonical protocol is not at or below 22.26.

---

# Image-conditioned iterative prototype adaptation (`use_padapt`) — batch-2 F3 candidate

- **Switch:** `use_padapt` (config default `false`) · **Attachment:** query/similarity-side, between `ExemplarEncoder` and `Condenser` · **Composition:** SOLO single-switch child of the **simprior-bearing live parent**.
- **One-line intent:** after the `ExemplarEncoder` produces `e` (B,K=3,256), refine each exemplar prototype against the image's frozen h2 feature tokens with 1–2 cross-attention adaptation steps, then feed the **adapted** `e` into the existing `Condenser`; the pooled `e` that XScale, `e_mean`, GCA and SimPrior already consume is left untouched.
- **Live parent (bar binds here):** N0002_h0001 = **val MAE 22.5641 @ep30** (seed 20260830, canonical 32ep/1800s; `tree/N0001_champion/N0002_h0001/result.json:5-6`). Confirm iff child final val MAE **≤ 22.26** (≥0.30 lower).

---

## 1. Multi-angle reasoning

### (a) Pure-mathematics lens — adapting the class definition vs. mixing a fixed one

The head's matching operator is a cross-attention whose keys and values are the K=3 pooled prototypes: for pixel `i`, `a_i = Σ_k α_{ik} v_k`. Both `q` (pixels) and `k,v` (prototypes) are *fixed functions of the frozen features*, but the prototypes were produced by ROI-pooling h3 **at one instant, in isolation from the rest of the image**: `e_k = pool(tr(roi_align(h3, b_k))) + xproj(pool(roi_align(h3,b_k)))` (`model.py:97-108`). They are dataset-generic class estimates; nothing in them depends on the particular test image's object instances. LOCA's OPE is the opposite factorization: the prototype is a *variable* that is iteratively optimized against the image features before matching —

```
e_{t+1} = e_t + XAttn(norm(e_t), F, F),   t = 0..n_steps-1,
```

with `F` the dense image tokens. The residual is a nonlinear function of image content, so `e_adapted` can leave the span of the original pooled vectors and land in the region of feature space that the *current image's* instances occupy. A gate `e' = m ⊙ e` (the refuted family, §5 below) can only rescale/zero existing directions; it cannot add a direction. This is the mathematical content of "re-conditioning": the same matcher is now pointed at the right class region, so its 3-way convex combination is a convex combination over *instance-relevant* evidence. The aggregation is unchanged — only the prototypes entering it move.

### (b) Champion-lineage lens — file:line facts from `tree/N0001_champion/N0002_h0001/model.py`

- **Prototype bottleneck (`model.py:91-108`):** exemplars are ROI-aligned from h3@1/16 (`roi_align(feat, rois, 7×7)`, line 97), projected 384→256, shape-biased, passed through a 2-layer transformer, then softmax-pooled over 49 tokens into one vector per exemplar (`a = self.attn(tok).softmax(1); out = (tok*a).sum(1)`, lines 102-103), plus the XScale coarse average (lines 104-107). **All 49 within-exemplar tokens are collapsed before the head ever sees them.** The only object that reaches the matcher is `e` (B,K=3,256).
- **The matcher consumes exactly those 3 vectors (`model.py:121-125`):** `a,_ = self.attn(self.norm1(tok), e, e)` — K=3 keys and values. Whatever the prototypes say, the per-pixel match is a 3-way combination of them. There is no path by which the image can correct a prototype that was pooled badly for this image (e.g. all 3 exemplars small/blurry, or background-heavy boxes) — first-principles §1.5 documents the h3 ROI degeneracy for small exemplars (49 resampled tokens from ≤1.5 h3 cells).
- **Where the new module slots in:** `CountingHead.forward` computes `e = self.exemplar(h3, bboxes_in, self.S)` (line 166) and immediately `cond = self.cond(fmap, e)` (line 167). The adapter is a pure insertion on the edge `e → Condenser`: it takes the same `e` plus `h2` (already an argument of `CountingHead.forward`, line 161) and returns a re-conditioned `e_adapted` used **only** as the Condenser K/V. `fine`, `e_mean` (`e.mean(dim=1)`, line 174), GCA (`model.py:178-190`) and SimPrior (lines 171-172, 193-221) keep reading the original `e`/`fine` — so the confirmed H0001 path, XScale, GCA and the decoder are byte-identical.
- **Lineage evidence that the interface is the binding constraint:** XScale (coarse exemplar enrichment, +0.95, `README.md:12`) and GCA (+1.6) are both *global/coarse* additions. The one cell never occupied is a prototype that is a function of the test image. LOCA's ablation (below) says that is the highest-value cell in this design space.

### (c) Counter-intuitive / low-cost lens

- **Counter-intuitive:** the head *already* cross-attends image pixels to exemplars, so "adding adaptation on the exemplar side" looks like doing the same thing twice. It is not the same functional: the Condenser adapts the *queries* (pixels) to a fixed prototype; OPE adapts the *keys/values* (prototype) to the image. The Condenser is trained only through the density loss, so it can satisfy that loss with a prototype that is systematically biased and let attention compensate; OPE changes the target the attention is compensating toward. LOCA's evidence that this inversion is the load-bearing move: removing OPE costs **+6.00 val MAE** (10.24→16.24, Table 6), the single largest architecture ablation in the surveyed field (survey `survey_mechanisms.md:25,77-87`).
- **Low-cost:** ≈62.2k params (≈9.6% of the ≈650k headroom), ≈0.087 GFLOPs/forward — deliberately kept at SimPrior's own cost scale (survey F3 prints a 50–150k envelope), two 1×1/linear projections plus one 64-d cross-attention over 2304 h2 tokens, zero-init final projection so `use_padapt=true` starts numerically identical to the live parent (clean ablation: `false` reproduces N0002 exactly; `true` starts from N0002 and learns only the adaptation). One canonical run decides it.
- **The counter-evidence is priced in:** CounTR reaches SOTA with pooled exemplar vectors (`arXiv:2208.13721` §3.1.1), so the bottleneck may not bind; and LOCA's own Table 7 shows iterations beyond the first give only sub-linear returns (L=1 → 11.07 vs L=2 → 10.81 joined-set MAE). This card therefore claims a modest, falsifiable win (0.30), with a null that is informative rather than a dead end (§6).

---

## 2. Web-survey citations (RULE 15 — satisfied by the two completed reports; one claim re-verified live)

The required web survey is the completed researcher reports; I re-verified **one** claim directly against the primary source.

- **LOCA — Iterative Prototype Adaptation (OPE)** · ICCV 2023, arXiv:2211.08217 · code `github.com/djukicn/loca`. **Re-verified live** from the paper HTML (`arxiv.org/html/2211.08217v2`), Table 6: full LOCA val **10.24** / test 10.79; `LOCA_no_ope` val **16.24** / test 15.53 → **+6.00 val MAE** for removing OPE (the field's largest single architecture ablation); `LOCA_no_att` 11.99; `LOCA_no_shape` 13.77. Table 7 (iterations): L=1/2/3 → 11.07/10.81/10.50. §4.5: replacing the **first OPE cross-attention with a plain sum** (`Q1' = QA + QS`) costs **+5% MAE / +22% RMSE** — the adaptation must be *modulated* attention, not additive mixing. Frozen SwAV-ResNet-50, 512px, L=3, emb 256, 8 heads, s=3 prototypes (paper §4.1; repo README `--num_ope_iterative_steps 3`). **Confidence H.**
- **Survey family F3** · `local/research/survey_mechanisms.md:77-87` (switch `use_padapt`; `fmap8` = projected h2 tokens or fine; params 50–150k; `use_padapt=false` returns parent exactly) and §2 row 1 (`:25`). Trusted as pre-existing researcher output, **confidence H** for interface reasoning. F3 lists the exact failure modes reused here: adaptation can hallucinate target appearance from background (LOCA needs modulated/object-normalized attention), and register friction (pre-Condenser exemplar modification).
- **Survey §4 cautionary rows** · `survey_mechanisms.md:118-120` (learned metric swaps alone underperform; BMNet+ 15.74 vs LOCA 10.24) and `:143-145` (more exemplars can hurt; LOCA 1-shot 10.90 vs 5-shot 11.11 val). These bound the claim: this is prototype **adaptation under a fixed cosine-like matcher**, not a new learned metric and not exemplar mixing.
- **First-principles report** · `local/research/first_principles.md:190-194` (S2/S3 small-exemplar and 3-key bottleneck), `:178-188` (plateau/overfit and the 0.30 ≈ 386-counts arithmetic). Survey §5 argues F3 should be tested after F1/F4 because it changes *where* the comparison happens. Supporting, confidence M (diagnoses inferred, not measured per-image).
- **Contrast (what this is NOT):** SAFECount FE / similarity-weighted query enhancement = the banned pre-Condenser exemplar gating family (`survey_mechanisms.md:27,121-124`); LSSD FEM = banned spatial-summary family (`:125-128`); DDCA / extra spatial summaries / final-layer readout / unfreeze (`AGENTS.md:241`). None are touched.

---

## 3. Exact implementation spec for the Coding Agent

**New module class** (name fixed): `PrototypeAdapter(nn.Module)` in the future child node's `model.py` (copied from the parent's `tree/N0001_champion/N0002_h0001/model.py`; parent file untouched).

```python
class PrototypeAdapter(nn.Module):
    """LOCA-style image-conditioned prototype adaptation (`use_padapt`).

    Re-conditions the pooled K=3 exemplar prototypes on the current image's
    frozen h2 tokens with n_steps of cross-attention, then hands the adapted
    prototypes to the Condenser. Pure additive residual; the pooled e that
    XScale/e_mean/GCA/SimPrior consume is read-only here (never modified).
    Zero-init final projection -> step-0 forward is numerically identical to
    the simprior-bearing parent.
    """
    def __init__(self, d_model=256, d_attn=64, n_heads=4, n_steps=2, d_feat=192):
        super().__init__()
        self.n_steps = n_steps
        self.fproj = nn.Linear(d_feat, d_attn)              # 192 -> 64 (h2 tokens)
        self.eproj = nn.Linear(d_model, d_attn)             # 256 -> 64 (prototype query)
        self.norm = nn.LayerNorm(d_attn)
        self.attn = nn.MultiheadAttention(d_attn, n_heads, batch_first=True)
        self.out = nn.Linear(d_attn, d_model)               # 64 -> 256, ZERO-INIT
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, e, h2):
        # e: (B,K=3,256); h2: (B,192,48,48) -> (B,2304,192)
        F = self.fproj(h2.flatten(2).transpose(1, 2))        # (B,2304,64)
        et = e
        for _ in range(self.n_steps):
            q = self.norm(self.eproj(et))                    # (B,K,64)
            a, _ = self.attn(q, F, F, need_weights=False)    # (B,K,64)
            et = et + self.out(a)                            # zero-init -> identity at step 0
        return et                                            # (B,K,256)
```

**Feature-map choice and cost (stated):** `F` = projected tokens of **h2** (B,192,48,48 = 2304 tokens at S=384), not `fine` (96×96 = 9216 tokens, 4× the keys, and head-derived rather than a frozen interface). h2 is the frozen `hs(2,3)` interface (`model.py:40-43`), consistent with the standing regime; it is also 4× cheaper. The keys/values are frozen in the sense that they come from the frozen backbone (the small `fproj` is trained).

**Iteration count:** `n_steps = 2` (module constant; LOCA uses L=3; the card allows 1–2, and Table 7 shows L=1 already captures most of the gain). Weights are shared across steps (recurrent refinement), which is why the param count is independent of `n_steps`.

**Constructor order (append-only RNG, AGENTS §5 rule 13):** in `Counter.__init__`, after the parent's existing `self.head.simprior = SimPrior(...)` line (i.e. **after every parent module and after SimPrior**), append:

```python
self.use_padapt = _get(cfg, "use_padapt", False)  # non-module flag, no RNG
self.head.padapt = PrototypeAdapter(d_model=_get(cfg, "embed_dim", 256),
                                    d_attn=64, n_heads=4, n_steps=2, d_feat=dims[0])
```

`PrototypeAdapter` must be the **last** `nn.Module` constructed, so all N0002 draws (`Backbone` → `CountingHead` → `GCA` → `SimPrior`) keep their exact RNG positions and only `PrototypeAdapter`'s own init draws are appended. `use_padapt` is read as a plain attribute (also set `self.use_padapt` in `CountingHead.__init__` next to `use_simprior`, `model.py:158-159` style) — it consumes no RNG.

**Where adapted `e` re-enters (child `CountingHead.forward`, replacing `model.py:166-173`):**

```python
e = self.exemplar(h3, bboxes_in, self.S)
e_cond = e
if self.use_padapt:
    e_cond = self.padapt(e, h2)                 # (B,K,256) re-conditioned prototypes
cond = self.cond(fmap, e_cond)                  # Condenser K/V = adapted prototypes
cond_map = cond.transpose(1, 2).reshape(B, -1, Hf, Wf)
if self.use_simprior:
    cond_map = cond_map + self.simprior(fine, e)   # H0001 path unchanged: ORIGINAL e
dens = self.decoder(torch.cat([fine, cond_map], 1))
e_mean = e.mean(dim=1)                          # ORIGINAL e
return dens, fine, e_mean
```

- **Flag off (`use_padapt=false`):** `e_cond = e`, the `if` is skipped, and the forward is the N0002 parent's byte-for-byte — the child reduces exactly to the live parent.
- **Adapted `e` is used only by the `Condenser`.** `XScale` is computed *inside* `ExemplarEncoder.forward` before adaptation (so it is structurally untouched); `e_mean`/GCA and the confirmed SimPrior read the original `e`/`fine`. Shapes are unchanged everywhere: Condenser still takes keys `(B,K=3,256)` and `d_in=128` queries; decoder `in_ch` stays 192; GCA reads `GCA(D, d_model)` unchanged.
- **Design note (deliberate, single-switch discipline):** SimPrior is intentionally left on the original `e`, so the switch alters exactly one edge (`e → Condenser`). Feeding the adapted `e` to SimPrior as well is a follow-up ablation, never co-composed here.

**Zero-init plan:** `self.out` (final `Linear(64→256)`) has weight and bias initialized to exact zeros. Consequence: at step 0 each residual is identically zero regardless of `fproj`/`eproj`/`norm`/`attn` values, so `e_cond = e` and the child's `(density, n_aux)` are **numerically identical to N0002's** at init; training learns only the trust/magnitude of the adaptation. `fproj`/`eproj`/`norm`/`attn` keep default init (their effect is gated by the zeroed `out`). Precedent: GCA and SimPrior both use a zeroed final projection (`model.py:183-184,208-210`). Known consequence (same as SimPrior): the attention parameters receive zero gradient until `out.weight` moves off zero; this was empirically benign for the confirmed H0001.

**Config key and default:** `use_padapt = false` in `config.toml` (mirroring `use_xscale`/`use_gca`/`use_simprior`; read via `_get(cfg, "use_padapt", False)`). Child config = N0002 config + this one line set to `true` (and keeps `use_simprior = true`); parent config untouched.

**Param delta arithmetic** (all biases included; GN/LN affine included):
- `fproj` Linear(192→64): 192·64 + 64 = 12,352
- `eproj` Linear(256→64): 256·64 + 64 = 16,448
- `norm` LayerNorm(64): 64 + 64 = 128
- `attn` MultiheadAttention(embed_dim=64, heads=4): in_proj (3·64)·64 = 12,288 + in_proj_bias 192; out_proj 64·64 = 4,096 + bias 64 → 16,640
- `out` Linear(64→256): 64·256 + 256 = 16,640
- **Total Δ = 12,352 + 16,448 + 128 + 16,640 + 16,640 = 62,208 (≈62.2k).**

**Total arithmetic (as required):** N0002 total = frozen DINOv3-ConvNeXt-Tiny **27,820,128** (survey `:5-7`, HF safetensors metadata) + N0001 head **3,504,643** (first_principles §1.1, exact from source) + SimPrior **24,769** (N0002 `idea.md:90`) = **31,349,540**, matching the card's reported figure. Child total = 31,349,540 + 62,208 = **31,411,748 ≤ 32,000,000**; headroom used 62,208 of 650,460 (≈9.6%). No other module changes size.

**Budget note (honest):** N0002 ran 1755 s of the 1800 s cap (`result.json:8,11`) → only **~45 s headroom**. The module's forward cost is ≈**0.087 GFLOPs** (fproj 2304·64·192 ≈ 28.3M MACs; attention in_proj on F 2304·64·192 ≈ 28.3M/step ×2 steps; QKᵀ/AV ≈ 0.9M; out_proj negligible) — deliberately on the same scale as SimPrior's ≈0.08 GFLOPs (`idea.md:92`), which fit the cap. Risk: the observed +61 s (1694→1755 s) for SimPrior cannot be attributed to FLOPs alone (server variance is material), so per-epoch overhead is the watch item; expected impact ≲1–2 s/epoch. **Documented fallback if smoke shows >3 s/epoch:** set `n_steps=1`, which halves the attention cost (≈58M MACs total) with no param change. No `_BudgetStop` risk is guaranteed; this is flagged, not hand-waved.

**RNG-order statement:** the only new RNG consumption is `PrototypeAdapter` parameter init, appended strictly after all parent + SimPrior init draws (constructed last in `Counter.__init__`); `forward` contains no stochastic ops (no dropout; linear/LN/MHA/softmax deterministic); data-loader seeding (`seed=20260830`, `augment=false`) is unchanged. Parent-vs-child training divergence under the canonical harness therefore comes only from the learned adaptation, satisfying AGENTS §5 rule 13 paired-contrast semantics (precision ~0.02).

**Contract/register compliance:** `build_model(cfg) → forward(imgs, bboxes[, bboxes3]) → {"density", ...}` unchanged; only `out["density"]` feeds the loss; GCA/XScale/hs(2,3) readout intact. No gating anywhere — see the explicit distinction in §5.

---

## 4. Exact one-line hypothesis text (to be booked verbatim by the Lead)

IF image-conditioned iterative prototype adaptation via cross-attention (use_padapt) IN a solo single-switch child of the simprior-bearing live parent with use_padapt=true, THEN final val MAE falls to 22.26 or lower versus live parent 22.5641 at seed 20260830 under the canonical 32ep/1800s protocol, BECAUSE two cross-attention refinement steps re-condition each pooled K=3 exemplar prototype on the specific image's frozen h2 feature tokens before the Condenser matches pixels to it, moving the class definition from a dataset-generic pooled average to the target instance's local appearance so matching happens against the right region of feature space, DISPROVED IF final val MAE under the canonical protocol is not at or below 22.26.

*(738 chars; markers IF→IN→THEN→BECAUSE→DISPROVED in order; no ids embedded; gate outputs in §7.)*

---

## 5. Register distinction (AGENTS §11) — stated plainly, with residual risk

`AGENTS.md:242-245` bans **pre-condenser exemplar gating in any granularity** — image-global channel gate, frozen-input control, per-exemplar scalar gate, per-exemplar channel gate. This proposal is **not a gate**, and the distinction is mechanical, not rhetorical:

- A gate is `e' = m(e, x) ⊙ e`, where `m` is a bounded mask (global scalar/channel, or per-exemplar scalar/channel). It can only rescale or zero directions already present in `e`; it cannot introduce information the pooled prototype never had.
- This module computes `e' = e + out(XAttn(norm(eproj(e)), F, F))` with `F` = image h2 tokens. The residual is an **additive, nonlinear, image-content-dependent** term; `out` maps 64-d attention outputs into the full 256-d prototype space, so it can move `e` into directions outside its original span. Nothing multiplies the prototype or the frozen input; there is no mask, no scalar gate, no channel gate. It is image-conditioned prototype **adaptation**, exactly LOCA's OPE class of operator.
- `AGENTS.md:243-245` explicitly records that the **query/similarity-side direction is live** ("unproven, re-book fresh if wanted"). Adaptation of the prototype before matching is a query/similarity-side mechanism.
- **Residual risk (not glossed):** the module does modify `e` *before* the Condenser, so a strict reading of the register's caution about "pre-condenser exemplar modification" could flag it as friction; the survey itself records this (`survey_mechanisms.md:84-87`, F3 failure modes: "register friction: this is pre-Condenser exemplar modification. Not a gate ... but must be booked with the distinction stated, not silently"). I am stating it, not hiding it; the Lead should confirm the reading at booking. Independently of the register, the mechanism's own failure mode is real: adaptation can hallucinate target appearance from background (LOCA needs modulated attention), which is why the residual is softmax-attention-normalized and zero-init (the model can back off to the parent).

---

## 6. Falsification reading and what a null means

- **Confirm/refute rule:** run the child at the canonical protocol (seed 20260830, 32ep/1800s, EMA eval) as a same-seed paired contrast against the live parent N0002_h0001 (val MAE **22.5641 @ep30**). **Confirm** iff child final val MAE **≤ 22.26** (≥0.30 lower). Otherwise **refute** — including a child that matches the parent (adapter ignored) or regresses. No cross-seed comparison, no re-run shopping (AGENTS §5 rules 1/5/6).
- **What to inspect in the curve/checkpoint:** (i) best-epoch EMA val MAE and its epoch (parent best ep30; a real win clears 22.26 by ep32, not only at the edge); (ii) val RMSE alongside MAE (parent ratio ≈3.9; adaptation should move both if it fixes prototype-region errors); (iii) training `loss_epoch` vs val MAE divergence (parent overfits late; adaptation must move **val**); (iv) **use-evidence**: the learned `out` weight norm (0 at init) and the attention residual norm — a confirm shows nonzero `out` norm and non-uniform attention over h2; a refute with `out` norm ≈0 means the optimizer discarded the adapter (mechanism rejected, not merely under-tuned). Because `out` gates the gradient into `fproj`/`eproj`/`attn`, also inspect whether `out` moved at all — if it stayed exactly 0, the run is uninformative about adaptation capacity and must be reported as such.
- **LOCA's own evidence that adaptation needs *modulated* attention (failure mode named):** unmodulated prototype adaptation **hallucinates target appearance from background**. LOCA's Table 6 shows removing the image-wide self-attention block (which consolidates same-category features) degrades 10.24→**11.99** val, and §4.5 shows replacing the first OPE cross-attention with a plain sum costs **+5% MAE / +22% RMSE** — the adaptation must be attention-modulated, not additive mixing, precisely because unmodulated blending pulls background statistics into the prototype. Our module uses softmax-normalized cross-attention (the modulation) plus a zero-init output (the escape hatch); if it still hallucinates background, val will regress and the design is falsified on schedule.
- **What a null means:** a clean null (child ≈ parent or worse, `out` norm →0 or attention degenerate) falsifies **only** this variant — 2-step, 64-d cross-attention adaptation of *pooled* K=3 prototypes against h2 tokens. It does **not** falsify prototype adaptation generally. Still live after a null, in the order the survey would test them: (1) the bank is the bottleneck — replace pooled `e` with per-token exemplar keys (first-principles S3; survey F1/R2 analogue), then adapt those; (2) the attention width is the bottleneck — widen `d_attn` or adapt against `h3`+`fine` jointly; (3) shape conditioning dominates appearance in LOCA (no_shape 13.77 vs 10.24), so pair adaptation with richer shape queries; (4) the plateau lives in catastrophic high-count images (first-principles S1/D1), in which case a matching-side tweak cannot clear 0.30. Per AGENTS §11 a refuted variant is never silently retried — any revisit books as new evidence with a new falsifier.

---

## 7. Gate outputs (pasted)

`python3 scripts/novelty_check.py "<line>"`:
```
top similarities: H0002=0.319, H0001=0.317
NOVEL (top sim 0.319 < 0.82)
```
(exit 0)

`python3 -c "... from cac.expt.hypothesis import validate; print(validate(open('/tmp/padapt_text.txt').read()))"`:
```
LEN 738
([], [])
```
(no errors, no warnings)

---

## Booked hypothesis set (machine-readable)

1. **H0003** — IF image-conditioned iterative prototype adaptation via cross-attention (use_padapt) IN a solo single-switch child of the simprior-bearing live parent with use_padapt=true, THEN final val MAE falls to 22.26 or lower versus live parent 22.5641 at seed 20260830 under the canonical 32ep/1800s protocol, BECAUSE two cross-attention refinement steps re-condition each pooled K=3 exemplar prototype on the specific image's frozen h2 feature tokens before the Condenser matches pixels to it, moving the class definition from a dataset-generic pooled average to the target instance's local appearance so matching happens against the right region of feature space, DISPROVED IF final val MAE under the canonical protocol is not at or below 22.26.
