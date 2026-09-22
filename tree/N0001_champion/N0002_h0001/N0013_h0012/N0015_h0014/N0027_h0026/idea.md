# H0026 — GemeCal: zero-init geometry magnitude embedding scale on SimPrior similarity evidence (SOLO card; decoder-side reserved for in-flight H0025)

- Parent: `tree/N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014` (LIVE parent N0015_h0014, val MAE 19.3431 @ep32 v2 protocol, seed 20260830).
- SOLO card: exactly one mechanism switch `use_geme` (default false), shipped as a **single-line config delta** (`use_geme=true`; `use_simprior`/`use_cellcal`/`use_peakcal` stay `true`, every other key byte-identical). When the flag is on, `CountingHead.forward` computes a per-image geometry magnitude factor `f = clamp((me/32)^w_g, 0.25, 4)` from the annotation exemplar boxes (`me = mean_K(S²/(w_k·h_k))`, CACViT order-of-magnitude capacity, stop-grad geometry) and threads `geme_f` into all four `SimPrior` call sites; inside `SimPrior.forward`, after the unchanged cellcal/peakcal gains and before the unchanged zero-init `out` projection, `ev ← ev · f`. `w_g` is a **single zero-initialized learnable scalar** inside a new `GemeCal` module. Similarity substrate, gain forms, sign handling, decoder, Condenser, backbone: byte-identical. **+1 parameter, +0 effective RNG draws** (`torch.zeros(())` does not consume RNG), constructed AFTER every parent module (append-only RNG). Flag off → new branch never taken, executed lines byte-identical to parent.
- Regime: frozen backbone, head-only training. Decoder `in_ch` untouched (still `D + cond_dim` = 192); optimizer/loss/schedule invariant (AdamW, MSE+w_cnt·L1, cosine, AMP).
- Protocol: v2 (augment=true, EMA eval, seeded loaders), seed FIXED 20260830, 32ep/1800s. Same-seed paired contrast vs live parent N0015 only.
- Triple conjunctive bars (§6 semantics, bound to live parent 19.3431 under v2): THEN final EMA val MAE ≤ 19.0431 (≥0.30 below 19.3431) AND dense-tail gt>500 below 330.81 AND sparse gt<50 at or below 5.45. Missing any one bar refutes the card. Launch carries `--set futility_bar=19.0431` (see §5).
- **Why THIS card now (reexamination mandate premise, not re-litigated):** the reexamination (2026-09-22) authorized the decoder/consumption fork AND left similarity-side open EXCEPT gain-domain masks. H0025 CapFilm immediately booked the optimal decoder-consumption site (post-GN pre-head FiLM, in-flight as N0026) — a second decoder amplitude card would be twin-dominated at the gate AND collapse the batch onto one failure mode (shared quiet-null). H0024 (response-z fixed 1/z post-out residual scale) REFUTED today (N0025 futility-halted ep24, val 22.2205, contradicts w=0.85) — closing the (response × post-out) cell of the magnitude fork. The remaining open cell with the higher-leverage substrate is **(geometry × similarity-side pre-out)**: the leverage probe measures the SimPrior residual as net-suppressive (remove → +29.9% count), the S-curve is an unmodelled order-of-magnitude covariate (gt7–10 ratio 1.40 → gt138+ 0.685, mid50–300 0.86 carrying 44% AE), and CACViT independently names normalization destroying order-of-magnitude on similarity scores — the exact pathway, with published code. Booking H0026 completes the three-site localization trilogy for the paper: similarity-side (this card) / decoder-side (H0025, in-flight) / residual post-out (H0024, refuted).
- **Honest expectation:** expected negative-to-marginal (HANDOFF discipline). Determinism crisis sd≈1.13; mechanism reads (R0/R1/R2) weighted above level bars where they conflict (user ruling pattern from H0024/H0025).

## §0 — Bookable one-liner (for `discovery hypo --new`)

IF a zero-init geometry magnitude embedding scale (`use_geme`) multiplies the SimPrior similarity evidence tensor ev = cat[top1, cons] by a per-image factor f = clamp((me/32)^w_g, 0.25, 4) where me = mean_K(S²/(w_k·h_k)) is the exemplar-box order-of-magnitude capacity (S=input size, each annotation box structurally holds one object) and w_g is a single zero-initialized learnable scalar, applied after the unchanged cellcal and peakcal gains and before the unchanged zero-init out-projection, while the similarity substrate, the gain forms, the decoder, and every other parent line stay byte-identical, IN the live N0015 head on FSC147 v2 at seed 20260830, THEN final EMA val MAE drops at least 0.30 below 19.3431 with dense-tail gt>500 below 330.81 and sparse gt<50 at or below 5.45, BECAUSE the measured S-curve miscalibration (gt7-10 ratio 1.40 over to gt138+ ratio 0.685 under, mid50-300 at 0.86 carrying 44 percent of AE) is an order-of-magnitude capacity covariate missing from a scale-free similarity pathway (cosine top1 in [-1,1], mean-1 cellcal gain, zero-init linear out) that the leverage probe measures as net-suppressive (removing the residual raises count 29.9 percent), so exemplar-box geometry me supplies the missing count-capacity covariate while the zero-init learnable w_g lets the loss set the sign that attenuates the suppressive residual on high-capacity images and reinforces it on low-capacity images, flattening the S-curve on the similarity side left open by the reexamination, without any decoder edit and without any gain-form, mask, or substrate change. DISPROVED IF final EMA val MAE is not at least 0.30 below 19.3431, or dense-tail gt>500 is not below 330.81, or sparse gt<50 exceeds 5.45, or exemplar-box magnitude carries no count information on val (Pearson corr of log(me) with log(1+gt) below 0.20), or the scale fails its engagement read (val coefficient of variation of f below 0.05, i.e. w_g collapsed to zero or me too narrow), or the capacity loading runs the wrong way (mean f on gt at least 300 not at most 0.90 times mean f on gt below 50).

## §1 — Survey: magnitude embedding / scale priors on counting similarity pathways (verified IDs with URLs; full log in `h0026_survey.md`)

Survey date 2026-09-22. Method: live web search + arXiv/AAAI/publisher/repo verification of every cited ID (AGENTS rule 15). Focus: has anyone multiplied an exemplar-box **order-of-magnitude capacity prior** (pure geometry, model-independent) onto a **trained similarity-evidence pathway** inside a pluggable frozen-head counting model, with a learnable sign, to fix a per-image S-curve? Verdict: no matching in-head mechanism; the covariate and the multiply-on-similarity actuator both exist in CACViT (encoder/attention path), decoder self-norm exists as SNDM/DensFiLM (H0025's fork, in-flight), response-mass normalization exists only training-free (CountingDINO/CounTR).

1. **CACViT: Vision Transformer Off-the-Shelf … Few-Shot Class-Agnostic Counting** — arXiv:2305.04440 (Wang, Xiao, Cao, Lü; AAAI 2024, DOI 10.1609/aaai.v38i6.28396). URL: https://arxiv.org/abs/2305.04440 · https://ojs.aaai.org/index.php/AAAI/article/view/28396. **Code: https://github.com/Xu3XiWang/CACViT (verified; same MAE-pretrained weights as CounTR).** Verified (abs + AAAI PDF + HTML + repo). Core (§Magnitude Embedding, eq. 3): `ME_k = (W_p·H_p)/(W_k·H_k)` — patch area over exemplar-box area = maximum patch capacity conditioned on exemplar k; mean over K; **"we multiply the embedding with the similarity scores from the attention map to obtain the final similarity map"** — because "the softmax used to normalize the attention map will weaken the ability to express the number of objects… results in the information loss of the order of magnitude." Ablations (Table 4): ME alone weak on MAE (RMSE +6.2/+7.35; "marginal or even declines"); **ME + SE together: +21.36% MAE / +41.05% RMSE test (B8 vs B10)** — paper's own reading: SE restores instance scale at input, ME restores order-of-magnitude at similarity output. **Relevance:** independent naming of our exact scale-free finding; ME is pure exemplar-box geometry already in `bboxes_in`; published actuator is a **multiply on similarity scores** — Candidate A's literal ancestor; our head has the SE-analog (`ExemplarEncoder.shape_mlp(wh)`) and lacks only the ME half of the documented synergy. **Difference / why direction is learned here:** CACViT's similarity integrates **directly** to density (excitatory); our residual is measured **net-suppressive** (+29.9% when removed), so their fixed positive direction would amplify suppression exactly where we undercount — H0026 keeps the ME covariate and makes the sign `w_g` learnable; site is the trained SimPrior readout, not ViT encoder tokens.
2. **TasselNetV4** — arXiv:2509.20857 (2025-09-25). URL: https://arxiv.org/html/2509.20857v1. Code promised: https://github.com/tiny-smart/tasselnetv4. Verified (HTML). Builds on CACViT (extract-and-match ViT + aspect-ratio scale embedding); restates ME as eq. 6; adds box-aware local counters switched by exemplar scale. **Relevance:** CACViT magnitude/scale family alive through 2025; exemplar geometry as count-capacity prior is publishable. **Difference:** routes scale to input tokens / branch selection (encoder side, out of regime); H0026 multiplies geometry-derived capacity onto an existing trained similarity readout.
3. **SATCount** — Neural Networks vol. 172 (2024) 106126. Code: https://github.com/nercms-mmap/SATCount (verified; CounTR-style env, FSC-147 scripts + weights). Fuses exemplar **scale** with visual features via cross-attention. **Relevance:** code-backed 2024–25 confirmation exemplar scale is first-class. **Difference:** scale enters feature fusion; H0026 never touches features — only post-gain evidence amplitude.
4. **SNDM: Self-Normalized Density Map** — arXiv:2203.09474 (Graczyk et al., Sci Rep 12:10583, 2022). URL: https://arxiv.org/abs/2203.09474 · https://www.nature.com/articles/s41598-022-14879-3. Verified (abs + Nature full text). U²-Net fails "not [at] localization… but rather the normalization of density maps for a large range of possible outcomes"; bypass from smallest encoder block predicts per-image β rescaling pre-sigmoid logits `σ(β·S_i)` with penalty ½(1−β)². **Code: no standalone repo; base https://github.com/xuebinqin/U-2-Net.** **Relevance:** published statement of our S-curve pathology; cure is a network-internal per-image scale — the decoder fork **already booked as H0025 and in-flight as N0026**. **Difference:** decoder-output site is occupied by CapFilm (richer form: per-channel FiLM γ,β); H0026 is the orthogonal similarity-side cell — a second decoder self-norm now would twin H0025 at the gate and share its failure mode.
5. **DensFiLM** — arXiv:2607.25465 (2026-07-28). Code: https://github.com/aniskhan25/crowdfix-saliency (verified). Zero-init FiLM at Video-Swin bottleneck from density embedding (~100K params); "lightweight bottleneck conditioning… more effective… than increasing model capacity." **Relevance:** freshest code-backed proof density-conditioned decoder FiLM is the right inductive bias — **confirms H0025 owns the decoder fork**; second FiLM/self-norm card now = twin. **Difference:** saliency, density-class labels; not our site.
6. **CountingDINO** — arXiv:2504.16570 (WACV 2026). Code: https://github.com/lorebianchi98/CountingDINO. **Reproduced in-lab** val 39.70. Unit-mass via in-box response `z` (eq. 1–2). **CounTR** test-time normalization: `https://github.com/Verg-Avesta/CounTR` divides pred by in-box `e_cnt` when >1.8; **CountGD** (NeurIPS 2024, https://github.com/niki-amini-naieni/CountGD) inherits the SAM variant. **Relevance:** response-mass normalization family — H0024's source, **REFUTED today in-head** (fixed 1/z, post-out, N0025 val 22.2205). **Difference:** H0026 is geometry-sourced (boxes' areas, model-independent, train-time), learnable sign, pre-out site — the untested (geometry × pre-out) cell, not a re-encode of the refuted (response × post-out) cell; no test-time hack.
7. **FiLM** — arXiv:1709.07871 (Perez et al., AAAI 2018), code https://github.com/ethanjperez/film. Recorded for the decoder-fork contrast (H0025's operator); H0026 does not use FiLM at all.
8. **Rejected in survey (recorded):** UpCount (arXiv:2607.16826, https://github.com/r28112072-rgb/upcount — per-image count-calibration factor but reference-free, wholesale head); EBC-ZIP (arXiv:2506.19955 — loss-side NLL, loss ban); Count2Density (arXiv:2509.03170 — training signal); L2HCount (arXiv:2503.12935 — data synthesis, protocol-invariant); Learn to Scale/AutoScale (arXiv:1907.12428 / arXiv:1912.09632, https://github.com/dk-liang/AutoScale — crop + re-predict, extra spatial summary); DCA-MoE (arXiv:2608.15213 — router experts); CoDi (latent diffusion); DAVE/DecideNet (hybrid regime split, err-err 0.566 dead).

**What the survey decides (Candidate A vs strictly-better decoder card):** (i) the ME covariate and the multiply-on-similarity actuator are published (CACViT AAAI'24 + code) and remain active (TasselNetV4 2025) — but never on a trained suppressive residual with a learned sign, never in a pluggable head; (ii) the decoder self-norm/family (SNDM, DensFiLM, FiLM) is exactly H0025 CapFilm — booked, in-flight as N0026, occupying the strictly-dominant consumption site (post-GN pre-head); any second decoder amplitude card is novelty-twin of H0025 (shared FiLM/zero-init/DensityDecoder/pre-final-conv vocabulary), information-redundant (one family-level MSE veto kills both without distinguishing sites), and cannot non-twin while N0026 runs; (iii) reexamination options: C cond-FiLM rides the measured-weak cond path (sens 10× below fine, cond already suppressed) with the same FiLM-family twin risk; exemplar-distinctness is encoding-layer, H0019 diversification already failed, expected-negative without S-curve targeting. **Strictly-better test fails for every decoder-side alternative: Candidate A wins** — orthogonal site (similarity vs decoder), completes the localization trilogy, +1 param, published covariate, and a mutually-unsatisfiable falsifier family vs H0025 (f-load direction DOWN vs γ-load UP; corr(log me, gt) information gate that no FiLM card has).

First-principles lens (pure math): the readout computes `residual = out(cellcal_g(peakcal_g(cat[top1, cons])))` where top1 ∈ [−1,1], cellcal is a relative multiplier of a mean-1 field, out is zero-init linear — **no term in the chain can express a per-image order-of-magnitude**; under MSE on a 69%-sparse training distribution the Bayes act compresses E[ĉ|feat] into an S-curve. Multiplying `ev` by `(me/32)^w_g` adds exactly one continuous absolute DOF per image, parameterized as a power law (the natural group for order-of-magnitude), sourced from data (box areas) not from a predicted count (H0010 dead), with `w_g=0` giving exact identity and `∂f/∂w_g = f·log(me/32) ≠ 0` whenever me ≠ 32 — the gate escapes zero iff MSE rewards capacity scaling (H0009 lesson: nonzero gradient at init). Champion-lineage lens: cellcal confirmed a *relative* gain (+0.55); H0024 transplanted a *response-based absolute* scale post-out and died (refuted today); H0025 claims the *decoder* amplitude; the untested quadrant is **geometry-based order-of-magnitude at the similarity evidence layer, learnable sign** — orthogonal to both. Counter-intuitive/low-cost lens: one scalar parameter and one multiply on an already-trained pathway — the cheapest legal capacity covariate injection possible; if MSE never wants it, `w_g → 0` and the card self-quiet-nulls with a full diagnostic (R0 says whether ME even carries count information on val — a dataset fact worth having either way).

## §2 — Exact mechanism: GemeCal — zero-init power-law scale on SimPrior evidence

### 2.1 The measured scale-free flaw (raison d'être)

Parent N0015 path (model.py:225–252 + CountingHead 175–186):

```
q, p → S (cosine, [-1,1]) → softmax/temp → top1, cons → ev = cat([top1, cons])
     → cellcal: ev *= exp(w_c · log1p(mhat))        # RELATIVE gain, mean-1 mhat
     → peakcal: ev *= g (≈1, w_p≈0.0017)
     → return self.out(ev)                           # zero-init Linear(2,64)
# CountingHead: cond_map = cond_map + residual; decoder: cat[fine, cond_map] → density
```

Three measured facts (reexamination + leverage probe):

1. **Leverage:** remove SimPrior residual → count **+29.9%** (net-suppressive); fine=0 → −91.6%; cond sens ~10× below fine. The similarity pathway's amplitude is a live lever — not dead code — but its sign runs against dense counts as trained.
2. **S-curve (val n=1286):** gt7–10 ratio 1.40 → gt138+ 0.685; mid50–300 0.86 = 44% AE (n=353, MAE 30.83); sparse<50 1.06; 11 wipes extreme tail; lifting 39 imgs (ratio<0.5, gt≥100) to 0.7 ⇒ −3.97 MAE. The miscalibration is monotone in gt — an **order-of-magnitude capacity** miss, not a phase/sign miss (H0022) and not a substrate miss (H0023), both refuted 0/11.
3. **Scale freedom:** every term above is bounded (cosine), mean-1 (cellcal arg), or relative — no channel encodes "this image's exemplar boxes imply ~500 objects." Annotation geometry `bboxes_in` (K boxes, one object each) already reaches the head and is already consumed for exemplar encoding (`shape_mlp(wh)` = CACViT's SE-analog) — but never as an order-of-magnitude factor on the similarity readout.

### 2.2 GemeCal exact math

New module (class lives after `PeakCal` in model.py for append-only style; constructed only when flag on):

```python
class GemeCal(nn.Module):
    """H0026 geometry magnitude embedding scale: zero-init w_g -> f = (me/32)^w_g.
    ME = mean_K(S^2 / (w_k * h_k)) from annotation boxes (CACViT geometry).
    torch.zeros(()) draws no RNG; step-0 w_g=0 => f=1 exact.
    """
    def __init__(self):
        super().__init__()
        self.w_g = nn.Parameter(torch.zeros(()))

    def forward(self, bboxes, img_size):
        wh = (bboxes[:, :, 2:4] - bboxes[:, :, :2]).clamp_min(1.0)
        area = (wh[..., 0] * wh[..., 1]).clamp_min(1.0)          # (B, K)
        me = (float(img_size) * float(img_size) / area).mean(dim=1)  # (B,) capacity prior
        me = me.clamp(1.0, 1.0e4)
        f = torch.exp(self.w_g * (torch.log(me) - math.log(32.0))).clamp(0.25, 4.0)
        return f                                                  # (B,), == 1 at init
```

`SimPrior.forward` gains optional kwarg — `__init__` **unchanged** (no params inside SimPrior → no parent RNG shift):

```python
def forward(self, ..., geme_f=None):
    ... # parent cascade byte-identical through peakcal
    if geme_f is not None:
        ev = ev * geme_f.view(-1, 1, 1, 1)   # post-gain, pre-out; step-0 f=1 => ev unchanged
    return self.out(ev)
```

`CountingHead.forward` (mirror of the shipped N0025 threading pattern):

```python
geme_f = None
if self.use_geme and getattr(self, "gemecal", None) is not None:
    geme_f = self.gemecal(bboxes_in, self.S)
# all four simprior(...) call sites gain: geme_f=geme_f
```

**Why this is not any ban:**
- **Not a gain-domain mask (§11):** no sign test, no `where`, no suppression of any cells — `f` is a single positive scalar per image applied uniformly to all of `ev`; cellcal's sign handling byte-identical.
- **Not H0018 lingain (gain form):** cellcal's `exp(w_c·log1p(mhat))` form untouched; `f` multiplies *after* the gain cascade, never inside it.
- **Not H0024 re-encode (rule 11):** different source (annotation geometry, model-independent, train-available — vs response `z`, model-dependent), different site (pre-`out` `ev` vs post-`out` residual), different operator (learnable power law `w_g` vs fixed clamp `1/z`), different parameters (+1 learned vs zero), different falsifier (corr(log me, gt) information gate + f-load direction vs alpha wipe-median). H0024's own ledger note routes the *response* fork to death and points forward at the consumption/similarity forks — this is the similarity fork with a new covariate. If the Lead nevertheless rules rule-11 collision, the card dies at review **with its own new falsifier already registered** — never a silent re-encode.
- **Not post-decoder additive / decoder edit:** nothing after `decoder(...)`; DensityDecoder byte-identical — CapFilm (N0026) owns that site untouched by this card.
- **Not pre-condenser exemplar gating:** reads `bboxes` geometry only; `e`, Condenser, queries untouched.
- **Not transfer-by-learned-fusion:** `me` is annotation geometry already in `bboxes_in`, not external evidence; no fusion conv.
- **Not learned count surrogate (H0010):** `f` is a fixed-form power law of box areas with one exponent; no count head, no regime split (H0015), no global scalar (H0002 — `f` varies per image, CV gate ≥ 0.05 unsatisfiable by a constant).
- **Not trainable temperature / shared projections:** temp-pin path unchanged; `GemeCal` has no shared weights with simprior/qproj/kproj (own single Parameter).
- **Not DDCA/RGA/final-layer readout:** no new spatial summary; no second count head.

**Gradient / dead-gate check (H0009):** at `w_g=0`, `∂L/∂w_g = ∂L/∂ev · ev · log(me/32)` — generically nonzero on any image with me ≠ 32 as soon as `out` leaves zero (which it does from step 0 via live gradients). The scale escapes zero iff the loss rewards capacity-dependent reweighting; otherwise `w_g → 0` and the card self-refutes as a clean quiet-null.

**Params:** +1 (`w_g`). One non-module flag `use_geme`.

### 2.3 Twin pre-emption (novelty gate will probe H0024/H0025 first)

- **vs H0024 CountAnchor (REFUTED 2026-09-22, N0025 val 22.2205):** H0024 multiplied the **post-`out` residual** by fixed `clamp(1/z,…)` from **response** mass; H0026 multiplies **pre-`out` `ev`** by `(me/32)^w_g` from **geometry**, learned sign, +1 param. Source/site/operator/params/falsifier all differ (see §2.2 and §6). H0024's failure mode (in-box self-match flat → alpha ≈ 1, or wrong-direction starvation) does not apply: `me` depends only on box pixel areas — it varies by construction across the S-curve regardless of model state.
- **vs H0025 CapFilm (IN-FLIGHT N0026):** CapFilm is per-channel FiLM (γ,β) on the **decoder hidden** from GAP(fine)+logME; H0026 is a per-image scalar on **SimPrior `ev`** from geometry only (no `fine` read). Different tensor, layer, operator family (FiLM affine vs power-law scalar), falsifier direction (γ must load capacity UP ≥1.5× vs f must load DOWN ≤0.90 — opposite because CapFilm raises the generator's operating point while GemeCal attenuates the suppressive residual). If N0026 and this card both run, they test **complementary delivery sites for the same published prior** — paper-grade localization, not duplication; if the gate still rules textual twin, §5 fallback is exemplar-distinctness, booked fresh with its own falsifier.
- **vs H0016/H0017 additive energy:** no new projection into cond, no energy substrate, no additive path; quiet-null signature (proj norm → 0) does not apply — `w_g` scales an existing trained pathway the loss already drives.
- **vs H0018/H0022 evidence layer (mask/gain form):** gain form byte-identical; no sign mask; uniform positive scalar.
- **vs H0010/H0015/H0002:** no predicted count, no regime split, no global constant — per-image `f` from per-image `me`, CV gate ≥ 0.05.
- **vs Family C / pre-condenser gating:** `e` never read; cond path untouched.
- **vs decoder-side alternatives (the strictly-better test):** see §1 decision — every decoder amplitude card either twins H0025 at the gate or rides the weak cond path; none dominates CapFilm's site+form while N0026 runs.

### 2.4 Attach points (file:line against live parent `…/N0015_h0014/model.py`)

| # | Site | Change |
|---|---|---|
| 1 | `SimPrior.forward` model.py:225 | signature `forward(self, …, geme_f=None)`; insert `if geme_f is not None: ev = ev * geme_f.view(-1,1,1,1)` immediately before `return self.out(ev)` (model.py:252). Body :227–251 untouched. |
| 2 | `CountingHead.__init__` model.py:163 (beside `use_peakcal`) | `self.use_geme = _get(cfg, "use_geme", False)` — non-module flag, no RNG. |
| 3 | `CountingHead.forward` model.py:175–185 | before cascade: if flag and `self.gemecal` exists → `geme_f = self.gemecal(bboxes_in, self.S)` (`bboxes_in` in scope :165); thread `geme_f=geme_f` through ALL FOUR `self.simprior(...)` sites (cardinal N0023 lesson: never exclusive-dispatch parent gains away). Flag off → `geme_f=None` → parent lines. |
| 4 | New class `GemeCal` after `PeakCal` model.py:281 | module body (§2.2). |
| 5 | `Counter.__init__` after PeakCal attach model.py:312 (before temp-pin :315) | `if self.head.use_geme: self.head.gemecal = GemeCal()` — **AFTER every parent module** (append-only RNG, rule 13); `torch.zeros(())` draws no RNG, so parent init stream is bit-identical even with flag on at construction. |
| 6 | Config `config.toml` | one new line `use_geme = true` at run time (`--set`); seed/config otherwise byte-identical. |

Untouched: `Backbone` :22–48, `FineFuser` :51–74, `ExemplarEncoder` :77–108, `Condenser` :111–125, `DensityDecoder` :129–140, `GCA` :191–203, `CellCal`/`PeakCal` math, qproj/kproj/temp :218–230, temp-pin :315–316, `Counter.forward` GCA bias path :332–335.

### 2.5 Step-0 identity + guards + RNG

- **Flag off:** `geme_f=None` → SimPrior executes parent lines → byte-identical to live N0015 at every step, fresh or trained.
- **Flag on at init:** `w_g=0` ⇒ `(me/32)^0 = 1` for all me > 0 ⇒ `clamp(1, 0.25, 4) = 1` exact ⇒ `ev * 1.0 == ev` (fp16-safe) ⇒ `out(ev)` parent-identical; additionally `out` is zero-init so the residual itself is 0 regardless — double identity.
- **RNG:** `GemeCal` constructed at `Counter.__init__` end after PeakCal; its only tensor is `torch.zeros(())` — **consumes zero RNG draws**; parent module init stream bit-identical (strongest rule-13 form, same as H0024's zero-module claim, but retaining +1 learnable param).
- **Guards:** `area.clamp_min(1)`, `me.clamp(1,1e4)` before log; `f.clamp(0.25, 4.0)` finite for any finite `w_g`; no division by learned quantity, no softmax, no temp, no new normalization layers (train/eval agree).
- **Smoke (Coding Agent):** flag-off max-abs-diff vs parent == 0; flag-on-at-init == flag-off == 0; param count delta == +1; temp still 0.07 pinned; `use_peakcal=true` retained.

## §3 — Why not (every exhausted/rejected family, with the leverage-probe premise)

- **Any decoder/consumption amplitude card (Family: SNDM literal, second FiLM, output β):** H0025 CapFilm already books the optimal site (pre-final-conv, per-channel, zero-init) and is **in-flight as N0026**; a second card there is novelty-twin + information-collapse (shared failure mode). Candidate A takes the orthogonal open site instead.
- **Reexamination option C (cond-FiLM):** rides the measured-weak suppressive cond path (cond sens 10× below fine) — strictly lower leverage than acting on the similarity residual the probe measures at +29.9%; same FiLM-family twin risk vs H0025.
- **More residual/evidence content (H0016/17/18/22/23):** content edits normalized away (quiet-null) or failed wipe rescue (0/11); sign masks and substrate swaps refuted. H0026 adds **no content** — only a scalar reweight of the existing evidence tensor.
- **H0024-style response-z post-out (REFUTED today):** closed cell (response × post-out); re-running it violates rule 11. H0026 changes source, site, operator, and falsifier (§2.3).
- **Fixed-direction CACViT literal (hard-coded positive ME multiply):** our residual is net-suppressive — fixed positive `f` would amplify suppression on high-ME (dense) images where we already undercount; learnable `w_g` is strictly more general and lets the loss decide (published covariate, our sign).
- **Post-decoder ADDITIVE (any):** STATE ban. CapFilm/H0026 both act pre-decoder.
- **Global calibration scalars / H0002 / H0010 / H0015:** dead; falsifier requires CV(f) ≥ 0.05 + capacity loading + corr information gate — unsatisfiable by a scalar or a predicted count.
- **Loss / optimizer / schedule / unfreeze / DDCA / RGA / final-layer readout / temp reopen / shared kernels / pre-condenser gating / transfer-by-fusion / gain-domain masks:** all §11 or standing bans — none present.
- **Hybrid / second counting method / exemplar-distinctness as primary:** hybrid dead (err-err 0.566); distinctness is encoding-layer, H0019 diversification failed, held only as novelty-gate fallback (registered, expected-negative).

## §4 — Predictions, failure modes, mechanism reads (R0–R3)

Verdict context: live parent N0015 v2 **19.3431 / dense 330.81 / sparse 5.814** (bar 5.45 standing miss); S-curve ratios above; leverage +29.9%; H0024 refuted (22.2205); H0025/N0026 in-flight (do not attribute across).

- **R0 — ME CARRIES COUNT INFORMATION (dataset mechanism gate, new to this card):** on val, Pearson corr of `log(me)` with `log(1+gt)` ≥ **0.20**. If annotation-box geometry does not track count capacity on FSC147, the covariate is dead on arrival regardless of bars — honest mechanism refutation, and a reusable dataset fact. (Computed at eval time from `bboxes_in`; requires val annotations, which exist for the val split.)
- **R1 — SCALE ENGAGES (mechanism read):** on val @ best.pth, coefficient of variation of per-image `f` ≥ **0.05** (not collapsed: `w_g ≈ 0` or me too narrow fails); also report `w_g` value and `f` histogram (clamp pile-up at 0.25/4.0 = F4).
- **R2 — CAPACITY LOADING, DIRECTION COMMITTED (the S-curve claim):** mean `f` on gt≥300 ≤ **0.90 ×** mean `f` on gt<50 — high-capacity images get **attenuated** suppressive residual (less suppression where we undercount), low-capacity get reinforced (helps the sparse-over side too: sparse ratio 1.06 → toward 1.00 while bar allows ≤5.45 from parent 5.814). Wrong-direction loading REFUTES even if bars flicker. Supporting reads (arbitration, not gates): dense gt≥300 ratio moves from ~0.685 toward ≥0.72; mid50–300 from 0.86 toward ≥0.88; sparse must not cross 5.45.
- **R3 — BARS:** final EMA val MAE ≤ 19.0431 **AND** dense-tail gt>500 < 330.81 **AND** sparse gt<50 ≤ 5.45 (same-seed v2 paired vs live N0015: 19.3431 / 330.81 / 5.814).

Failure modes (each an honest refutation):

- **F1 — QUIET NULL:** `w_g` stays ≈0 (MSE never rewards geometry scaling in 32ep) → R1 fails → REFUTE; diagnostic: similarity-side absolute magnitude is unnecessary under joint MSE — successor points to decoder fork (H0025's verdict) or distinctness fallback.
- **F2 — UNINFORMATIVE COVARIATE:** R0 corr < 0.20 → mechanism dead regardless of level noise; report as dataset-level refutation (boxes don't track capacity on this corpus).
- **F3 — WRONG-DIRECTION LOAD / SPARSE BLOWUP:** `f` loads UP on dense (amplifying suppression) or sparse > 5.45 → REFUTE; the committed attenuate-on-dense direction is falsified in-head even though CACViT's excitatory direction works in their architecture.
- **F4 — CLAMP SATURATION:** `f` piles at 0.25 or 4.0 (`|w_g|` too large or me out of range) → effective hard gain clip (H0018-adjacent overshoot risk); read histogram, report; bars + R1/R2 still arbitrate.
- **F5 — ENGAGE BUT WRONG FORK:** R0–R2 hold but R3 bars fail on level noise / mid AE → REFUTE as booked; mechanism-true + level-miss under determinism crisis noted for Lead's evidence weighting (user ruling pattern).
- **F6 — LATE CROSSOVER:** if curve crosses parent ~ep16 like N0023 → ep24 futility HALT path; verdict = futility refutation + full eval_test; R0–R2 still reported on best.pth.
- **F7 — N0026 COLLATERAL:** CapFilm child is a direct child of N0015; do not attribute decoder-side effects to GemeCal; this card's child is also a **direct child of N0015**, never of N0025/N0026.

## §5 — Risks, v2 pair protocol, futility

- **Dominant risk — F1 quiet null / F2 weak covariate** (priced): 32ep may leave `w_g` small; R0+R1 gates make this a clean mechanism refutation, not a mystery. Expected-negative-to-marginal stance recorded in header.
- **Secondary:** (a) determinism crisis sd≈1.13 — single-seed, mechanism reads weighted (user ruling); (b) `f` interacts with cellcal's relative gain multiplicatively — both are scalars on `ev`, so identification relies on `f` being geometry-driven and `g` being mhat-driven; report both CVs; (c) sparse 5.45 standing miss may fail independently → truthful card refutation isolating sparse; (d) if Lead reads §11 "gain-domain" to ban ANY similarity-side amplitude edit (not just masks), card dies at review — flagged explicitly in §6; the reexamination text as surveyed says similarity-side open EXCEPT gain-domain masks, and `f` is not a mask.
- **Protocol:** SAME-SEED v2 pair — child (`use_geme=true` over parent flags, `--set augment=true`) vs live N0015 v2; seed 20260830 FIXED; 32ep/1800s; EMA eval; never cross-protocol; `repro_run --set seed=` FORBIDDEN.
- **Futility (rule 16):** launch `--set futility_bar=19.0431`; ep16 WARN / ep24 HALT (best > 20.00 ⇒ HALT); HALT in (20.0, 21.6) draw-contaminated — arbitrate R0/R1/R2 first. Futility-halted run keeps best.pth; full eval_test + f histogram + corr read still valid and mandatory.

## §6 — Line-by-line §11 / prior-hypothesis disambiguation

| Constraint / prior hyp | Why H0026 does not re-encode it |
|---|---|
| §11 similarity-side OPEN except gain-domain masks | `f` is a uniform positive per-image scalar — no sign test, no mask, no cell suppression; cellcal sign handling byte-identical. (If Lead rules ANY similarity amplitude = banned, card dies at review with R0–R2 already registered — never re-encoded elsewhere.) |
| §11 decoder-side live / H0025 CapFilm in-flight (N0026) | Decoder byte-identical; CapFilm owns post-GN pre-head FiLM untouched. Complementary site, not a second decoder card. |
| §11 post-decoder ADDITIVE (any) | Nothing added after `decoder(...)`; multiply is pre-`out` inside SimPrior. |
| §11 gain-domain masks (H0018 form, H0022 domain) | No mask; gain form `exp(w_c·log1p(mhat))` untouched; `f` applies after gains uniformly. |
| §11 pre-condenser exemplar gating | Never reads `e`; only `bboxes` geometry. |
| §11 transfer-by-learned-fusion external evidence | `me` = annotation box areas already in `bboxes_in`; no fusion module. |
| §11 frozen hs(2,3) + cross-attn condenser load-bearing | Both untouched; decoder `in_ch` 192 unchanged. |
| §11 DDCA / RGA / final-layer readout / unfreeze | None; no new spatial summary; no second count head. |
| §11 temp reopen / shared kernels | Temp-pin path unchanged (`use_peakcal=true`); `GemeCal` owns its Parameter, no alias to simprior/qproj/kproj. |
| Rule 11 — H0024 CountAnchor (REFUTED, response-z post-out) | Source: geometry vs response; site: pre-`ev` vs post-`out`; operator: learnable power law vs fixed `1/z` clamp; params: +1 vs 0; falsifier: corr(log me, gt) + f-load ≤0.90 vs alpha wipe-median ≥1.3×. New falsifier present as rule 11 requires. Books as **fresh H0026**, not contradicts — different mechanism claim (order-of-magnitude geometry covariate with learned sign, not unit-mass response transfer). |
| H0016/H0017 additive energy quiet-null | No additive path into cond; scales existing evidence tensor; different failure signature (f CV vs proj norm). |
| H0022/H0023 evidence-layer wipes 0/11 | Never touches evidence sign/substrate/gain form; phase/substrate questions were theirs — this is the amplitude/capacity question. |
| H0010/H0015/H0002 scalar/regime calibration | No predicted count, no regime split, no global constant; per-image `me`-driven `f` with CV ≥ 0.05 gate. |
| H0006/H0007 second grad path | `GemeCal` has no shared weights; gradients flow through `w_g` and the parent `out` path only (parent trains as before). |
| Loss/optimizer/schedule invariant | Untouched; architecture scalar, not a loss term. |
| Seed / pluggable / append-only | Seed 20260830; one switch `use_geme`; GemeCal built after PeakCal with zero RNG draws; flag-off ≡ parent. |

Runner parse line (STATE gotcha 2):

1. **H0026** — IF a zero-init geometry magnitude embedding scale (`use_geme`) multiplies the SimPrior similarity evidence tensor ev = cat[top1, cons] by a per-image factor f = clamp((me/32)^w_g, 0.25, 4) where me = mean_K(S²/(w_k·h_k)) is the exemplar-box order-of-magnitude capacity (S=input size, each annotation box structurally holds one object) and w_g is a single zero-initialized learnable scalar, applied after the unchanged cellcal and peakcal gains and before the unchanged zero-init out-projection, while the similarity substrate, the gain forms, the decoder, and every other parent line stay byte-identical, IN the live N0015 head on FSC147 v2 at seed 20260830, THEN final EMA val MAE drops at least 0.30 below 19.3431 with dense-tail gt>500 below 330.81 and sparse gt<50 at or below 5.45, BECAUSE the measured S-curve miscalibration (gt7-10 ratio 1.40 over to gt138+ ratio 0.685 under, mid50-300 at 0.86 carrying 44 percent of AE) is an order-of-magnitude capacity covariate missing from a scale-free similarity pathway (cosine top1 in [-1,1], mean-1 cellcal gain, zero-init linear out) that the leverage probe measures as net-suppressive (removing the residual raises count 29.9 percent), so exemplar-box geometry me supplies the missing count-capacity covariate while the zero-init learnable w_g lets the loss set the sign that attenuates the suppressive residual on high-capacity images and reinforces it on low-capacity images, flattening the S-curve on the similarity side left open by the reexamination, without any decoder edit and without any gain-form, mask, or substrate change. DISPROVED IF final EMA val MAE is not at least 0.30 below 19.3431, or dense-tail gt>500 is not below 330.81, or sparse gt<50 exceeds 5.45, or exemplar-box magnitude carries no count information on val (Pearson corr of log(me) with log(1+gt) below 0.20), or the scale fails its engagement read (val coefficient of variation of f below 0.05, i.e. w_g collapsed to zero or me too narrow), or the capacity loading runs the wrong way (mean f on gt at least 300 not at most 0.90 times mean f on gt below 50).

## Verification record (fill as gate proceeds)

- novelty gate: **PASS** — exit 0, top sim H0025=0.601, H0024=0.473, H0023=0.397 (all < 0.82); no structural twin. Raw result: `local/research/h0026_novelty.json`
- booking: (Lead books via `discovery hypo --new --solo` on the live parent — Idea Agent does NOT book)
- coding: (delta + CPU checks — Coding Agent; up to 3 fix retries)
- launch: `run_node N00XX_h0026 --parent N0015_h0014 --hyp … --set augment=true --set futility_bar=19.0431`
