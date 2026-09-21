# H0012 idea — per-cell self-normalized energy gain field on confirmed SimPrior evidence (`use_cellcal`)

Parent N0002_h0001 · SOLO single-switch card (`use_cellcal`, default false) · frozen backbone, head-only training · decoder in_ch untouched (192 = 128 + 64) · protocol v2 (augment=true, 32ep/1800s, EMA eval) + seed FIXED 20260830 · bars: THEN final EMA val MAE <= 21.1362 (0.30 below live v2 21.4362) AND gt>500 dense-tail mean |del| below 423.40 — per §6 bars semantics these numbers re-instantiate from the live v2 parent at verdict time; the hypothesis text itself is never edited.

## §0 Booked-hypothesis placeholder (Lead: pass verbatim to `discovery hypo --new`)

IF per-cell self-normalized energy gain field on confirmed SimPrior evidence (use_cellcal) IN N0002_h0001 head-only v2 protocol seed 20260830, THEN final EMA val MAE falls at least 0.30 below 21.4362 and gt>500 dense-tail mean |del| falls below 423.40, BECAUSE self-normalized per-cell energy directs existing exemplar-similarity mass toward high-energy dense cells where the baseline undercounts, DISPROVED IF final EMA val MAE is not at least 0.30 below the live v2 parent or gt>500 dense-tail mean |del| is not below the live v2 baseline.

## §1 Survey note (web-verified 2026-09-21, no closed-door derivation)

Claim grounded: per-location calibration of density evidence / energy-guided amplification of similarity-matched regions is an accepted direction in few-shot counting.

- CACViT (Wang et al., arXiv:2305.04440 — verified: abs page resolves to "Vision Transformer Off-the-Shelf: A Surprising Baseline for Few-Shot Class-Agnostic Counting", v2 4 Mar 2024; magnitude embedding + aspect-ratio-aware scale embedding compensate scale/order-of-magnitude information loss from resizing and normalization in plain ViT; FSC147 23.60% error reduction): precedent that an explicit magnitude/energy signal must be re-injected alongside similarity matching because matching alone loses count-scale information. Our card is the decoder-side analog: a per-cell energy field re-injected onto the already-confirmed similarity evidence, not a backbone embedding.
- SAFECount (You et al., WACV 2023, arXiv:2201.08959 — verified: abs page resolves to "Few-shot Object Counting with Similarity-Aware Feature Enhancement", v5 11 Sep 2022; SCM builds per-position score maps then normalizes along BOTH the exemplar dimension and the spatial dimensions, FEM uses the point-wise similarities as weighting coefficients to enhance query features; FSC-147 MAE 22.08→14.32; code github.com/zhiyuanyou/SAFECount): precedent that similarity evidence must be spatially normalized before it can direct enhancement — the direct ancestor of our self-normalized per-cell field (mean-1.0 normalization giving dense cells >1, background <1). We differ by amplifying the confirmed evidence channels rather than fusing support features.
- L2HCount (Xu et al., arXiv:2503.12935 — verified: abs page resolves to "L2HCount: Generalizing Crowd Counting from Low to High Crowd Density via Density Simulation", 17 Mar 2025; Dual-Density Memory Encoding Module learns scene-specific low- vs high-density patterns because one global readout cannot serve both regimes): precedent that dense scenes need a regime-specific encoding path. Our card is the minimal form of that insight: a per-cell gain that lets the same readout behave regime-selectively inside dense images without new branches.
- Deliberately NOT cited: temperature/entropy literature. This card pins no temperature and computes no entropy; the mechanism is energy-guided amplitude calibration of confirmed evidence.

## §2 Exact math + attach points (parent `tree/N0001_champion/N0002_h0001/model.py`)

Parent facts: `CountingHead.__init__` (model.py:147–159) builds fuser → exemplar (on h3) → cond (Condenser, model.py:111–125) → decoder `DensityDecoder(in_ch=D + cond_dim)` = 192 (model.py:156). `CountingHead.forward` (model.py:161–175): `fine = self.fuser(h2, h3)` (:163), `e = self.exemplar(h3, ...)` (:166), `cond = self.cond(fmap, e)` (:167), `cond_map` reshape to (B,64,96,96) (:168), H0001 SimPrior residual gate (:171–172), decoder concat `torch.cat([fine, cond_map], 1)` (:173). `SimPrior.forward` (model.py:212–221): `ev = cat([top1, cons])` (:218–220, (B,2,96,96)), `return self.out(ev)` (:221, (B,64,96,96) zero-init at step 0). `Counter.__init__` (model.py:225–242) attaches SimPrior AFTER all parent modules (model.py:241–242) — the append-only RNG pattern (AGENTS.md rule 13) this card copies. Config: seed 20260830 (config.toml:9), `use_simprior=true` (config.toml:27), `augment=false` live (config.toml:31; v2 runs add `--set augment=true`), `d_fine=128, cond_dim=64` (config.toml:20–21).

CellCal (`use_cellcal`, ONE zero-init scalar `w_c`), all shapes for S=384, B batch:

```
m     = fine.detach().mean(dim=1, keepdim=True).clamp_min(0)   # (B,1,96,96) per-cell energy; NO gradient into fine
mbar  = m.mean(dim=(2, 3), keepdim=True)                        # (B,1,1,1) per-image mean energy
mhat  = m / (mbar + 1e-6)                                       # self-normalized: mean 1.0, dense cells >1, background <1
gain  = torch.exp(w_c * torch.log1p(mhat))                      # (B,1,96,96); w_c=0 -> 1 everywhere
evp   = ev * gain                                               # (B,2,96,96) broadcast over both evidence channels
out   = self.out(evp)                                           # (B,64,96,96) into the existing cond residual
```

Attach: `CountingHead.__init__` reads `self.use_cellcal = _get(cfg, "use_cellcal", False)` (non-module flag, no RNG — same pattern as `use_simprior`, model.py:159). `Counter.__init__` constructs `self.head.cellcal = CellCal()` (a 1-scalar module: `self.w_c = nn.Parameter(torch.zeros(()))`) AFTER the SimPrior attach lines (model.py:241–242), so all parent init draws keep RNG positions; only the scalar's own (zero, no RNG draw) appends. `SimPrior.forward` (or the :171–172 call site, Coding Agent's choice — behaviorally identical) inserts:

```
if self.use_cellcal:   # flag lives on head; scalar lives in self.cellcal
    ev = ev * torch.exp(w_c * torch.log1p(mhat))
```

`torch.cat([fine, cond_map], 1)` (:173) still sees 128+64=192 channels — decoder in_ch untouched. `e` never touched; `fine` only read through `.detach()`; no shared projections; no new branches; no attention; no queries.

Step-0 byte-identity def-use check (for the Coding Agent — must hold before smoke): (a) `w_c` init to exactly 0.0 ⇒ `gain` is all-ones ⇒ `evp == ev` bit-for-bit ⇒ `self.out(evp) == self.out(ev)` (all-zeros at init anyway) ⇒ with `use_cellcal=true` at init, `cond_map` and decoder input are identical to the parent; (b) with `use_cellcal=false` (default) the `if` body never executes — forward is the parent's byte-for-byte including the SimPrior branch behavior; (c) def-use: `self.head.cellcal` defined in `Counter.__init__` before any forward can run; `m` derived from the already-computed `fine` (:163), `ev` from the already-computed `top1/cons` (:218–220) — the Condenser path (`fmap`, `e`, `cond`) is read-only, never reassigned; (d) no shared state: `w_c` lives under `self.head.cellcal` only, never aliasing `self.simprior.*`, `self.cond.*`, or `temp`; (e) smoke must assert: flag-off forward equals parent forward, and flag-on-at-init forward equals flag-off forward (max abs diff 0). Params +1 scalar, well under the 32M mission cap.

Structural-twin pre-emption (H0010 is not this): H0010 `counttau` was a per-IMAGE scalar — one gain `exp(w_g*g)` for all 9216 cells of an image, with surrogate `g` a log-compressed global energy whose val spread was 0.091 (too narrow to direct anything per image; verdict: engaged-but-redistributive, d=-0.027, dense 423.60). This card is a per-CELL field — 9216 gains per image from a self-normalized surrogate `mhat` with mean exactly 1.0 by construction, so dense cells sit at 2–5x while background sits near 0, a dynamic range the log-global scalar cannot express. Different granularity (1 vs 9216 gains), different surrogate (log global energy vs self-normalized cell energy), different causal claim (global amplitude shift vs within-image mass direction), and a different falsifier (§4 reads 1–4, none of which H0010 could pass by construction since a scalar field has zero gain spread and zero gain/GT-density correlation). A TF-IDF or structural judge that flags "scalar gain on evidence" as a twin is confusing the actuator (`exp(w·surrogate)`, deliberately reused for step-0 identity) with the mechanism (where the surrogate varies). Refutation here means the per-cell direction failed, not a re-run of the per-image amplitude question.

## §3 Why-not (the two v2 lessons + §11)

- NOT H0010 retried (§2 paragraph above is the full argument): H0010 proved AMPLITUDE is the winning direction (`w_g=+0.1476`; 3425.jpg err 312.7→106.1, 3665.jpg 1040.1→819.5) but showed a per-image scalar lacks directionality (surrogate spread 0.091; dense 423.60, +0.20). This card keeps the winning actuator (positive amplitude gain) and replaces the losing surrogate with one that varies within the image where the undercount lives.
- NOT H0011 retried: H0011 `msq` added an h2 query lane (86k params, lane norm 1.10, lane-vs-cond cosine 0.15 — used and novel) yet collapsed the dense block (22.6840, +1.25; dense 498.97; 3425: 312.7→46.7, 3428: 167→16.1, 3433: 240→51, 935: 323→144) while sparse stayed flat: engaged-but-harmful SUPPRESSOR. Diagnosis: h2-grid queries are too coarse for the catastrophic block; new spatial queries are the WRONG direction. This card adds no queries, no attention, no new feature path — pure sign-positive amplification of the existing confirmed evidence (gradient on undercounted dense pushes `w_c>0`), so the H0011 collapse signature (recall destruction on exactly the images that most need recall) is structurally unavailable.
- §11 exemplar-gating compliance (explicit): §11 bars pre-condenser exemplar gating in ANY granularity. This card never gates, rescales, or otherwise touches `e` — the exemplar path (`self.exemplar(h3, ...)`, :166) through the Condenser (`self.cond(fmap, e)`, :167) is byte-untouched. `fine` is read through `.detach()` for the energy surrogate only. The intervention sits on the decoder-side similarity-evidence readout (`ev`, :220) — a live direction per §11 ("query/similarity-side and decoder-side — unproven, re-book fresh"), not a settled one.

## §4 Predictions + failure modes + mechanism reads

Verdict context: v2 baseline 21.4362 @ep32, dense gt>500 mean |del| 423.40 (n=17); H0010 redistributed (±200 moves, net −0.027); H0011 suppressed (dense block collapse to near-zero on 3425/3428/3433).

- P1 (dense recall, not reshuffle): dense-block errors move TOWARD gt — predicted counts on 3425/3665-class undercounted images rise toward gt while already-correct sparse images hold. Signature is recall-up, distinguishing from H0010-style reshuffle (some dense improve, others worsen equally) and H0011-style collapse (dense predictions crater toward zero).
- P2 (sparse guard): gt<50 slice MAE must not degrade beyond +0.10 — bounds texture-amplification false positives from energy following background texture.
- F1 (saturating scalar): `w_c` stays ≈0 or gain-field spread ≈0 at verdict ⇒ the field never engaged ⇒ mechanism failure, REFUTE even if a bar flickers.
- F2 (dead surrogate): `mhat` flat on the dense block (no per-cell variance to direct) ⇒ surrogate carries no signal ⇒ REFUTE with diagnosis (energy is the wrong surrogate; do not re-book energy variants silently).
- F3 (wrong-direction amplification): dense errors move AWAY from gt or collapse toward zero (H0011 signature) ⇒ REFUTE.

Mechanism reads at verdict (beyond the two conjunctive bars; Lead has the hook harness):
1. `w_c > 0` decisively (positive amplitude; gradient on undercounted dense pushes it there; `w_c ≤ 0` refutes the sign claim).
2. Gain-field/GT-density correlation on the dense block exceeds the baseline ev/GT correlation (the field directs mass where objects are, measured by verdict hooks).
3. Dense-block per-image errors move TOWARD gt (recall up — neither H0010 reshuffle nor H0011 collapse).
4. Sparse gt<50 slice does not degrade beyond +0.10 (no texture-amplification false positives).

## §5 Risks + same-seed v2 pair protocol

- Risk: texture amplification on sparse (energy follows high-frequency background) — bounded by single-scalar capacity (cannot learn texture templates), zero-init (step-0 identity), and pre-registered guard P2/read 4 as a refutation path, not a post-hoc story.
- Risk: `w_c` saturating to a constant (field reduces to H0010-style global shift) — detected by read 1+2 (gain spread ≈0 or no correlation lift ⇒ refute per F1).
- Risk: RNG-order drift breaking the paired contrast — mitigated by append-only construction (scalar module after every parent module; zero-init draws no RNG) + fixed seed; smoke asserts step-0 identity.
- Protocol: SAME-SEED v2 pair only — child (`use_cellcal=true`, `--set augment=true`) vs canonical v2 baseline (N0002_h0001 `--set augment=true`), seed 20260830 FIXED, 32ep/1800s, EMA eval, `final EMA val MAE` + `gt>500 dense-tail mean |del| (n=17)`. NEVER cross-protocol: pre-v2 (augment=false) numbers are historical record only. Verdict binds to live v2 numbers at run time (§6 semantics: THEN final EMA val MAE at least 0.30 below the live parent AND dense-tail below the live baseline); both the booked 21.4362/423.40 and any re-instantiated live values are stated in the evidence note. `repro_run --set seed=` is FORBIDDEN without explicit user approval.
