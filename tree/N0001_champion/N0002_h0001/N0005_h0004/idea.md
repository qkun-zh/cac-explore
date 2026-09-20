# N0005_h0004 — idea (booked H0004, `use_simbank`)

> **Booking header — mechanical record only, not a content change.**
> Node: **N0005_h0004** · Parent: **N0002_h0001** (live parent, best val MAE **22.5641 @ep30**, `use_simprior=true`) · Booked hypothesis: **H0004** · Switch: `use_simbank` (config default `false`; child sets `true`) · Falsifier bar: final EMA val MAE **≤ 22.26** at seed 20260830 under the canonical 32ep/1800s protocol (≥0.30 below the live parent 22.5641).
> The §5 hypothesis line is byte-identical to the booked text (`/tmp/book_simbank.txt`, 642 chars, no embedded H/N ids beyond the booking header).

---

# H0004 — Finer 1/8 exemplar-token similarity bank (`use_simbank`)

- **Hypothesis id:** H0004 · **Child node:** N0005_h0004 (child of N0002_h0001) · **Switch:** `use_simbank` (config default `false`) · **Composition:** single-switch delta on the simprior-bearing live parent (`use_simprior=true` inherited, `use_simbank=true` added).
- **One-line intent:** Augment the confirmed N0002 simprior with a second, finer similarity readout whose exemplar bank is built from **per-part ROI tokens of h2@1/8** (192ch, 48×48) rather than the pooled 1/16 `e`, recovering small-object match evidence the pooled prior averages away — startable as the exact parent function via a zero-init projection.

## 2. Multi-angle reasoning

### (a) Pure-mathematics lens — the pooled bank is a degenerate (rank-K, K=3) special case

The confirmed simprior (`tree/N0001_champion/N0002_h0001/model.py:212-221`) is a similarity readout over the K=3 pooled exemplar vectors:

```
e    = ExemplarEncoder(h3@1/16, boxes)      # (B, K=3, 256) — one vector per exemplar
p    = normalize(kproj(e))                   # (B, 3, 64)
q    = normalize(qproj(fine))                # (B, 64, 96, 96)
S    = <q, p>                                # (B, 3, 96, 96) cosine
ev   = [ max_k S ,  softmax_k(S/τ)·S ]       # (B, 2, 96, 96)
cond += out(ev)                              # out: 2→64, zero-init
```

For every cell the evidence is a **max over exactly 3 hypotheses** (the three pooled exemplar vectors). The part-level geometry *inside* each exemplar is destroyed before the readout: `ExemplarEncoder` ROI-aligns h3@1/16 with a 7×7 grid (`model.py:97`) and then softmax-pools all 49 tokens to a single vector per exemplar (`model.py:102-103`). At 1/16 a small FSC147 object occupies ≈1–2 map cells, so most of the 49 ROI samples are padding/background and the pooled vector is close to a background average — its cosine field is nearly flat and carries little object evidence. The evidence functional is therefore `max` over K pooled keys, and for small objects those K keys are near-collinear.

H0004 keeps a **bank of keys per exemplar**: ROI-align h2@1/8 at the same boxes (r×r tokens), project each token to a d_proj space, and compute cosine against every key:

```
p_bank = normalize(bank_proj(roi_align(h2@1/8, boxes, r×r)))   # (B, K·J, 64), J=r²
S_bank = <q, p_bank>                                            # (B, K·J, 96, 96)
top1   = max_{k,j} S_bank                                       # (B, 1, 96, 96)
per_k  = max_j S_bank                                           # (B, K, 96, 96)
cons   = softmax_k(per_k/τ) · per_k                             # (B, 1, 96, 96)
cond  += out([top1, cons])                                      # out: 2→64, zero-init
```

Mathematically the new evidence field is a strict generalisation of the old one: with J=1 and a mean-pool instead of max, `max_{k,j}` collapses to `max_k` over pooled keys. Because `max` over a larger hypothesis set can only be ≥ the max over a subset, the bank bound is tighter around true part matches and the exemplar-consensus `cons` is now built from genuinely independent part detectors (the softmax-over-K of per-exemplar maxima is non-degenerate when different exemplars win on different parts, whereas pooled vectors often agree). The readout stays a *readout*: `q` is unchanged (`fine`), no representation is learned inside the frozen backbone, and the zero-init `out` lets the optimizer choose the trust of the extra evidence — at init the child forward is exactly the parent's.

### (b) Champion-lineage lens — file:line facts against `tree/N0001_champion/N0002_h0001/model.py`

- **The pooled bank that H0004 refines.** `ExemplarEncoder.forward` (`model.py:91-108`): `rois` are built at the h3 feature scale `s = W/img_size` (`model.py:94`, W=24 at 1/16), `roi = roi_align(feat, rois, output_size=(7,7))` (`model.py:97`), `tok = proj(roi)` 384→256 (`model.py:98`), transformer (`model.py:101`), **softmax pool to one vector**: `a = attn(tok).softmax(1); out = (tok*a).sum(1)` (`model.py:102-103`), plus XScale coarse average (`model.py:104-107`). Output `e` is (B,K,256) — all intra-exemplar geometry is gone here.
- **The confirmed readout.** `SimPrior` (`model.py:193-221`): `qproj` 128→64 (`:205`), `kproj` 256→64 (`:206`), `temp` (`:207`), zero-init `out` 2→64 (`:208-210`), forward `S = einsum('bchw,bkc->bkhw', q, p)` (`:216`), `top1`/`cons` (`:218-219`), add into `cond_map` at `CountingHead.forward:171-172`.
- **The open cell — 1/8 is already on the interface.** `Backbone.hs_map = (2,3)` (`model.py:38`), so `h2` is hidden_states[2] at 1/8 (B,192,48,48) and is passed into `CountingHead.forward(h2, h3, bboxes_in)` (`model.py:161`, wired from `Counter.forward:253-254`). `FineFuser` already converts h2/h3 to `fine` (`model.py:67-74,163`). Nothing in the lineage currently reads exemplar evidence from h2; the pooled prior and the Condenser both consume the same 3 pooled h3 vectors, so they share the small-exemplar blind spot.
- **Live parent and bar.** Parent N0002_h0001 best EMA val MAE **22.5641 @ep30** (its `result.json`), 32/32ep, `budget_hit false`, 1755.0 s wall. H0001's mechanism was confirmed and its trained readout is live (`head.simprior.out` norm 0.5874 off zero-init). H0004 asks for a further **≥0.30** gain: bar **≤22.26**.
- **Untouched load-bearing interfaces.** hs(2,3) readout, `Condenser`, `DensityDecoder` (in_ch stays 192), `GCA` (`model.py:186-190`), `SimPrior` (read-only neighbour), XScale all stay intact; contract `build_model(cfg) → forward → {"density", ...}` unchanged, only `out["density"]` feeds the loss.
- **Register compliance (AGENTS §11).** No DDCA/dilated branch, no extra spatial summaries/RGA, no readout/backbone unfreeze, no final-layer readout, and **no pre-Condenser exemplar gating of any granularity**: SimBank never gates, scales, or reshapes `e` or `fine`; it reads them as-is and adds a residual into `cond` *after* the Condenser. This is explicitly a query/similarity-side direction the register marks live (re-book fresh if wanted).

### (c) Counter-intuitive / low-cost lens

The head already contains **two** matchers (Condenser cross-attention + SimPrior), so a third looks redundant. It is not, because the bottleneck is not the amount of matching but the **resolution of the exemplar bank**: all existing matchers consume the same three pooled h3 vectors, so they share one degenerate representation for small objects. H0004 adds no matching *capacity* around `e` (which would be pre-Condenser gating, banned) — it adds a parallel, finer *read-out key set* and lets one zero-init projection decide whether to trust it. Cost is ≈20.7k params (0.067% of the 32M cap, ≈3.3% of the remaining headroom) and ≈0.12 GFLOPs/forward (dominated by the 1×1 query projection the parent already pays); flag-off reproduces the parent forward byte-for-byte, and init-forward is identical because `out` is zeroed. A negative is cleanly informative: if the bank projection stays ≈0 or MAE misses the bar, then the pooled prior was not the small-object bottleneck and the family should be redirected (e.g. background/negative prototypes), not retried.

## 3. Citations (RULE 15 — satisfied by the two completed research reports)

- **CountingDINO** — ROI-Align prototypes used as convolution kernels → normalized similarity maps on **frozen** DINO features, training-free; val 25.48 / test 20.93. <https://arxiv.org/abs/2504.16570> · confidence **H** for "normalized similarity readout over ROI prototypes carries counting signal on frozen features". The survey's own risk note on this row states the exact failure mode H0004 targets: *"FSC147 objects small → 1/16 ROI prototype noisy"* (`local/research/survey_mechanisms.md:26`).
- **Survey F1 failure mode (a)** — *"1/16 ROI for small objects gives mostly padding → prototype = background average; mitigate by ROI-aligning at 1/8 (h2) or max-pooling"* (`local/research/survey_mechanisms.md:61-62`, family F1 notes `:50-64`). H0004 is the h2-ROI + max-over-token mitigation applied to the already-confirmed similarity readout; confidence **H** as interface reasoning on our exact hs(2,3)+e shapes (not a measured gain).
- **SAFECount** — score-map normalization across exemplar AND spatial dims; val 15.28 / test 14.32, ResNet-18. <https://arxiv.org/abs/2201.08959> · confidence **H** for the softmax-over-K consensus and the need to calibrate raw score maps. Its similarity-weighted query-feature enhancement (FE) is the banned pre-Condenser gating family and is explicitly **not** reused here.
- **CACViT** — magnitude/scale calibration + similarity-based background suppression; 3-shot val 10.63 / test 9.13, ViT-B. <https://arxiv.org/abs/2305.04440> · confidence **H** that raw similarity needs calibration (temperature) and that positive-only matching leaves false positives — bounds H0004 to matching evidence only; suppression is a separate future switch.
- **Contrast (what H0004 is not).** LOCA OPE (<https://arxiv.org/abs/2211.08217>) modifies `e` pre-Condenser (survey F3, register edge); the sibling survey family F2 (`use_msq`) cross-attends `e` into per-level maps; family F6 (`use_negproto`) builds background prototypes. H0004 modifies no `e`, adds no attention, and uses positive exemplar tokens only — it refines *where/how finely the comparison is read*, the survey's ranked essence-coverage direction (survey §5: frozen-backbone mechanisms that change where comparison happens should dominate post-processing).

## 4. Exact implementation spec for the Coding Agent (child node N0005_h0004 only)

**Parent reference.** Copy `tree/N0001_champion/N0002_h0001/model.py` verbatim, then apply exactly the delta below. Parent already contains `SimPrior` (`model.py:193-221`) and constructs it in `Counter.__init__` after every parent module (`model.py:236-242`). Child config = parent `config.toml` + one key; parent has `use_simprior = true`.

**Config delta (child `config.toml` only):**
```toml
use_simprior = true    # unchanged from parent (confirmed H0001)
use_simbank = true     # NEW key; default false (see below) — child run sets true
```
Default-handling rule: every reader uses `_get(cfg, "use_simbank", False)`, so a config without the key takes the parent path bit-identically.

**New module class** (name fixed): `SimBank(nn.Module)`.

```python
class SimBank(nn.Module):
    """H0004 finer 1/8 exemplar-token similarity bank (`use_simbank`).

    Builds per-part exemplar keys by ROI-aligning h2@1/8 (B,192,48,48) at the
    exemplar boxes (r x r tokens per exemplar, NO pooling), projects them to a
    d_proj space, and computes per-cell cosine match evidence of `fine` against
    the full K*r*r key bank. max-over-token + softmax-over-K consensus are
    compressed by a ZERO-INIT 1x1 projection added residually into `cond`.
    Reads h2/fine read-only; never modifies `e`; decoder in_ch stays 192.
    """
    def __init__(self, d_fine=128, d_bank=192, d_proj=64, cond_dim=64, roi_size=5, n_ev=2):
        super().__init__()
        self.r = roi_size
        self.qproj = nn.Conv2d(d_fine, d_proj, 1, bias=False)      # 128 -> 64
        self.bank_proj = nn.Conv2d(d_bank, d_proj, 1, bias=False)  # 192 -> 64
        self.temp = nn.Parameter(torch.tensor(0.07))               # learnable temperature
        self.out = nn.Conv2d(n_ev, cond_dim, 1)                    # 2 -> 64, ZERO-INIT
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h2, bboxes, fine, img_size):
        # h2: (B,192,48,48) 1/8; bboxes: (B,K,4) input-scale; fine: (B,128,96,96)
        B, C, H, W = h2.shape
        K = bboxes.shape[1]
        s = W / float(img_size)                                    # 48/384 = 1/8
        idx = torch.arange(B, device=bboxes.device, dtype=bboxes.dtype).view(B, 1, 1).expand(B, K, 1)
        rois = torch.cat([idx, bboxes * s], -1).reshape(B * K, 5)
        roi = roi_align(h2, rois, output_size=(self.r, self.r))    # (B*K,192,r,r)
        p = self.bank_proj(roi).flatten(2).transpose(1, 2)         # (B*K, J, 64)
        p = F.normalize(p, dim=-1).view(B, K * self.r * self.r, -1)  # (B, K*J, 64)
        q = F.normalize(self.qproj(fine), dim=1)                   # (B, 64, 96, 96)
        S = torch.einsum('bchw,bmc->bmhw', q, p)                   # (B, K*J, 96, 96)
        top1 = S.amax(dim=1, keepdim=True)                         # (B, 1, 96, 96)
        per_k = S.view(B, K, self.r * self.r, *S.shape[-2:]).amax(dim=2)  # (B, K, 96, 96)
        Wk = (per_k / self.temp.clamp_min(1e-3)).softmax(dim=1)    # over K
        cons = (Wk * per_k).sum(dim=1, keepdim=True)               # (B, 1, 96, 96)
        ev = torch.cat([top1, cons], dim=1)                        # (B, 2, 96, 96)
        return self.out(ev)                                        # (B, 64, 96, 96), zero at init
```

**Constructor order (append-only RNG, AGENTS rule 13).** In `Counter.__init__`, after the existing `self.head.simprior = SimPrior(...)` block (`model.py:236-242`), append:

```python
self.head.simbank = SimBank(d_fine=D, d_bank=dims[0],
                            cond_dim=_get(cfg, "cond_dim", 64),
                            roi_size=5)
```

`SimBank` is the **last** `nn.Module` constructed, so every parent-module and `SimPrior` init draw keeps its exact RNG position; only `SimBank`'s own `qproj`/`bank_proj` default-init draws are appended. It is constructed unconditionally (like `SimPrior`), with forward gating by the flag; with `use_simbank=false` its params receive no gradient and are never used in the forward, so they stay at init and the parent trajectory is reproduced (AdamW skips params with `grad is None`; EMA shadows them inertly).

**Attachment point and shapes** (in `CountingHead.forward`, `model.py:161-175`):
- After the existing simprior residual (`model.py:171-172`), before the decoder call (`model.py:173`):
  ```python
  if self.use_simbank:
      cond_map = cond_map + self.simbank(h2, bboxes_in, fine, self.S)
  ```
- Inputs: `h2` (B,192,48,48) read-only, `bboxes_in` (B,K,4), `fine` (B,128,96,96) read-only, `self.S` = `input_size` (384). `cond_map` stays (B,64,96,96); decoder `in_ch` stays 192. `self.use_simbank` is set in `CountingHead.__init__` next to `self.use_simprior` (`model.py:159`): `self.use_simbank = _get(cfg, "use_simbank", False)`.
- With the flag off the line is skipped and the forward is the parent's byte-for-byte (simprior residual + decoder unchanged).
- **Explicit non-violation statement:** this does NOT alter the shape, init, or RNG position of any parent module or `SimPrior`; does NOT modify `e` or `fine` (no gate, no rescale — pure additive residual into `cond_map` post-Condenser); does NOT change decoder `in_ch`; and reuses the frozen `hs(2,3)` interface only. It is therefore not pre-Condenser exemplar gating in any granularity.

**Zero-init plan.** `SimBank.out` (final 1×1 conv 2→64) is weight/bias zeroed, so at step 0 `delta ≡ 0` regardless of `qproj`/`bank_proj`/`temp`; the child's (`density`, `n_aux`) outputs are numerically identical to the parent's at init (lineage precedent: GCA `model.py:183-184`, SimPrior `model.py:209-210`). The `roi_size=5` default is a deliberate budget choice (see below); it is a fixed hyperparameter of this switch, not a co-composed hypothesis.

**Param delta arithmetic** (all shapes at the parent config: `d_fine=128`, `bank=192`, `d_proj=64`, `cond_dim=64`):
- `bank_proj` Conv2d(192→64,1, no bias): 192·64 = 12,288
- `qproj` Conv2d(128→64,1, no bias): 128·64 = 8,192
- `temp`: 1
- `out` Conv2d(2→64,1, with bias): 2·64 + 64 = 192
- **Total Δ = 20,673 (≈20.7k).** Parent total ≈31.345M+0.000? — concretely parent head ≈3,529,412 (3,504,643 base + 24,769 SimPrior) → child head ≈3,550,085; node total ≈27,820,128 + 3,550,085 ≈ **31,370,213 (≈31.37M) ≤ 32M** (≈0.63M headroom).

**Expected FLOPs change** (per 384px image forward; J=25, K=3 → K·J=75 bank keys, 96×96 query grid): `bank_proj` 192·64·75 ≈ 0.92M MACs; `qproj` 128·64·96·96 ≈ 75.5M; `einsum` 64·75·96·96 ≈ 44.2M; `out` 2·64·96·96 ≈ 1.2M; **total ≈ 122M MACs (≈0.12 GFLOPs)**. Backward is proportional; the extra activation is (B,75,96,96) ≈ 11M elements at batch 16 (≈44 MB fp32), negligible on 12 GB.

**Budget note.** τ_max = 1800 s; the parent N0002_h0001 ran 1755.0 s, i.e. ≈45 s headroom. The bank adds ≈0.12 GFLOPs to a forward dominated by the frozen ConvNeXt-Tiny (~10 GFLOPs at 384px), i.e. <1% → an estimated ≈0.5 s/epoch, ≈16 s over 32 epochs, expected to stay under τ_max. If `_BudgetStop` nevertheless fires, the node is marked `timeout` (not `done`) per AGENTS §4 step 6 and reported honestly; the backup lever is `SimBank(roi_size=3)` (9 tokens/exemplar, einsum ≈15.9M MACs) — a coded constant, not a new hypothesis, and to be used only to stay inside τ_max, never to chase the MAE bar.

**RNG-order statement.** The only new RNG consumption is `SimBank` parameter init, appended strictly after all parent modules and after `SimPrior`; forward contains no stochastic ops (roi_align/normalize/einsum/amax/softmax deterministic); dropout is zero everywhere; data-loader seeding (`seed=20260830`, `augment=false`) is unchanged. Parent-vs-child divergence under the canonical harness therefore comes only from the learned bank residual, satisfying AGENTS §5 rule 13 paired-contrast semantics (precision ~0.02).

## 5. Exact hypothesis text (booked verbatim — single line, byte-identical to `/tmp/book_simbank.txt`)

```
IF augmenting the live simprior readout with a finer 1/8 exemplar-token similarity bank via use_simbank IN a single-switch extension of the simprior-bearing parent node trained under the canonical protocol at seed 20260830 for 32ep/1800s, THEN final EMA val MAE falls at least 0.30 below the live parent 22.5641 to 22.26 or lower, BECAUSE per-token 1/8 ROI keys keep part-level exemplar geometry that 1/16 softmax-pooling averages into a background-dominated vector for small objects, so a max-over-token match recovers small-instance evidence the pooled prior drops, DISPROVED IF final EMA val MAE exceeds 22.26 under the canonical protocol.
```

## 6. Falsification reading

- **Confirm/refute rule (canonical protocol, seed 20260830, 32ep/1800s).** Live parent: **22.5641 @ep30**. Bar: child final EMA val MAE **≤22.26** (≥0.30 lower). **Confirm** iff ≤22.26; otherwise **REFUTE** — including match/regress and any outcome where `out` norm stays ≈0 (bank ignored). Same-seed paired contrast per AGENTS §6; no cross-seed comparison, no re-run shopping (rules 1/5/6).
- **What to inspect in the curve.** (i) best-epoch EMA val MAE and epoch; (ii) val RMSE alongside MAE (small-object recovery should move RMSE too, not just MAE); (iii) train-loss vs val divergence (the bank must move val, not deepen the train fall); (iv) learned `simbank.out` weight norm — nonzero growth with a val win means the finer bank carried real evidence; ≈0 means the pooled prior was not the bottleneck.
- **What a negative means.** A clean null (≤22.26 missed, or `out` norm ≈0) falsifies **only** this variant — "per-part 1/8 ROI keys add small-object evidence beyond the pooled 1/16 prior on the frozen backbone". It does **not** falsify the similarity-readout family; still live after a null: score distribution/statistics calibration (survey F4), negative/background prototype suppression (F6), and exemplar-side adaptation (F3, register edge). Per AGENTS §11, a refuted variant is never silently retried — any revisit books as `contradicts`-evidence on H0004 with a new falsifier.

## Booked hypothesis set (machine-readable)

1. **H0004** — IF augmenting the live simprior readout with a finer 1/8 exemplar-token similarity bank via use_simbank IN a single-switch extension of the simprior-bearing parent node trained under the canonical protocol at seed 20260830 for 32ep/1800s, THEN final EMA val MAE falls at least 0.30 below the live parent 22.5641 to 22.26 or lower, BECAUSE per-token 1/8 ROI keys keep part-level exemplar geometry that 1/16 softmax-pooling averages into a background-dominated vector for small objects, so a max-over-token match recovers small-instance evidence the pooled prior drops, DISPROVED IF final EMA val MAE exceeds 22.26 under the canonical protocol.
