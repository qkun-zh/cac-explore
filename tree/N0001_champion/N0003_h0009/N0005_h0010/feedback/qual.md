# QUAL — N0005_h0010 (H0010 per-exemplar gate + H0011 frozen-random gate control)

Result: 22.810 @ ep32 (32/32, 1705.8s). Parent N0003 = 22.608. Champion = 21.459.

## 0. What actually changed vs parent (code verification)

Parent N0003 feeds the 33k-param channel gate the pooled fine map: `gate = sigmoid(channel_gate(fine.mean(dim=(2,3))))` (N0003 model.py:169). Child adds two switches, both confirmed literal:

1. **Const gate input is truly frozen/random.** `frozen_gate_input = torch.randn(D, generator=manual_seed(cfg.seed))`, `requires_grad_(False)`, `register_buffer(..., persistent=False)` (model.py:162-166). No gradient path, no data dependence, sampled once at build, reproducible for the same seed, and omitted from the checkpoint. In forward it is a constant: `gate_in = self.frozen_gate_input.expand(B, -1)` (model.py:181). Param count stands: `Linear(128,256)=33,024`, matching the H0011 text.
2. **Exemplar gate is genuinely per-exemplar.** `exemplar_gate = nn.Linear(256, 1)`; weight zero-inited, bias +3 (model.py:169-171); applied `e = e * sigmoid(exemplar_gate(e))` (model.py:187). This yields B×K distinct scalars broadcast over channels — unlike the channel gate's `gate.unsqueeze(1)` (model.py:185), which is one B×256 scale shared by all K exemplars. Composition order: `e ← ExemplarEncoder(e) → ×channel_gate (const) → ×exemplar_gate` (model.py:185 then :187); the exemplar gate's input e is thus *downstream* of the channel gate.

## 1. Does the +0.202 delta make mechanistic sense?

Yes — and it lands squarely inside the inert-capacity bar. With a constant input, `sigmoid(W·x0+b)` collapses to a *fixed* 256-dim multiplier in (0,1)^256 applied identically to every exemplar of every image: a dataset-constant per-channel rescale of the exemplar embedding.

What that rescale can do: e doubles as key **and** value in the condenser cross-attention (`attn(norm1(tok), e, e)`, model.py:123), so a per-channel scale re-weights the q·k similarity geometry *inside* the softmax and also scales the V output; any constant offset is re-absorbable by `proj_in`/`out`/decoder. That subspace — "generic exemplar channel re-weighting" — is exactly the subspace a content-conditioned gate must *move within* to earn its keep. If per-image content-conditioning were load-bearing, the gate would have to sit far from its dataset-average operating point on a material share of samples; instead the constant version holds within +0.202, i.e. the residual input-conditional motion was worth < ~0.3 MAE across the whole dataset.

Two sub-readings, both consistent:
- (a) the gate learned ≈ a near-constant output anyway (readout `W` collapsing toward the dataset-average direction, or `W→~0`) → feature-conditioning was never exercised;
- (b) the gate output did move per image, but the condenser/decoder absorbed the variation → the motion had no net value.

Distinguishing them needs the learned weights (not available from result.json), but it does not matter for verdict: under either reading the gate's benefit is dominated by its constant/global component. H0011's falsifier (within 0.30 of 22.608) is satisfied; ledger already holds `contradicts` → conf 0.400. Context caveat: this control shows the *conditioning component* was not worth >0.30 — it does **not** rehabilitate the gate itself. Parent N0003 (22.608) is already ~1.15 *worse* than champion (21.459): the gate is net-negative on both counts (H0009) and its conditioning is inert (H0011).

Scale sanity: +0.202 is small against the ±0.01–0.04/epoch tail wobble documented around the 22.6 basin (N0003 qual §3) and far below the 1.1–1.9 deltas that mark real mechanism differences in this lineage — consistent with "no meaningful signal, control reproduces the conditioned run."

## 2. H0010's per-exemplar construction — was N0005 an adequate test?

**Expressibility: yes.** A per-exemplar scalar multiplier computed from each exemplar token's own vector, applied along the exemplar (K) axis, not the image-global channel axis — the thing the channel gate could *not* do is expressible here (model.py:187 vs :185). And its init is good hygiene: `sigmoid(3) ≈ 0.953`, i.e. ~5% attenuation at step 0, a gentle ablatable default — the stark opposite of the channel gate's halving init (sigmoid≈0.5) that N0003 flagged as a paid early cost.

**Adequacy: no.** Two confounds, one load-bearing:
- **Set-level (load-bearing):** N0005 toggles **two** switches against its parent: adds `use_exemplar_gate` (H0010) **and** flips `use_const_gate_input` on (H0011). The 22.810 is a joint delta; the required comparator for H0010 ("champion without any exemplar gating" = 21.459) flips the sign depending on which baseline you pick (−1.351 vs champion, +0.202 vs own parent). Un-attributable. The ledger's `neutral` booking is correct; H0010 after two runs (N0004 joint-with-H0009, N0005 joint-with-H0011) has still *never* been run univariately and stays unconcluded.
- **Context-level (softer):** the exemplar gate reads `e`, which sits downstream of the severed channel gate. Its per-exemplar *content* still varies (that variation comes from the ExemplarEncoder, model.py:178), so this is **not** an information-free input — but the `e` it sees is rescaled by a fixed constant channel scale rather than the content-conditioned one the H0010 hypothesis assumes, so the exact operating regime differs from any clean H0010 test. I flag this precisely and softly: the set-level confound does the real work; "broken context" is a secondary regime shift, not a second severed signal.

## 3. Pluggability / ablation hygiene

- `use_exemplar_gate` — toggles only the module (model.py:167-171) and its single application point (model.py:187). Removing it returns to parent exactly. ✓
- `use_const_gate_input` — toggles only the module (model.py:161-166) and the input branch (model.py:180-183). Removing it **does restore content-conditioning** (`gate_in = fine.mean(dim=(2,3))`, model.py:183) with the exemplar gate still active — this is precisely the next-net experiment that isolates H0010.
- Hidden couplings (none are gate violations, but should be known):
  1. `use_const_gate_input` is silent unless `use_channel_gate` is on — the module is built under `if self.use_channel_gate and use_const_gate_input` (model.py:162). Setting it alone does nothing. Undocumented dependency, benign, worth a config comment.
  2. `persistent=False` (model.py:166): the frozen vector is not in the checkpoint and is regenerated from `cfg.seed`. Consistent across identical seed; a changed seed would silently swap the constant. Fine for this control (any constant behaves the same) but a foot-gun for any "checkpoint reuse" scenario.
  3. Gradient path: both gates multiply the same tensor in series (model.py:185→187); their gradients interleave through the common `e`. Modular yes, attribution-independent no — single-switch toggling remains the right method; expect interaction effects by construction.

## 4. Direction for a next qualitative net test (no fabrication)

Cheapest confound-resolution first, each one switch at a time:
1. **Univariate H0010** — child of N0003 with only `use_exemplar_gate=true` (+`use_const_gate_input=false`, channel gate intact). Comparator = champion 21.459, 0.30 bar. Third attempt, first clean one.
2. **Univariate H0011** — the inverse of this node's other half: same parent, toggle only `use_const_gate_input` (no exemplar gate). If ≈22.8 → this node's +0.202 is fully explained by severed conditioning and H0010 is still untested (contradicts nothing); if ≈22.6 → +0.202 is the exemplar gate's own doing, which for a near-identity-inited 1-weight-per-exemplar readout would be a strong signal.
3. **Inert-capacity check one level down** — replicate this node's trick on the exemplar gate itself: equal-capacity per-exemplar gate vs per-exemplar *shared constant* scalar, same near-identity init, toggle only the per-exemplar input. Does per-K variation pay for anything, or is it inert again?
4. Only if 1–3 lean positive: inspect gate activations on the worst-decile (per-image MAE) val scenes with heterogeneous exemplars — is the per-exemplar multiplier actually spread across K on the samples the BECAUSE clause cares about? If spread ≈ 0 everywhere, the per-K expressibility is unexercised; kill it.

## Budget stance

Single event, no re-run recommendation. H0011 is settled (falsifier met); H0010 remains open and must not be treated as either confirmed or refuted from this confounded node.