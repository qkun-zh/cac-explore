# Causal feedback — N0005_h0010 (JOINT: H0010 per-exemplar gate added + H0011 channel-gate conditioning severed)

Observed: best val MAE **22.8102 @ ep32** (final epoch), 32 real-data epochs, 1705.8s/1800s, `budget_hit=false`. Verified from `result.json` (config_sha256 `f922e3663ad3efb5`, model_sha256 `1c80d0133f69ea63`). Config delta from parent N0003: `use_exemplar_gate=true` (H0010) **and** `use_const_gate_input=true` (H0011). Both switches flipped at once.

| node | config | best val MAE | Δ | 
|---|---|---|---|
| N0001_champion | no gates | 21.4589 | — |
| N0003_h0009 | channel gate, content-conditioned | 22.6076 | +1.149 vs champion (H0009 refuted) |
| N0005_h0010 | exemplar gate + channel gate, random-input | 22.8102 | +0.2022 vs N0003 |

## 1. What causal quantity this node actually estimates

Chain: champion (21.459) → +channel gate, content-conditioned (22.608, **+1.15 worse → H0009 refuted**) → +exemplar gate AND severed channel-gate input (22.810, **+0.202 vs parent**).

The estimand of this node is **NOT the marginal effect of H0010**. It is the joint contrast

    θ(N0005 config) − θ(N0003 config) = (effect of adding H0010) + (effect of H0011 severing the channel-gate input),

measured once, on one seeded lineage, at best-at-last-epoch. The two interventions act on DIFFERENT coordinates of the same forward path: H0011 rewrites the gate's input (pooled fine features → frozen random vector, `model.py:179-185`) while H0010 inserts a new multiplicative factor downstream (`e = e * sigmoid(linear(e))`, `model.py:186-187`) inside the same K/V path the gate already rescales. They are nested in the softmax, not separable in the loss, and they train jointly under a shared optimizer. **One point in two-parameter joint space cannot identify either marginal.** The +0.202 is the confounded sum; attribution of any component is unidentified from this number alone.

## 2. Sign/mechanism argument for the two parts

Even though the point estimate is confounded, mechanism constrains the sign composition:

- **The severing alone should cost ≥ 0.** The channel gate with its input severed can still absorb as a constant channel-wise rescale (`gate_in` is fixed across the run); it loses only input-contingent in-sample rescaling, i.e. it can no longer help, and a harmful/neutral constant rescale is always reachable. So the H0011 component of the sum contributes +≥0 (worse-or-equal). H0011's own falsifier bar says an amputated gate must produce ≥0.30 MAE degradation to matter — and the **combined** (confounded) sum was +0.202, i.e. **below the bar even before any countervailing exemplar-gate help**.

- **Therefore the H0010 component is at most small-negative or ~0.** If the exemplar gate were net-beneficial on this lineage, the severing's ≥0 cost would push the observed sum above the pure severing effect — but it could disguise a benefit only up to 0.202 minus the (unknown, ≥0) severing cost. Since a beneficial H0010 would need ≥0.30 to clear its own champion-bar and the sum sits at +0.202, the exemplar gate cannot have delivered anything near its pre-registered effect here. Its marginal on THIS (already-gated, refuted) lineage is bounded: ≤ small and negative, plausibly ~0.

- **Lineage context check against the champion.** Every addition since the seed has hurt or been neutral: 21.459 → (H0009) 22.608 → (H0010}+{H0011) 22.810. The node under test sits on the worst parent; whatever the exemplar gate did, it did not lift even the refuted parent back toward or past the champion. On this trajectory the gate is ~attribute-nothing in both directions.

## 3. Is the H0011 "inert capacity" falsifier causally sound?

H0011's logic: an equal-capacity gate fed frozen-random noise can only behave as a constant rescale; if it reproduces the conditioned gate's score within 0.30, the gate's benefit is **inert capacity, not feature conditioning**.

Mechanically the falsifier is well-formed as a **sever** contrast: same parameter count, same read-out, input only. The result (22.810 vs 22.608 = +0.202, within the 0.30 bar) is consistent with H0011's claim of **refutation**: the gate does not need its conditioning to deliver this score, on this lineage. The H0011 severing component of the confounded +0.202 did NOT exceed its own 0.30 falsifier bar.

But the caveat is real and must stay on the record: H0011's bar ("within 0.30 of 22.608") was itself evaluated through the joint with H0010. The severs ran in a model that ALSO gained the exemplar gate's changed exemplar context (`e` rescaled before the condenser). A gate on garbage input that happens to land within 0.30 of the conditioned score **after the downstream context moved** is weaker evidence than the same gap measured with nothing else changed. Put differently: "within 0.30" here is a confounded bound — the true sever-only gap could have been larger or smaller, and the 0.30 margin was consumed by a sum that included a second intervention.

**Honest confidence: LOW-TO-MEDIUM** on attributing "inert capacity" (rather than just "severance cost ≤0.202 on a joint run") to gate no. The refutation of the *specific* claim "a gate on this lineage needs its input to matter, and severance must cost >0.30" is supported (0.202 < 0.30, joint-run caveat cuts both ways but pushes toward a floor, not a ceiling that would rescue H0011). Attributing *why* (inert capacity) is under identified. Do not book H0011 as anything stronger than "not disproven-bar-critical; consistent with inert-capacity."

## 4. Next causal design that actually separates H0010

Still not run (flagged identically in N0004_h0010/feedback/causal.md §3, §4):

- **Parent: N0001_champion** (NOT any gated child — the only lineage where a gate can be tested against the fresh champion).
- **Single switch**: `use_exemplar_gate = true`, everything else byte-identical to champion config (`use_channel_gate = false`, seed 20260830, epochs=32, wd=0.08, lr=1e-3, GCA/XScale on).
- **Pre-registered bar, as booked**: DISPROVED IF best val MAE is not **≤ 21.1589** (≥0.30 lower than 21.4589) at best epoch, EMA-inference, one seeded trajectory, no re-roll.
- **Add passive logging only** (no gradient-bearing change): per-epoch exemplar-gate mean/std/saturation and gate-vs-ROI-scale correlation, so the BECAUSE clause is observable — the exact measurement gap flagged by N0004 and still open.
- The const-input sever (H0011) and the channel gate (H0009) must stay OFF on that node; neither belongs near a clean H0010 test. If the champion+gate node then wants a H0011-style control, do that as a SIBLING of that node, not inside it.

Synthesis/Lead booking note: three nodes (N0004, N0005, and this recommendation) all say the same thing — clean H0010 needs a champion child with exactly one switch. If it is not run, H0010 is dead inventory: two joint tests that book zero attributable evidence.

## 5. Verdict

- **H0011: identifiable, refuted.** The relevant contrast (sever conditioning, count for count, within-run) was measured; the confounded +0.202 sits under the 0.30 bar even in the worst-case reading. Whether "refuted" is interpreted as "severance costs ≤0.30" (direct) or "gate does not need its input" (mechanism-level, LOW-MED confidence, see §3), the falsifier was satisfied → ledger `contradicts` stands (conf 0.400).
- **H0010: not identified.** Both of its measured tests are joint-with-another-switch; this node yields **no attributable solo evidence in either direction** → ledger `neutral` (no conf move) is the correct, honest status. H0010 remains an untested-in-isolation hypothesis, not a tested-and-failed one.

Footnote for anyone tempted to read +0.202 against the parent as "H0010 is fine-ish": that quants the sum; §2 shows the H0010 share of that sum is at most small-negative/~0 even under the most favorable decomposition, and §4 is the only reading that settles it.