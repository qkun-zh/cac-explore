# Qualitative feedback — N0009_h0015 (H0015 solo, BMNet+-style dynamic per-exemplar channel gate)

Result: child best val MAE 23.6818 @ep23 (`result.json:5-6`) vs canonical parent 23.2932 @ep26 (`tree/N0001_champion/result.json:5-6`), delta +0.3886 worse; live bar (parent − 0.30) = 22.9932, missed by ~0.69. Ledger `contradicts` (conf 0.400) is the verdict; this note is mechanism fidelity + behavior reading only. Baseline note honored: parent numbers are the canonical noaug+EMA+seeded+deterministic run (seed 20260830); the older local v1 artifacts are stale history and are not used here (see `feedback/quant.md:8-16` for provenance).

## 1. Fidelity: implementation matches the pre-registration, with one unregistered fan-out

Pre-registered mechanism (`idea.md:7`): a small exemplar-conditioned gating MLP maps EACH exemplar embedding to its OWN per-channel weight vector, multiplied elementwise into that exemplar's token BEFORE the condenser cross-attention — motivated as per-exemplar channel-subspace selection (object-bearing channels up, context channels down per prototype).

Implementation (`model.py:120-139`, class `DynExemplarGate`):

- Shared MLP `Linear(d_model=256 → hidden=64) + GELU + Linear(64 → 256)` (`model.py:134`); hidden=64 wired at construction (`model.py:191`).
- Last layer zero-initialized, weight AND bias (`model.py:135-136`), forward `2.0 * sigmoid(mlp(e))` (`model.py:139`) → exactly 1.0 at init (sigmoid(0) = 0.5). The docstring states this explicitly (`model.py:124-127`).
- K-free / token-wise: `nn.Linear` acts on the last dim, so the same MLP maps each `[B, K, 256]` token independently with no K-dependence (`model.py:138-139`); the module header documents this (`model.py:121-125`).
- Hook point is exactly as registered: `e = e * self.dyn_exemplar_gate(e)` (`model.py:199-200`) placed after the exemplar encoder (`model.py:198`) and before the condenser (`model.py:201`).

Side effect — gated `e_mean` into GCA: `e_mean = e.mean(dim=1)` (`model.py:203`) is computed from the ALREADY-GATED `e`, so the GCA aux head (`Counter.forward`, `model.py:250-252`: `n_aux, bias` from `e_mean`, bias added to density) sees gated prototypes too. The pre-registered text (`idea.md:7`) scopes the mechanism to "before the token enters the condenser cross-attention" and never mentions GCA. Verdict: the SWITCH is clean (one flag, solo test), but its downstream fan-out is twofold — matching keys/values AND the global-count aux mean. That coupling is in-scope as far as "this is what the switch does," but out-of-scope as far as "this is what the hypothesis mechanism claims credit for." Any harm attribution must carry the qualifier: matching-path vs GCA-path harm is unseparated by this design (agrees with `feedback/causal.md:47-56,83-88`; my own reading of the same lines).

## 2. Init/identifiability: identity at step 0, so the +0.39 is learning dynamics, not init damage

At step 0 the gate is the exact identity (2·sigmoid(0) = 1.0 by construction), so "the gate damaged features at init" is arithmetically excluded. The deficit must come from what the gate LEARNS (or prevents learning) from step 1 on. Curve evidence (`feedback/quant.md:44-104`, same-seed paired `val/mae`, child TB `run/latest/tb/t/events.out.tfevents.1789819679...` vs canonical parent):

- Ep1 only lead: child −7.32 ahead at ep1, then behind for all 31 remaining epochs — identity lasts zero updates (`quant.md:83,158-162`).
- Early descent stall: worst gap ep4 (+4.44) — parent drops 49.44→39.54 while child stalls 50.54→43.98; one full epoch of early descent never recovered (`quant.md:85-87`).
- Persistent mid-run deficit: closest approach ep9 (+0.061); ep7–ep23 gap sits +0.06…+0.32, same sign, never crossing (`quant.md:88-89`).
- Late qualitative divergence: child best@23 then 9 straight non-improving epochs, monotonic +0.186 drift to ep32; parent best@26 with flat +0.043 tail bounce — plateau, not divergence. Tail gap widens +0.35 (ep24) → +0.53 (ep32) (`quant.md:90-96`).
- Train fit rules out "fits harder, generalizes worse": child train loss starts much lower (ep1 −52.99) but flips sign by ep9 and is almost uniformly HIGHER from ep11 (ep32 +0.20: 3.0339 vs 2.8345). Extra capacity + worse late train fit = optimization friction, not overfit-from-fitting-harder (`quant.md:100-104`).

Reading: two-phase harm — an early-trajectory distortion (ep2–ep4 stall the descent permanently) plus a late drift (gated run cannot hold its own best while the champion plateaus). Both point at optimizer dynamics under the fixed 32ep cosine budget, with a possibly-actively-harmful learned mask on top (gate statistics from the checkpoint would split "mask unused near 1.0" vs "mask diverged toward 0/2" — not run here, flagged for synthesis).

## 3. Interaction: per-channel key scaling vs the condenser's softmax attention

The condenser (`model.py:142-156`) calls `attn(norm1(tok), e, e)` (`model.py:154`): queries are LayerNormed, keys/values `e` are UN-normed (code comment `model.py:128-130` notes the scaling is therefore not cancelled). A per-channel mask `g(e_k) ∈ (0,2)^256` on each key/value token changes (a) key norms → query-key dot-product magnitudes → softmax sharpness over the K prototypes, and (b) value vectors → the decoded mixture. Because the mask varies PER CHANNEL, it rotates each prototype's matching DIRECTION, not just its weight — 256× the degrees of freedom of a scalar gate, all multiplicative at the highest-leverage point in the head (every attention query + the GCA mean).

Contrast H0010 scalar gate (queued N0011, `tree/N0001_champion/N0011_h0010/model.py:158-172`): `Linear(256,1)` with bias 3.0 (≈0.95 at init), `e * eg` with scalar `eg` per exemplar — direction-preserving; it rescales attention MASS across prototypes (which prototype to trust) without changing what any prototype means. Contrast H0009 image-global gate (refuted twice, `feedback/causal.md:113-118`): one channel mask from a pooled-fine global vector applied to ALL exemplars — cannot choose a per-exemplar subspace. H0015 fixes exactly H0009's conditioning gap (per-exemplar instead of global) and STILL fails, so the failure cannot be blamed on global-vs-local conditioning. The remaining axis is per-CHANNEL freedom itself.

## 4. Pluggability: OFF restores the champion; diff surface is minimal

- OFF path: parent `model.py:143-168` has no gate class, no flag, no hook; parent `config.toml` has NO `use_dyn_exemplar_gate` key at all (keys end at `xscale_size`, `config.toml:23-26`), so `_get` defaults to False. Child guards both construction (`model.py:189-191`) and forward (`model.py:199-200`) on the flag — with the switch absent the forward is the champion's exactly (docstring `model.py:12-19`).
- ON diff parent→child: exactly one config line (`config.toml:27` `use_dyn_exemplar_gate = true`; server hparams snapshots differ ONLY in this key per `quant.md:39-42`) + one new class (`model.py:120-139`) + three head lines (flag read `model.py:189`, conditional construction `model.py:190-191`, gated multiply `model.py:199-200`) + docstring. Rule-13 append-last holds: the gate is constructed AFTER fuser/exemplar/cond/decoder (`model.py:182-191` order), dropout is 0, data stream is seed-pinned — shared-module init draws identical to the parent's.
- Cost: +33,088 params (256·64+64 + 64·256+256), checkpoint +133,676 B, wall +4.1 s (+0.24%) — compute-invisible (`quant.md:31,125-131,147-149`). The harm is representational/optimizational, not budgetary.

## 5. Pattern: second exemplar-gating variant to fail; scalar H0010 is the last clean cut

Family scoreboard: H0009 (image-global channel gate) refuted twice; H0015 (per-exemplar channel gate) now decisively missed solo, same-seed, single-switch, full-horizon. H0010 (scalar per-exemplar, K×1) was never cleanly tested — N0004 joint confounded, N0005 neutral on severed conditioning — and stays at conf 0.400 with N0011 queued as its designated decider (`feedback/causal.md:126-132`).

What per-CHANNEL failing suggests about where exemplar-side interventions go wrong: post-ExemplarEncoder+XScale tokens are calibrated matching keys AND (via `e_mean`) the GCA prototype mean — dual-use. A per-channel mask perturbs both blast radii at once while rotating matching direction under a tiny-K softmax (1–3 prototypes), where small key-norm changes swing attention sharply and gradient paths into the condenser/exemplar-encoder shift from step 1. The richer K×C class is harder to optimize and easier to overfit than the selection problem it claims to solve; the observed early stall + late drift + worse late train fit is exactly that signature. The remaining untested cut is direction-preserving K×1 selection (down-weight untrustworthy prototypes, leave trustworthy ones' subspaces untouched) — if N0011 supports while N0009 fails, the lesson is "selection helps, subspace rotation hurts/is unlearnable in 32ep"; if N0011 also fails, the lesson upgrades to "any pre-condenser exemplar perturbation hurts in this head/budget — leave exemplar tokens alone and intervene query/similarity/decoder-side (e.g. extend H0014's direction), never re-encode gating" (pre-registered joint readings in `feedback/causal.md:152-184`, endorsed here).

- Fidelity holds at the switch level (identity-init K-free per-exemplar channel gate exactly as pre-registered); the gated-`e_mean`-into-GCA fan-out is an unregistered side effect that confounds matching-path vs aux-path attribution.
- Behavior shows early-descent stall plus late monotonic drift with worse late train fit — optimization friction under the fixed cosine budget, not init damage and not fit-harder overfitting.
- Per-channel (K×C) failure after global-channel failure isolates channel-subspace freedom as the suspect; direction-preserving scalar (K×1, N0011) is the only exemplar-gating cut left standing.
