# H0020 — Champion consolidation: remove dead PeakCal path, prove full reproducibility (SOLO consolidation card)

- Parent: `N0015_h0014` (live: val MAE 19.3431 @ep32, dense 330.81, sparse 5.814, test MAE 19.066; w_c = +0.148; temp pinned 0.07; w_p = 0.0017 verdict-recorded NULL).
- Type: SOLO consolidation / reproducibility card. NOT a mechanism hunt, NOT an improvement claim — stated plainly.
- Regime: backbone frozen, head-only training. Decoder `in_ch` untouched; `e` untouched; `fine` read via detach only.
- Protocol: v2 + fixed seed 20260830, 32ep/1800s, EMA eval, test-split eval via `eval_test`.
- Claim semantics: reproduction BAND (val MAE 19.3431 ± 0.30 → [19.04, 19.64]), not an improvement bar. §6 note: falsifier bars are semantics bound to the live parent number under the current protocol; confirmation/refutation thresholds and Eq.1 improvement-evidence semantics apply to improvement claims — this card asserts band-membership (identity preservation), so the Lead records the verdict as reproducibility evidence, not as improvement support. No confidence numbers are computed here.

## §0 — one-liner for `discovery hypo --new`

IF the dead PeakCal path (PeakCal class, use_peakcal flag, peak-gain block in SimPrior.forward, peakcal attach plus conditional temp-pin) is removed IN the N0015_h0014 champion retrained same-seed under the v2 protocol (seed 20260830, 32ep, EMA eval), THEN final EMA val MAE lands inside the reproduction band 19.3431 ± 0.30 ([19.04, 19.64]) with dense-tail inside [300, 361] and sparse inside [5.3, 6.3], BECAUSE at fresh init w_p = 0 exactly forces the removed peak gain to identity (exp(0) ≡ 1.0) so the cleaned forward is bit-identical to the parent's fresh-init forward and PeakCal's torch.zeros construction consumes zero RNG draws, leaving the retrain trajectory deterministic-identical. DISPROVED IF final EMA val MAE falls outside [19.04, 19.64] or dense-tail outside [300, 361] or sparse outside [5.3, 6.3] or w_c ≤ 0 or temp ≠ 0.07 or slice stats are incomplete on val and test.

## §1 — purpose (user directive)

The user directed consolidation of the champion: remove the dead PeakCal path, keep everything confirmed (SimPrior + cellcal + temp-pin), clean the code, prove FULL reproducibility with a fresh same-seed retrain, and publish comprehensive slice statistics — making this run the paper's canonical artifact. Its value is a clean, trustworthy reference implementation plus a same-seed reproduction certificate, not a lower MAE. No mechanism literature applies to this card (nothing new is claimed; the only numbers cited are the repo's own ledger figures for N0015_h0014), so the web survey is skipped and zero arXiv IDs are needed.

## §2 — exact cleaning spec

Flag-scheme decision: NO new flag. Deletion-only consolidation is the cleanest scheme — `use_peakcal` is removed entirely (not set false; the concept never existed mechanistically), and no `use_consol` switch is added because there is nothing left to gate. Kept flags: `use_simprior = true`, `use_cellcal = true`.

Removed (parent `model.py` line refs), class-of-change per line group:

1. `PeakCal` class (lines 269–281) — deleted in full. It owns one scalar `w_p = nn.Parameter(torch.zeros(()))`; `torch.zeros` performs zero RNG draws.
2. Head flag read `self.use_peakcal = _get(cfg, "use_peakcal", False)` (line 163) — deleted.
3. `CountingHead.forward` four-way simprior call block (lines 175–185) — collapsed to the single cellcal branch: `cond_map = cond_map + self.simprior(fine, e, use_cellcal=True, w_c=self.cellcal.w_c)`.
4. `SimPrior.forward` peak path: `use_peakcal`/`w_p` parameters (line 225 signature) and the entire peak-gain block (lines 240–251) — deleted; signature keeps `(fine, e, use_cellcal=False, w_c=None)`.
5. `Counter` attach `self.head.peakcal = PeakCal()` (line 312) — deleted, with its append-only RNG comment block.
6. Conditional temp-pin `if self.head.use_peakcal: self.head.simprior.temp.requires_grad_(False)` (lines 315–316, which duplicate the cellcal pin) — replaced by an UNCONDITIONAL pin: `self.head.simprior.temp.requires_grad_(False)` with a comment recording that the pin is the confirmed temp-pin hygiene carried over from N0015.

Kept exactly: SimPrior (qproj/kproj/temp/out, zero-init `out`), CellCal (`w_c`), the cellcal gain math in `SimPrior.forward`, the temp value 0.07, decoder `in_ch = D + cond_dim`, `e`/`fine` read-only usage, GCA, Counter attach order of all surviving modules.

Config delta (parent `config.toml` verbatim otherwise): the `use_peakcal = true` line (line 29) is REMOVED, not set false. `augment = false` stays as stored (v2 augment is applied via `--set` at launch, as for the parent). Seed stays 20260830.

RNG argument (rule 13): PeakCal construction is `torch.zeros(())` — zero RNG draws — and the `use_peakcal` flag is a non-module bool. Deleting the LAST-constructed module shifts zero earlier init draws, so append-only order is preserved trivially and every surviving module (Fuser, ExemplarEncoder, Condenser, Decoder, GCA, SimPrior qproj/kproj, CellCal) draws at its exact parent RNG positions. The removed forward ops are deterministic elementwise math on detached tensors with no RNG consumption.

Step-0 identity proof (airtight): at fresh init `w_p = 0` exactly → `arg = (0 · f).clamp(-3, 3) = 0` exactly → `gain = exp(0).clamp(0.7, 1) = 1.0` exactly (with the NaN fallback also defined as identity) → `ev * 1.0 ≡ ev` bit-for-bit. Hence the cleaned forward, which skips the block, is bit-identical to the parent's fresh-init forward on every input; fresh-init parameters are draw-identical by the RNG argument above. The retrain trajectory is therefore deterministic-identical up to floating-point association differences from the REMOVED ops (none exist — the removed ops contributed exactly identity). Conclusion: any verdict deviation outside fp noise is a harness finding, not a mechanism difference.

## §3 — why this is safe

Zero-draw removal: the deleted module consumed no RNG at construction and no RNG in forward, so initialization and data-stream RNG positions are untouched. Identity-at-init: the removed gain provably equals 1.0 at the fresh-init point from which the retrain starts (the parent's trained `w_p = 0.0017` is irrelevant — no weights are carried over; both runs start from `w_p = 0`). Temp-pin kept: the unconditional pin preserves the confirmed temp-pin hygiene (temp 0.07, the load-bearing half of the parent's win alongside cellcal `w_c = +0.148`) that the conditional pin previously provided. The only behavioral delta versus the parent retrain is the absence of bit-identical no-op multiplications.

## §4 — acceptance reads (all six required) + failure diagnoses

1. Final EMA val MAE inside [19.04, 19.64] (19.3431 ± 0.30).
2. Dense-tail (top1pct share proxy band) inside [300, 361] (330.81 ± 30).
3. Sparse MAE inside [5.3, 6.3] (5.814 ± 0.5).
4. `w_c > 0` at end of training (cellcal confirmed active).
5. `temp` exactly 0.07 (pinned, `requires_grad_(False)`).
6. Full slice stats (sparse / mid / dense + top1pct share via `eval_test`) complete on BOTH val and test.

Failure = any single miss → verdict REFUTED with diagnosis: either the dead code MATTERED (e.g. an unlisted coupling such as the conditional pin masking a second temp path, or an attach-order side effect — itself a publishable finding), or the harness is nondeterministic at same seed (also a finding — rerun forensics required, never a silent rerun). No partial pass: a band card lands inside on every read or it fails.

## §5 — protocol

Same-seed v2 pair: fresh retrain at seed 20260830, 32 epochs, 1800 s ceiling, EMA eval, v2 augment applied via `--set` at launch (config stored with `augment = false` exactly as the parent's). The LIVE parent numbers (val 19.3431, dense 330.81, sparse 5.814, test 19.066) serve as the BAND reference, not as an improvement bar to beat. NO futility: rule 16 is waived here because there is no improvement bar to be futile against — only a band to land inside — so a `futility_bar` slope gate is undefined for this card (it would halt a run for failing to approach a bar that does not exist); like a canonical baseline, this consolidation run must complete all 32 epochs and then receive full `eval_test` slice statistics on val and test. 32ep completion is mandatory.
