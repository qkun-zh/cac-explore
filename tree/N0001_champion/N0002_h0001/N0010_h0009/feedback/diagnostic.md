# Diagnostic — N0010_h0009 (H0009 `use_densexp`) · done-status full check

- **Node:** N0010_h0009 · **Parent:** N0002_h0001 (22.564146041870117 @ep30, bar 22.26) · **Status:** `done` (`code: ok`, `budget_hit: false`, 32/32 epochs, best **24.049957275390625 @ep32**, `elapsed` 1789.418 s / `elapsed_s`/`train_seconds` 1794.9 s, shas `8f8d14b1c6e058f0` / `70410fe91dd6d346`).
- **Question asked:** gate missed (best 24.0500 vs bar 22.26 → +1.7900; vs parent +1.4858), so no lean path — full check: edge-best @ep32 trustworthiness, budget/harness, NaN, TB completeness.
- **Verdict: CLEAN DONE — trustworthy number, H0009 REFUTED (null-and-dragging).** No truncation, no harness discount, no NaN, no re-run. Details below.
- **Sources read (all verified, no recollection):** local `info.json` / `result.json` / `idea.md:29` / `config.toml` / `model.py` for N0010; parent N0002 `result.json`; `feedback/{quant,qual,causal}.md` (adopted numbers only where they trace to TB/server artifacts); sibling N0009 `feedback/diagnostic.md` as method precedent. Server: single short ssh commands only (no sleep-loops, no tmux, no new TB dump — full 32-pt series already dumped by quant/qual): `ls -R run/` + `cat tb/t/hparams.yaml`, and `grep -c` anomaly scan + `tail` of `/tmp/train-b4-n0010.log`. Checkpoint norms / functional probe / train-loss values below are causal's measurements, cited as such — not re-probed here.

## 1. Edge-best @ep32 — trustworthy plateau-edge, not rescuable

- Best = final @ep32 (24.049957 = final 24.049957, delta +0.0000). Last-5 range ep28–32: 24.065378 − 24.049957 = **0.0154**; late crawl ≈ −0.0039/ep (quant §3, adopted — monotone 32/32 descent, every epoch improves, gains ~0.004/ep by ep28).
- At that rate closing the 1.79 bar gap takes ~460 further epochs. The edge-best is a flat floor ~1.79 above the bar, not a lucky dip — and not a still-descending optimum that matters.
- Position contrast: parent best @ep30/32 (interior) vs child best @ep32/32 (edge), but with zero uptick anywhere there is no overfit-selection confound by construction. Late-best at 100% of schedule on a converged floor = finished run that lost, not an interrupted one (qual §1).
- Never-tracking corroboration (quant §2): child trails from ep01 (+117.27) narrowing monotonically to +1.49; child ep32 sits where parent was between ep12–ep13, then plateaus. No epoch within 1.7 of the bar; no late recovery.

## 2. Budget / harness — ruled out

- `budget_hit: false`, `code: ok`, 32/32 epochs, 1794.9 s = **5.1 s UNDER τ_max 1800**. +39.9 s (+2.3%) vs parent 1755.0 s; final-epoch step 45 s @ 5.03it/s (qual §3) — idle-plateau epochs, not expert-overhead distress. Near-ceiling elapsed is not a near-timeout.
- Server canonical protocol confirmed via `tb/t/hparams.yaml`: seed 20260830, augment false, deterministic true, AdamW/cosine/AMP/MSE+0.4-L1 — matches child `config.toml:9,33`. Server config diff vs parent is exactly one switch `+ use_densexp = true` with shas matching `result.json` (causal §5, `sha256sum` re-verified on server).
- Construction verified append-only (`model.py:279-284`, after all parent modules; out/gate zero-init `model.py:243-248`; flag-off op-identical `model.py:178-179`): step-0 ≡ parent, shared-module init draws preserved (AGENTS rule 13). Decoder `in_ch` 192 untouched, ~4,322 added params (~31.349M ≤ 32M).
- Server `run/latest/` holds `best.pth` (120M) + `tb/` — no `result.json` on server is expected (result lives locally + in log tail, which re-prints the identical JSON: best 24.049957 @32, budget_hit false). No truncation confound.

## 3. NaN — clean

- Server log single-grep scan: **0 hits** for `Traceback|BudgetStop|CUDA out of memory|Misconfiguration`; **0 hits** for `nan|inf` (case-insensitive).
- All three 32-pt TB series (`val/mae`, `val/rmse`, `train/loss_epoch`) 32/32 finite, no spikes, no U-turns (quant §6, qual §1–2). `train/mae` 32/32-NaN on BOTH parent and child = standing logger artifact (quant §4, qual §3), non-deciding, still unfixed.
- No N0007-style stall attractor (~14/~48), no N0008-style synchronized NaN wall (best @ep9 + 23-NaN tail), no N0005/N0006-style timeout. Boring-healthy shape.

## 4. TB completeness — complete

- Server `tb/t/` holds `events.out.tfevents.1789903448.*` (26K) + `hparams.yaml`. Full 32-pt `val/mae`, `val/rmse`, `train/loss_epoch` series already dumped by quant/qual (identical values cross-checked by causal) — every epoch validated, best @ep32 has no following point because it IS the final scheduled epoch on a flat floor (see §1; a following point cannot move a 1.79 miss at −0.004/ep).
- Text-log gap noted (qual §2): stdout carries no per-epoch scalars (rich bars overwrite; only final `Epoch 31/31 … val/mae: 24.050` survives) — trajectory comes from TB scalars only. From stdout alone this reads "smoke ok → done, 24.05". No action; logged so the next reader does not mistake log-thinness for incompleteness.
- Log tail confirms the number independently: `24.050` + the `code: ok` result block.

## 5. Implication for the H0009 verdict (for Synthesis; evidence booking stays Lead-only)

- Global bar: 24.0500 vs required ≤ 22.26 → **misses by +1.7900** (~50–60× the ~0.03 same-seed paired-contrast precision), worse than parent by **+1.4858**. The MAE conjunct of the dual falsifier (`idea.md:29`) fails outright on a clean late-plateau number with no truncation confound.
- MAE-worse / RMSE-better split does not help quantitatively (quant §6): child RMSE best=final 84.1496 @ep32 vs parent best 88.9234 @ep28 / final 89.2400 (−4.77 best-best, −5.09 final-final; monotone 32/32 descent, no N0008-style divergence). Bar is written in MAE; ratio 3.95 → 3.50 is a distribution-shape observation for causal, not a rescue.
- Mechanism (causal §§2–4,6, cited not re-measured): Lens-3 cheap-falsification branch fired cleanly — gate never opens (≤0.15 incl. gt=2092; 0.072 sparse → 0.145 dense, relatively 2× but absolutely closed), expert trunk decayed below init (−18%/−28% on exp1/exp2), out norm 0.0090 at zero; r RMS 3 orders below operating range, base-vs-gated sums <2%. Yet hosting the dead branch still cost +1.49 with temp collapse 9.0× toward clamp (0.004367 vs confirmed 0.03932), `simprior.out` +80%, broad head drift (+17.4% decoder agg, +17.6% fuser, +24.0% GCA), 2.04× train stall (final 5.345 vs 2.618). Readout NOT left undisturbed.
- Dense-tail conjunct (gt>500 mean |Δ| vs 511.7): unevaluated at scale (no per-image dump); n=1 anecdote (gt=2092: base 301.6 → gated 305.0, still massively undercounted) gives it no support. Global-bar miss alone decides.
- Family update (causal §6, endorsed): N0009's ban is confirmed in outcome but too narrow in mechanism — N0010 had r≈0 with full detach and still paid 1.49 with the same temp-to-floor + re-amplification + stall attractor (now 4/4 trainable head perturbations of N0002 collapsing temp below 0.0393: N0006 0.0214, N0008 0.0255, N0009 0.0033, N0010 0.0044). Extended rule: **no trainable post-decoder additive on the density path, however detached, however private, however small.**
- Recommendation: book H0009 `contradicts` at **w ≈ 0.85** (clean full-schedule run, decisive miss ~50× precision, mechanism-measured null + 9× temp-collapse signature; docked below the 0.90 catastrophic/NaN class per the N0006 +1.02 precedent — causal §6). Weight choice is Lead-only.

## 6. What NOT to do

- Do NOT re-run N0010 (AGENTS §5.1 — outcome shopping; number is best=final on a flat floor with full completion).
- Do NOT discount the miss as edge-best fluke, near-timeout, or NaN-adjacent (ruled out §§1–3).
- Do NOT book "RMSE improved so partial credit" — bar is MAE; the split is recorded as shape observation only.
- Do NOT retry "smaller r / harder detach" post-decoder additives — the null member already failed (§5 extended rule).
- Seed stays 20260830; no cross-seed anything (AGENTS §6).
