# H0022 idea — `use_margin`: margin-sign-selective per-cell energy gain (wipe-rescue repair of the confirmed cellcal actuator)

Parent: `tree/N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014` (live N0015, val 19.3431 @ep32 v2). SOLO switch `use_margin`, default false; replaces cellcal's sign-blind gain via a call-site-exclusive margin path (one-line config delta: `use_margin=true`; `use_simprior`/`use_cellcal`/`use_peakcal` stay true). Frozen backbone, head-only, ≤32M (one zero-init scalar added), decoder in_ch 192 untouched. Protocol v2, seed 20260830 fixed. Triple conjunctive bars (§6 semantics, bound to live parent 19.3431): final EMA val MAE ≤19.0431 (≥0.30 below live) AND dense-tail gt>500 MAE <330.81 AND sparse gt<50 MAE ≤5.45. Launch with `--set futility_bar=19.0431`.

Full authored card (survey ids, twin pre-emption, attach-point diffs, reads/failure modes): `local/research/h0022_idea_DRAFT.md`.

## §0 Hypothesis one-liner (for `discovery hypo --new`)

IF a margin-sign-selective per-cell energy gain (`use_margin`) that multiplies only the positive evidence channels of the confirmed SimPrior readout by `exp(w_m·log1p(mhat))` under a stop-gradient sign mask while routing negative-evidence cells through at identity, as a replacement for the sign-blind cellcal gain, IN the live N0015 head on FSC147 v2 at seed 20260830, THEN final EMA val MAE drops at least 0.30 below 19.3431 with dense-tail gt>500 below 330.81 and sparse gt<50 at or below 5.45, BECAUSE the measured dense wipe is a sign-blind-gain artifact — the confirmed cellcal gain `exp(w_c·log1p(mhat))` multiplies predominantly negative similarity evidence (top1 mean ≈ −0.12, H0013/H0019 measured) on high-energy dense cells, so gain>1 deepens a negative contribution to cond_map instead of adding mass, and margin-sign-selective amplification restores the mass-adding invariant of the winner by never amplifying negative margins while keeping the dense energy gain on positive margins. DISPROVED IF final EMA val MAE is not at least 0.30 below 19.3431, or dense-tail is not below 330.81, or sparse gt<50 exceeds 5.45, or the negative-margin amplification fraction does not collapse to identity (gain ≈ 1) on the wiped images.

## §1 Measured motivation: the bimodal dense wipe

val per-image accounting (N0015): sparse gt<50 (n=895, MAE 5.81) AE 5200 = 21%; mid 50–500 (n=374, MAE 37.5) AE 14025 = 56%; dense gt>500 (n=17, MAE 330.81) AE 5624 = 23%. <18 needs AE < 23148 (cut ≥1701).

- The dense tail is BIMODAL with zero count-anchoring: among gt≥300 (38 imgs) exactly 11 are WIPED to pred/gt < 0.5 — 840(637→179), 851(320→145), 865(1022→266), 935(2092→843), 3428(458→104), 3481(431→22), 3482(315→28), 3484(356→27), 3487(321→99), 3488(431→83), 7656(1231→381). Same gt≈431 predicts either 421 (ratio 0.96) or 22 (ratio 0.05).
- Pairwise control: CountingDINO on the SAME 11 images only gently undercounts (22→235, 28→234, 83→191...) → the wipe is base-specific, living in exemplar-encoding → similarity-evidence → (cellcal×gain) → cond interface, NOT a shared input wall.
- Mechanism authored: sign-blind gain amplifying predominantly-negative similarity evidence (top1 mean −0.12 measured, H0013/H0019) at high-energy dense cells → gain>1 deepens the negative cond contribution instead of adding mass (a "boosting silent-collapse on negative margins").

## §2 Mechanism (see draft §2 for full math + line refs)

Inside `SimPrior.forward`, a new `if use_margin:` block placed BEFORE the parent `if use_cellcal:` block (parent blocks byte-identical; margin path is call-site exclusive):

```
gain = torch.exp((w_m * torch.log1p(mhat)).clamp(-3, 3))   # one zero-init scalar w_m
pos  = (ev > 0).float().detach()                            # hard stop-gradient sign mask
ev   = torch.where(pos > 0.5, ev * gain, ev)                # amplify positives only; negatives = parent identity
```

- Negative branch is a NULL-OP (identity routing, no suppression, no ReLU(neg−pos)); mass-adding invariant by construction in a global undercount regime.
- New `MarginCal` module (mirror of `CellCal`): one zero-init scalar, constructed LAST (append-only RNG). Temp-pin condition extended to `use_peakcal or use_margin`.
- Flag off → byte-identical to live N0015. Flag on at step 0 → gain=1 exactly, `torch.where` selects `ev` → zero-output delta vs parent fresh-init.
- Untouched: qproj/kproj (no second grad path), `e` read-only (no pre-condenser gating), decoder, condenser, GCA, temp lines.

Mechanism reads (from draft §4): R1 sign-selectivity engaged (w_m ≥ +0.10, negative cells gain ≈ 1.0, positive p99/p1 ≥ 1.3x); R2 wipe rescue — of the same 11 wiped images ≥7 reach ratio ≥0.65, mean ratio 0.33→≥0.60, and no image now ≥0.8 drops below 0.6; R3 sparse ≤5.45; R4 triple bars. Failure modes F1 quiet-null / F2 rescue-with-slippage / F3 wipe-not-gain-caused (→ exemplar-side successor) / F4 sparse overshoot.

## §3 Verification record (fill as run proceeds)

- novelty gate: exit 0 (top sim H0018 0.392 < 0.82), `local/research/h0022_novelty.json`.
- booking: N0023_h0022 via `discovery hypo --new --solo` on 2026-09-22.
- coding: (model.py/config.toml delta + CPU byte-identity/engage/determinism/temp-pin checks — Coding Agent)
- smoke: (server GPU)
- launch: `run_node N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014/N0023_h0022 --set augment=true --set futility_bar=19.0431`