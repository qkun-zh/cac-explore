# H0023 idea — `use_antimatch`: fixed stop-gradient substrate swap on the anti-match minority (zero-param wipe rescue)

Parent: `tree/N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014/N0024_h0023` ← N0015_h0014 (live N0015, val 19.3431 @ep32 v2). SOLO switch `use_antimatch`, default false; on images whose whole-image top1 cosine mean is negative, the two evidence channels fed to the confirmed zero-init `out` projection are swapped from the anti-phase similarity tensors to the in-phase mean-1 fine-energy field `mhat`. **ZERO new parameters** (no module, no RNG draw → strongest append-only compliance). One-line config delta. Temp-pin unchanged.

Full authored card (survey ids, twin pre-emption, attach diffs, reads): `local/research/h0023a_idea_DRAFT.md`.

## §0 Hypothesis one-liner (for `discovery hypo --new`)

IF a fixed stop-gradient route (use_antimatch) swaps, on images whose whole-image top1 cosine mean is negative, the two evidence channels fed to the confirmed zero-init out-projection from the anti-phase similarity tensors to the in-phase mean-1 fine-energy field mhat while leaving the cellcal gain and every other parent line byte-identical, IN the live N0015 head on FSC147 v2 at seed 20260830, THEN final EMA val MAE drops at least 0.30 below 19.3431 with dense-tail gt>500 below 330.81 and sparse gt<50 at or below 5.45, BECAUSE the dense wipe is a measured anti-match artifact (wiped-set top1 mean −0.258 and 66.7 percent negative cells; post-gain evidence sum −5301), so the sign-blind cellcal gain deepens negative similarity evidence instead of adding mass, and on exactly that anti-match minority the energy field is the only in-phase mass signal the gain/decoder interface accepts, converting the identical confirmed gain from negative-deepener to positive mass source without touching the healthy majority. DISPROVED IF final EMA val MAE is not at least 0.30 below 19.3431, or dense-tail gt>500 is not below 330.81, or sparse gt<50 exceeds 5.45, or fewer than 7 of the 11 measured wiped images reach pred/gt at least 0.65 with the routed post-gain evidence sum not non-negative.

## §1 Measured motivation

- Wiped dense scenes carry net NEGATIVE exemplar-similarity evidence: top1 mean −0.258, 66.7% cells negative (val dump, N0015 best.pth). Phenomenon spans gt≥250 (12/12 ratio<0.5 images at gt[250,500) → 10 have top1_mean<0; 71% precision selector).
- cellcal (w_c=+0.148) multiplies negatives (+1.098x), deepening the wipe evidence sum −4924 → −5301 (~377 AE-units) — HANDOFF diagnosis item (c).
- Same gt≈431 predicts 421 or 22 (scene-conditioned, not count-anchored); CountingDINO on the same 11 images undercounts gracefully (22→235) — wipe lives in our simprior substrate, not the input.

Mechanism: where the matched filter (S=⟨q,p⟩) is anti-phase to mass (E[S | mass]<0), feeding S into a mass-adding gain adds NEGATIVE mass; energy field m=mean_c(h²)·≥0 has ⟨m,ρ⟩≥0 for ρ≥0 — routing to it is the cheapest phase-correcting operator. Hard per-image route on aggregate top1-mean sign, stop-gradient (no gradient to qproj/kproj/fine from routed images).

## §2 Mechanism (exact; after `ev = cat([top1,cons],1)`, BEFORE cellcal block)

```
if use_antimatch:
    m    = fine.detach().mean(dim=1, keepdim=True).clamp_min(0)
    mbar = m.mean(dim=(2,3), keepdim=True)
    mhat = (m / (mbar + 1e-6)).clamp(0, 10)
    anti = (ev[:, :1].mean(dim=(1,2,3), keepdim=True) < 0).float().detach()   # (B,1,1,1)
    en_ev = torch.cat([mhat, mhat], dim=1)
    ev    = anti * en_ev + (1 - anti) * ev                                    # hard swap; 0 keeps parent ev
```

- Routed images: ev becomes a detached positive substrate → cellcal gain is pure mass-add; qproj/kproj/fine get zero gradient from them. Non-routed: byte-identical parent tensor+gradient.
- Flag off → branch never taken, executed lines byte-identical to parent. Step-0 flag-on: `out` zero-init → `cond_map+0`, identical output.
- Nothing new trains: `ev`, `w_c`, `out` are all parent-owned. Bounds: mhat clamp [0,10] (peakcal hygiene).

Mechanism reads (draft §4): R1 routed-evidence-sum non-negative on ≥90% of routed images & ≥7/11 wiped reach ratio≥0.65; R2 sparse ≤5.45; R3 triple bars; no new wipes (any routing-caused dense drop >3% refutes). Failures: F1 route-ineffective (wipes stay wiped); F2 route-too-coarse (whole-image gate misroutes healthy majority); F3 KEEP degradation.

## §3 Verification record (fill as run proceeds)

- novelty gate: exit 0 (top sim H0022 0.579 < 0.82), `local/research/h0023_novelty.json`.
- booking: N0024_h0023 via `discovery hypo --new --solo` on 2026-09-22.
- coding: (delta + CPU checks — Coding Agent)
- launch: `run_node N0024_h0023 --set augment=true --set futility_bar=19.0431`