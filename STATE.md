# STATE — Current Situation

**Mission**: ≤32M params, same-param-class SOTA MAE on FSC147 test. Full FT allowed.
**Stage**: gen-6 clean restart — incremental improvement from champion 21.53
**Champion**: N0010_dino_multilayer_long = **21.53** val MAE (archived in tree/archive_gen0_5/)
**Blockers**: none — server OK, GPU free

## Strategy: Incremental Steps Toward Lower MAE
Not trying to jump to ≤4. Instead, make ONE improvement per node, verify it works,
then stack confirmed improvements. Each step must be small, testable, and reversible.

## Confirmed Levers (from gen0-5)
| Lever | Gain | Source |
|---|---|---|
| DINOv2-S substrate | −15.3% | N0007 H0014 ✓ |
| Multi-tap(6,11) + gate | −6.1% | N0010 H0017 ✓ |
| Area-prompt conditioning | included | N0005 H0008 ✓ |

## Failed Levers (do NOT retry)
Per-token gate · Huber loss · high-res output · seqcount AR · tail-reweight ±
proto-iterative · scale-aware deformable · full-FT at lr=1e-3 · point detection threshold

## Next Step (ONE thing)
N0021: Unfreeze backbone with backbone_lr=1e-4 (0.1× of head lr=1e-3).
This is the single most impactful remaining lever. Everything else stays champion verbatim.

## Active Tasks
- N0021_dino_fullft coded+ready to launch
EOF_MARKER_NOT_NEEDED
