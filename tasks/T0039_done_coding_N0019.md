# T0039 — pending_coding — N0019_dino_tailup

**Status**: pending coding (claim by rename → `T0039_done_coding_N0019.md`)

## Task
Implement node `tree/nodes/N0019_dino_tailup` per its `idea.md`.

- **model.py**: verbatim copy of parent's `tree/nodes/N0010_dino_multilayer_long/model.py`
  (zero architecture change).
- **config.py**: verbatim N0010 `cfg` + exactly two added keys:
  `tail_reweight=True, tail_exp=-0.5`
  (sign-corrected vs sibling N0017 which used +0.5 and starved the dense tail:
  MAE 22.19, ratio 3.80). Per-image loss weight ∝ count^0.5, batch-mean-1 normalized.
- Flip `tree/tree.json` status N0019 → `"coded"`.

## Smoke
`python code/engine/train.py --node_dir tree/nodes/N0019_dino_tailup --smoke --epochs 2`
(local torch; else draft commit+push and run on server via ssh).

## Acceptance
Green smoke; config diff vs N0010 is ONLY the two tail keys; no commit/push by subagent.
Read `memory/failure_modes.md` before coding. Do NOT book hypotheses.jsonl (Synthesis owns it).

Hypothesis under test: **H0028** — RMSE/MAE <3.4 AND val MAE ≤21.53;
DISPROVED IF ratio ≥3.63 OR MAE >22.2.
