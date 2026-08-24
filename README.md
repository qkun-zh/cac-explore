# cac-explore — Hypothesis-Driven Multi-Agent Discovery for Crowd Counting

## Mission
> **≤32M total params achieving same-parameter-class SOTA MAE on FSC147 test.** Full fine-tuning allowed.

## Current Best
> **Effective MAE 19.18 / RMSE 66.37** — eval-only resolution routing on champion ckpt (zero training)
> **Trained champion: val MAE 20.44** — N0021_dino_partialft @ 23.26M params, 24min/GPU
>
> Frozen DINOv2-S reg4 @392 → multi-layer taps + area-prompt → adapter → conv head;
> blocks 10-11 unfrozen at lr×0.1; route N̂≥200 → re-read at 518px.

## The Method
Implements [HypoExplore](https://arxiv.org/abs/2604.12999): discovery as hypothesis-driven scientific inquiry.
Every experiment tests 1-2 falsifiable hypotheses against a Trajectory Tree (`tree/tree.json`)
and a Hypothesis Memory Bank (`memory/hypotheses.jsonl`). Confidence scores accumulate across
experiments; confirmed hypotheses become building blocks; refuted ones are never retried.

Deviations from the paper are explicit and recorded in AGENTS.md §7.

## Quick Start
```bash
git pull --ff-only && cat AGENTS.md STATE.md   # startup sequence lives in AGENTS.md §0
```

## Repo Map
| Path | Role |
|---|---|
| `AGENTS.md` | Protocol: roles, research cycle, hard rules, gates |
| `STATE.md` | Single-block session state (champion, queue) |
| `docs/PROTOCOL.md` | Extended protocol details · `docs/research_direction.md` owns the mission target |
| `code/engine/train.py` | Shared training engine |
| `code/selection/select_next.py` | Dual selection: parent (quality×avail), hypotheses (Thompson+epistemic) |
| `scripts/check_hypothesis.py` | Pre-booking gate: hypothesis format validator |
| `scripts/novelty_check.py` | Pre-registration gate: similarity retrieval vs past ideas |
| `scripts/calibration_report.py` | Confidence-calibration monitor (paper §4.3 analog) |
| `scripts/run_node.sh` / `collect_node.sh` | Server launch / collect |
| `scripts/install_key.py` | Server rotation onboarding |
| `tree/nodes/<ID>/` | One directory per experiment: idea.md, model.py, config.py, feedback/, synthesis.md |
| `memory/hypotheses.jsonl` | Append-only hypothesis ledger (create/evidence events) |
| `memory/failure_modes.md` | Diagnostic-agent learnings fed back to Coding Agent |
| `journal/events.jsonl` | Append-only operational audit log |

## Gates (never skip)
1. **Novelty**: `novelty_check.py` + structural judge before registering a node
2. **Format**: `check_hypothesis.py` before booking any hypothesis (max 2 new/node)
3. **Calibration**: `calibration_report.py` table into every synthesis.md

## Hygiene Rules
- hypotheses.jsonl and journal/events.jsonl are append-only
- STATE.md holds exactly ONE session block; old sessions go to journal/
- Mission target changes only via docs/research_direction.md + journal entry at session start
