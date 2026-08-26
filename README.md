# cac-explore — Hypothesis-Driven Multi-Agent Discovery for Crowd Counting

> **Mission**: ≤32M total params achieving same-parameter-class SOTA MAE on FSC147 test. Full fine-tuning allowed.

## Status

| Result | Value | How |
|---|---|---|
| Trained champion | val MAE **19.15** | CAC-D simplified (cnt+density losses only), ablation Ep16 |
| Effective best | MAE **19.18** / RMSE **66.37** | Eval-only resolution routing on N0021_dino_partialft (zero training) |

Champion recipe: frozen DINOv2-S reg4 @392px → multi-layer taps + area-prompt → adapter → conv head; blocks 10-11 unfrozen at lr×0.1; route N̂≥200 → re-read at 518px.

## Method

Implements [HypoExplore](https://arxiv.org/abs/2604.12999): discovery as hypothesis-driven scientific inquiry. Every experiment tests 1-2 falsifiable hypotheses against a Trajectory Tree (`tree/tree.json`) and a Hypothesis Memory Bank (`memory/hypotheses.jsonl`). Confidence accumulates across experiments — confirmed hypotheses become building blocks, refuted ones are never retried. Deviations from the paper are recorded in AGENTS.md §9.

## Operating Modes

- **Free-Research** (default): the Lead agent runs the full discovery cycle autonomously under protocol gates and hard rules.
- **User-Guided**: user directives override protocol defaults — everything follows the user's direction.

The user sets the mode; it is recorded per-session in `STATE.md`.

## Quick Start

```bash
git pull --ff-only && cat AGENTS.md STATE.md   # startup sequence lives in AGENTS.md §2
```

## Repo Map

| Path | Role |
|---|---|
| `AGENTS.md` | Protocol: modes, roles, research cycle, hard rules, gates |
| `STATE.md` | Single-block session state (mode, champion, queue) |
| `docs/research_direction.md` | Owns the mission target · `docs/PROTOCOL.md` extends the protocol |
| `cac_d/` | Active experiment line: simplified cnt+density model (current champion) |
| `code/engine/train.py` | Shared training engine |
| `code/selection/select_next.py` | Dual selection: parent (quality×avail), hypotheses (Thompson+epistemic) |
| `tree/nodes/<ID>/` | One directory per experiment: idea.md, model.py, config.py, feedback/, synthesis.md |
| `memory/hypotheses.jsonl` | Append-only hypothesis ledger (create/evidence events) |
| `memory/failure_modes.md` | Diagnostic-agent learnings fed back to the Coding Agent |
| `journal/events.jsonl` | Append-only operational audit log |
| `scripts/` | Gates (`check_hypothesis.py`, `novelty_check.py`, `calibration_report.py`) + server ops (`run_node.sh`, `collect_node.sh`, `install_key.py`) |

## Gates (never skip)

1. **Novelty** — `novelty_check.py` + structural judge before registering a node
2. **Format** — `check_hypothesis.py` before booking any hypothesis (max 2 new/node)
3. **Calibration** — `calibration_report.py` table into every synthesis.md

## Hygiene

- `hypotheses.jsonl` and `journal/events.jsonl` are append-only
- STATE.md holds exactly ONE session block; old sessions go to `journal/`
- Mission target changes only via `docs/research_direction.md` + journal entry at session start
