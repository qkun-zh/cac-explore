# cac-explore — Hypothesis-Driven Discovery for Crowd Counting

> **Mission**: ≤32M total params, same-parameter-class SOTA MAE on FSC147 test.
> **Current regime**: FROZEN backbone + pluggable head parts (structural/paradigm innovation only; §5.14).

## Champion (today)

| Node | Config | Val MAE | Params | Recipe |
|---|---|---|---|---|
| **N0054_xscale_exemplar** | GCA + **XScale** multi-scale exemplar, frozen DINOv3-ConvNeXt-Tiny | **19.647** (RMSE 74.05) | 31.32M | frozen h2@1/8+h3@1/16 → CountingHead → GCA(global) + XScale(coarse exemplar summary) |

Lineage: N0021(20.44, partFT) → N0026(19.18 eval@448) → N0036(20.49) → N0051 GCA-only(20.599, frozen) → **N0054 19.647**.

## Method

Implements [HypoExplore (arXiv:2604.12999)](https://arxiv.org/abs/2604.12999): discovery as hypothesis-driven scientific inquiry. Every experiment tests 1-2 falsifiable hypotheses against a Trajectory Tree (`tree/tree.json`) + Hypothesis Memory Bank. Confirmed hypotheses become building blocks; refuted ones are never retried. Deviations from the paper are recorded in AGENTS.md §9.

## Frozen-backbone pluggable rule (§5.14)

Every component bolted onto the frozen backbone is an **independent, self-contained module**. Components may serialize (串联) or parallelize (并联) but never couple. Coupling is limited to (a) frozen-backbone features and (b) exemplar embeddings — the shared stable interfaces. Single-switch ablations required: toggling a component must not touch another. Non-pluggable designs are rejected before smoke.

**Empirical lesson (2026-08-27)**: GCA (global log-count aux, bias-injection) is the only surviving density-path aux. Feature-modulators and density-bias additions (DDCA, RGA, SALF, FILM, cross-attn, bg-token, MoE) all *degrade* a near-optimal condenser under 30ep. The **exemplar-embedding interface** proved the real lever — XScale (coarse multi-scale exemplar summary) is the first pluggable part to *beat* GCA-only.

## Operating Modes

- **Free-Research** (default): Lead runs the full discovery cycle autonomously under gates/hard rules.
- **User-Guided**: user directives override protocol defaults; records back-filled.

Active mode lives in `STATE.md`.

## Quick Start

```bash
git pull --ff-only && cat AGENTS.md STATE.md   # startup sequence in AGENTS.md §2
```

## Repo Map

| Path | Role |
|---|---|
| `AGENTS.md` | Protocol: modes, roles, cycle, hard rules, gates |
| `STATE.md` | Single-block session state (mode, champion, queue, gotchas) |
| `docs/research_direction.md` | Owns the mission target · PROTOCOL extends the protocol |
| `code/engine/train.py` | Shared training engine (engine contract §below) |
| `code/selection/select_next.py` | Dual selection (parent, hypotheses) |
| `tree/nodes/<ID>/` | One dir per experiment: idea.md, model.py, config.py, feedback/, synthesis.md |
| `tree/archive_*/` | Archived (retired) lineage — not active |
| `tree/tree.json` | Node registry (status/best/quality) |
| `memory/` | Hypothesis ledger, failure modes, analyses |
| `journal/events.jsonl` | Append-only operational audit log |
| `scripts/` | Gates + server ops |

## Engine contract (frozen)

- `config.py` → `cfg=dict(...)`; `model.py` → `build_model(cfg)` → `forward(imgs,bboxes[,bboxes3])` → `{"density", optional "n_aux"}`.
- Engine loss = MSE(dens, gt_d) + w_cnt·L1(sum(dens), gt_c); **only `out["density"]` is used** in the loss/gradient.
- Asserts `params ≤ max_params_M`.

## Gates (never skip)

1. **Novelty** (`novelty_check.py` + structural judge) before registering a node.
2. **Format** (`check_hypothesis.py`) before booking any hypothesis (≤2 new/node).
3. **Calibration** (`calibration_report.py`) table into every synthesis.md.

## Hygiene

- `hypotheses.jsonl` + `journal/events.jsonl` append-only. STATE.md = one session block. Mission target changes only via `docs/research_direction.md` + journal line.
- Remote sync via `scp` (server = source of training runs; never `git pull` — proxy fails).
- Server creep hygiene: `/data/runs` holds only active lineage runs; stale runs archived.
