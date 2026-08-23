# cac-explore — Hypothesis-Driven Multi-Agent Discovery for Class-Agnostic Counting

## Mission

> **Build an innovative CAC model with ≤32M total parameters achieving same-parameter-class SOTA MAE on FSC147 test.** Full fine-tuning allowed, any architecture.

## The Method (READ THIS FIRST — this is not generic A/B testing)

This system implements **HypoExplore** (arXiv:2604.12999): automated scientific discovery via **hypothesis-driven exploration**. It is NOT grid search, NOT ablation study, NOT architecture mutation. It is structured scientific inquiry.

### Core Philosophy

Every experiment tests 1-2 explicit **falsifiable hypotheses**. A hypothesis has the form:

```
IF [architectural choice] IN [scope], THEN [predicted effect], BECAUSE [mechanism].
DISPROVED IF [criterion].
```

Results update a **confidence score** per hypothesis (0→1). Confirmed hypotheses (>0.75) become building blocks for future designs; refuted ones (<0.25) are never retried. Uncertain ones get tested from new angles.

### What Makes This Different From Tuning

| Grid Search / Ablation | HypoExplore |
|---|---|
| Try random combinations | Test specific causal claims |
| No memory between runs | Confidence accumulates across ALL experiments |
| Parent = best previous | Parent selected by quality × unexplored-hypothesis-availability |
| Change many things at once | 1-2 targeted changes for causal attribution |
| Success/failure binary | Rich evidence type (supports/contradicts) + strength |
| Human decides what to try next | Thompson sampling + epistemic value guide exploration |

### The Trajectory Tree

Each node = one architecture + its experimental result + parent link. Lineage preserves full evolutionary history. Branches that repeatedly fail get pruned by selection; branches with confirmed hypotheses get expanded.

### Root Bootstrap (Generation 0)

Before any evolution, generate K=4 **fundamentally different paradigms** (e.g., density regression vs detection vs sequence generation vs retrieval). Each root explores a different corner of design space. The best paradigm wins through natural selection; insights transfer across lineages.

## Non-Negotiable Rules

1. **Lead is an orchestrator**, NOT a worker. Idea / Coding / Feedback / Synthesis run as INDEPENDENT subagents. Lead only dispatches, monitors GPU, collects results, owns git/tree/journal/memory.
2. **Research BEFORE design.** Never propose architecture without understanding SOTA. Use websearch extensively.
3. **One card = one subagent = one fresh context.** Never batch multiple roles into one agent.
4. **Never idle while GPU runs.** Dispatch next Idea/Coding in parallel.
5. **Early-stop unpromising runs.** If same-epoch trajectory ≥+1.5 worse than parent at ep16+, kill.
6. **Verify subagent git claims** — hallucinated commits have occurred.

## System Architecture

```
Local (Lead orchestrator) ──push/pull──> GitHub <──pull── Server (GPU training)
     │                                        │
     ├─ dispatches subagents                   ├─ /data/repo (code)
     ├─ owns tree.json, journal,               ├─ /data/runs/<ID>/ (artifacts)
     │  memory bookings                        └─ collect back via scripts/collect_node.sh
     └─ websearch for grounding
```

## File Map

| Path | Purpose |
|---|---|
| `AGENTS.md` | Protocol: startup sequence, work loops, hard rules |
| `STATE.md` | Current snapshot: stage, facts, tasks, next steps |
| `docs/PROTOCOL.md` | File contracts: node dir structure, formulas, state machine |
| `docs/research_direction.md` | High-level direction memo |
| `docs/arXiv-2604.12999_HypoExplore_summary.txt` | The framework paper we implement |
| `docs/inspiration_from_GOD.txt` | User-provided direction hints (check at each Idea dispatch) |
| `code/engine/train.py` | Shared training engine — reads node model/config |
| `code/data/fsc147.py` | FSC147 VarV2 dataset loader |
| `code/selection/select_next.py` | Parent/hypothesis selection policy |
| `scripts/run_node.sh` | Launch training on server in tmux |
| `scripts/collect_node.sh` | Pull results from server to local |
| `memory/hypotheses.jsonl` | Append-only hypothesis bank |
| `memory/index.json` | Rebuildable confidence snapshot |
| `tree/tree.json` | Trajectory tree with lineage |
| `tree/nodes/<ID>/` | Self-contained node: idea → code → result → feedback → synthesis |
| `tasks/T####_*.md` | Task cards; rename to claim |

## Quick Start

```bash
git pull --ff-only && cat AGENTS.md STATE.md
```
