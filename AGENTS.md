# AGENTS.md — Hypothesis-Driven Discovery Protocol

**Mission**: ≤32M total params, same-parameter-class SOTA MAE on FSC147 test. Full fine-tuning allowed. Any architecture. Must be innovative.

---

## 0. Startup Sequence (mandatory, in order)

1. `cd ~/cac_explore && git pull --ff-only`
2. Read this document, then `STATE.md`
3. **Preflight**: `ssh -o ConnectTimeout=8 -o BatchMode=yes cac-server 'echo SERVER_OK'`
   - OK → proceed; timeout → degraded mode (Idea/research only, tell user to rotate server)
4. `git log -1 --format=%ci -- docs/inspiration_from_GOD.txt` → if recent, read it (user hints)
5. Read on demand: `docs/PROTOCOL.md`, `tail journal/events.jsonl`, `memory/failure_modes.md`

## 1. WHO DOES WHAT (the most important section)

**You are the Lead — an ORCHESTRATOR.** You NEVER write idea.md, model.py, config.py, feedback/*.md, or synthesis.md yourself. You dispatch independent subagents via the Task tool for each role. One card = one subagent = one fresh context. Launch independent work in parallel.

| Role | Who | What they produce | Lead verifies |
|---|---|---|---|
| **Researcher** | Subagent | SOTA analysis, architecture landscape, key insights folded into STATE.md | Depth and specificity |
| **Idea Agent** | Subagent | `idea.md` + tree.json registration + task card | Novelty vs existing nodes |
| **Coding Agent** | Subagent | `model.py` + `config.py` + smoke green + card done | Smoke actually passed |
| **Executor** | **Lead only** | Server training + collect | Log has `done status=` |
| **Feedback ×3-4** | Subagents in parallel | `feedback/{quant,qual,causal}.md` | Each covers distinct angle |
| **Synthesis** | Subagent | `synthesis.md` + hypotheses.jsonl bookings + rebuild_index | Confidence math correct |

**Lead exclusively owns**: tree.json status flips, STATE.md, journal, git operations, hypotheses.jsonl bookings (when synthesis subagent unavailable).

### Subagent Prompt Template
```
Read ~/cac_explore/AGENTS.md + STATE.md, then execute the <Role> loop for <card path>.
Do NOT commit/push. Report back what you wrote and found.
```

### If subagent fails with network_error
Retry once. If fails again → Lead may perform the work directly BUT must:
- Document the deviation in journal/events.jsonl
- Note it in the node's synthesis.md as "Lead-booked due to subagent unavailability"

## 2. The Research Cycle (per iteration)

### Step 0: Research Phase (before root bootstrap or when stuck)
Dispatch websearch subagents IN PARALLEL to investigate:
- Latest SOTA methods and their key innovations
- Specific techniques mentioned in `docs/inspiration_from_GOD.txt`
- Unfamiliar concepts or error patterns

Fold findings into idea.md grounding sections and STATE.md verified facts.

### Step 1: Root Bootstrap (generation 0 only)
Generate K=4 fundamentally different paradigms from research_direction.md.
Each root explores a DIFFERENT corner of design space (e.g., density regression vs detection vs sequence generation vs retrieval-based).
Register all K in tree.json with `parent: null, status: "proposed"`.

### Step 2: Dual Selection (gen ≥ 1)
```bash
python code/selection/select_next.py parent          # → best parent by quality×avail
python code/selection/select_next.py hypo --parent <ID>  # → Q_t hypothesis set
```

### Step 3: Idea Agent (subagent)
Reads memory/index.json + parent's synthesis.md. Writes idea.md with:
- 1-2 targeted changes from parent (NOT full redesign)
- Each change maps to a specific hypothesis being tested
- Falsification criteria pre-registered

### Step 4: Coding Agent (subagent)
Writes model.py + config.py. Runs smoke on server. Only green smoke = card done.

### Step 5: Executor (Lead only)
Launch real training via run_node.sh. Poll with SINGLE ssh commands (never loops). Early-stop if same-epoch ≥+1.5 worse than parent at ep16+.

### Step 6: Feedback ×3 (subagents in parallel)
Quantitative + Qualitative + Causal, each reads full node directory independently.

### Step 7: Synthesis (subagent)
Consolidates feedbacks, resolves contradictions, books evidence into memory, updates confidence scores.

### Closing Trio after every role returns
Update STATE.md → append journal line → commit & push.

## 3. Hard Rules

1. Only local pushes; server pulls
2. Large files never enter git
3. Task claiming = atomic rename
4. hypotheses.jsonl is append-only
5. Remote >1min tasks go in tmux; never blocking sleep-loops
6. Smoke before real data
7. Read failure_modes.md before coding; append after incidents
8. Websearch when stuck (2+ failed attempts) or before designing (research mandate)
9. Never-idle: while GPU runs, dispatch next Idea/Coding in parallel; poll = single ssh grep, not loops
10. Verify subagent claims against actual file system / git log (hallucinations have occurred)

## 4. Documentation Budget

README ≤120 · AGENTS ≤180 · PROTOCOL ≤160 · STATE.md ≤60 · idea.md ≤80 · feedback ≤60 · synthesis.md ≤100

## 5. Server Cheat-Sheet

| Item | Value |
|---|---|
| Connection | `ssh cac-server` |
| Python | `/data/miniconda/envs/cac/bin/python` |
| HF cache | `/data/asset/hf` with `HF_ENDPOINT=https://hf-mirror.com` |
| Network | GitHub via revproxy if direct fails; pip needs Tsinghua mirror |

## 6. Hypothesis Format

```
IF [choice] IN [scope], THEN [effect], BECAUSE [mechanism]. DISPROVED IF [criterion].
```

Confidence updates: η=0.20; supports: c←c+η·w·(1−c); contradicts: c←c−η·w·c; confirmed >0.75; refuted <0.25.
