# AGENTS.md — Hypothesis-Driven Discovery Protocol

Implements [HypoExplore (arXiv:2604.12999)](https://arxiv.org/abs/2604.12999) adapted for FSC147 crowd counting. Known deviations from the paper are recorded in §7, not silently introduced.

**Mission**: ≤32M total params, same-parameter-class SOTA MAE on FSC147 test. Full fine-tuning allowed. Any architecture. Must be innovative.

---

## 0. Startup Sequence (mandatory, in order)

1. `cd ~/cac_explore && git pull --ff-only`
2. Read this document, then `STATE.md`. STATE.md holds exactly ONE session block; if it contains duplicated sections from a prior session, archive old content into `journal/` and fix before proceeding.
3. **Preflight**: `ssh -o ConnectTimeout=8 -o BatchMode=yes cac-server 'echo SERVER_OK'`
   - OK → proceed; timeout → check `local/address_and_password.md` (mtime) for fresh creds and re-onboard: `python3 scripts/install_key.py`, then retry
   - Still down → degraded mode (Idea/research/eval-lab only, tell user to rotate server)
   - ALWAYS treat `local/address_and_password.md` as the source of truth for host/port/password — never assume the old address works
4. Read on demand: `docs/PROTOCOL.md`, `tail journal/events.jsonl`, `memory/failure_modes.md`

## 1. WHO DOES WHAT (the most important section)

**You are the Lead — an ORCHESTRATOR.** You NEVER write idea.md, model.py, config.py, feedback/*.md, or synthesis.md yourself. You dispatch independent subagents via the Task tool for each role. One card = one subagent = one fresh context. Launch independent work in parallel.

| Role | Who | What they produce | Lead verifies |
|---|---|---|---|
| **Researcher** | Subagent | SOTA analysis, architecture landscape, key insights folded into STATE.md | Depth and specificity |
| **Idea Agent** | Subagent | `idea.md` + novelty.json + task card | Novelty gate passed |
| **Coding Agent** | Subagent | `model.py` + `config.py` + smoke green + card done | Smoke actually passed |
| **Executor** | **Lead only** | Server training + collect | Log has `done status=` |
| **Feedback ×3+1** | Subagents in parallel | `feedback/{quant,qual,causal}.md` (+`diagnostic.md` on failure) | Each covers distinct angle |
| **Synthesis** | Subagent | `synthesis.md` + hypothesis bookings + calibration table | Gate + confidence math correct |

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
Dispatch websearch subagents IN PARALLEL to investigate latest SOTA, unfamiliar concepts or error patterns. Fold findings into idea.md grounding sections and STATE.md verified facts.

### Step 1: Root Bootstrap (generation 0 only)
Generate K=4 fundamentally different paradigms from research_direction.md. Each root explores a DIFFERENT corner of design space. Register all K in tree.json with `parent: null, status: "proposed"`.

### Step 2: Dual Selection (gen ≥ 1)
```bash
python code/selection/select_next.py parent          # → best parent by quality×avail
python code/selection/select_next.py hypo --parent <ID>  # → Q_t hypothesis set
```

### Step 3: Idea Agent (subagent) — MANDATORY MULTI-ANGLE DISPATCH
Idea generation MUST be parallel multi-angle, never a single agent:
- **≥1 pure-mathematics lens** (first-principles: point processes, decision theory, equivariance, identifiability)
- **≥1 pure-physics lens** (measurement/inverse-problem theory: particle counting, deconvolution, shot noise, super-resolution)
- **≥1 champion-lineage agent** that builds incrementally on the current best node — champion benefits are NEVER dropped while exploring disruption
- Optional extra lenses: counter-intuitive, low-cost/high-yield details, training dynamics

Zero-base lenses get NO champion anchoring, NO refuted-list foreclosure, NO minimal-experiment bias; each returns ONE sharpest proposal with mechanism + kill-or-confirm ladder. The Lead integrates and picks.

The Idea agent then reads memory/index.json + parent's synthesis.md. Writes idea.md with:
- 1-2 targeted changes from parent (NOT full redesign); each change maps to a specific hypothesis being tested; falsification criteria pre-registered

**Novelty Gate (mandatory, before tree registration):**
```bash
python scripts/novelty_check.py --file <node>/idea.md   # stage-1 retrieval
```
Stage-2 is structural: a judge subagent compares design principles (not surface wording) against the top matches and writes `novelty.json` {novel: bool, most_similar_to, shared_principles, new_contribution}. Duplicate → regenerate ONCE with explicit avoid-instruction; second rejection kills the proposal.

### Step 4: Coding Agent (subagent)
Writes model.py + config.py. Runs smoke on server. Only green smoke = card done. On smoke failure: diagnose, retry up to 2 times, then fail the card honestly (see §7 waiver).

### Step 5: Executor (Lead only)
Launch real training via run_node.sh. Poll with SINGLE ssh commands (never loops). Early-stop if same-epoch ≥+1.5 worse than parent at ep16+.

### Step 6: Feedback — 3 agents always, +1 on failure (subagents in parallel)
- Quantitative + Qualitative + Causal always run, each reads the full node directory independently.
- **Diagnostic** additionally runs whenever the node FAILED, timed out, or was early-stopped: root-causes the failure and appends implementation_notes to `memory/failure_modes.md`.
- Lean path (early-stop with pre-registered gate met cleanly): minimum = Quantitative + Diagnostic; skipping even those requires a journal-documented deviation. Never run zero feedbacks.

### Step 7: Synthesis (subagent)
Consolidates feedbacks, deduplicates overlapping updates, resolves disagreements by specificity. Then applies the **Quality Gate** before any booking:
1. Every new hypothesis passes `python scripts/check_hypothesis.py --text "..."` (format: IF choice IN scope THEN effect BECAUSE mechanism DISPROVED IF measurable-criterion). Malformed → revise or discard.
2. Max **2 new hypotheses per node** (K_synth=2); prefer updating existing ones over creating new.
3. Contradiction remap: if a proposed hypothesis states the opposite of an existing mechanism, book it as `contradicts` evidence instead of a new hypothesis.
4. Misattribution check: evidence reasoning must match the hypothesis text; otherwise remap to the right hyp_id or discard.

After booking, run `python scripts/calibration_report.py` and paste the bin table into synthesis.md.

### Closing Trio after every role returns
Update STATE.md (REWRITE the single session block; never append duplicate sections) → append journal line (real UTC+8 timestamp; refs must be existing paths) → commit & push.

## 3. Hard Rules

1. Only local pushes; server pulls
2. Large files never enter git
3. Task claiming = atomic rename
4. hypotheses.jsonl is append-only (corrections = new events, never edits)
5. Remote >1min tasks go in tmux; never blocking sleep-loops
6. Smoke before real data
7. Read failure_modes.md before coding; append after incidents
8. Websearch when stuck (2+ failed attempts) or before designing (research mandate)
9. Never-idle: while GPU runs, dispatch next Idea/Coding in parallel; poll = single ssh grep, not loops
10. Verify subagent claims against actual file system / git log (hallucinations have occurred)
11. **Target stability**: the mission metric/target changes ONLY via editing `docs/research_direction.md` at session start with a journal entry. Never drift mid-session or mid-experiment — target drift invalidates early-stop bars and evidence weights.
12. **Gate order**: no tree registration without novelty gate; no booking without check_hypothesis pass; no session close without calibration table in the latest synthesis.
13. **Docs-sync on ops changes**: any environment/ops change (tool installed/removed, creds rotation, path/env change, server quirk discovered) is documented IMMEDIATELY in the affected docs (STATE/DISTILLED gotchas, `memory/failure_modes.md`, §5 cheat-sheet) and committed before proceeding — never deferred to session close.

## 4. Documentation Budget

README ≤120 · AGENTS ≤180 · PROTOCOL ≤160 · STATE.md ≤60 · idea.md ≤80 · feedback ≤60 · synthesis.md ≤100

## 5. Server Cheat-Sheet

| Item | Value |
|---|---|
| Creds (source of truth) | `local/address_and_password.md` — check mtime EVERY session; after any change run `python3 scripts/install_key.py` once |
| Connection | `ssh cac-server` (alias rewritten by install_key.py on rotation) |
| Python | `/data/miniconda/envs/cac/bin/python` |
| HF cache | `/data/asset/hf` with `HF_ENDPOINT=https://hf-mirror.com` |
| Network | GitHub via revproxy if direct fails; pip needs Tsinghua mirror |

## 6. Hypothesis Format & Memory Tools

```
IF [choice] IN [scope], THEN [effect], BECAUSE [mechanism]. DISPROVED IF [criterion].
```

Confidence updates: η=0.20; supports: c←c+η·w·(1−c); contradicts: c←c−η·w·c; confirmed >0.75; refuted <0.25.
Note: under η=0.20 a single strong contradiction cannot cross 0.25 — STATE.md's operational "refuted" list governs retries; ledger confidences are advisory (calibration_report prints the lag).

Tools: `scripts/check_hypothesis.py` (pre-booking gate) · `scripts/novelty_check.py` (pre-registration gate) · `scripts/calibration_report.py` (per-synthesis health check).

## 7. Recorded Deviations from arXiv:2604.12999

| Paper component | Local practice | Why |
|---|---|---|
| Coding error-recovery R_max=10 | smoke-first + max 2 fix retries, else honest fail | tau_max budget + server queue too costly for 10 retries |
| Hyperparameter refinement F_max=5 | none (config authored once) | same budget reason; revisit if smoke-fail rate rises |
| Qualitative VLM heatmaps | text-log qualitative when no dump available | eval-lab dumps only when needed |
| Embedding API redundancy filter | TF-IDF retrieval + LLM structural judge | no external embedding dependency |
