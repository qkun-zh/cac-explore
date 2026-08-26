# AGENTS.md — Hypothesis-Driven Discovery Protocol

Implementation of [HypoExplore (arXiv:2604.12999)](https://arxiv.org/abs/2604.12999) adapted for FSC147 crowd counting. Deviations from the paper are explicit (§9), never silent.

**Mission**: ≤32M total params · same-parameter-class SOTA MAE on FSC147 test · full fine-tuning allowed · any architecture · must be innovative.

## 1. Operating Modes

The user sets the mode at session start or at any point mid-session. Switches are logged in the journal; the active mode is recorded in STATE.md's session block.

| Mode | Behavior |
|---|---|
| **Free-Research** (default) | The Lead autonomously drives the research cycle (§4) under all gates and hard rules |
| **User-Guided** | User directives override protocol defaults, gates, and cycle order. The Lead executes them directly, then back-fills mandatory records (journal entry + tree/STATE honesty pass) |

Even User-Guided mode never breaks history integrity: append-only ledgers stay append-only, large files stay out of git. Where a directive conflicts with this document, the directive wins within its scope.

## 2. Startup Sequence (mandatory, in order)

1. `cd ~/cac_explore && git pull --ff-only`
2. Read this document, then `STATE.md`. STATE.md holds exactly ONE session block; duplicated sections from a prior session are archived into `journal/` before anything else happens.
3. **Preflight**: `ssh -o ConnectTimeout=8 -o BatchMode=yes cac-server 'echo SERVER_OK'`
   - OK → proceed
   - Timeout → check `local/address_and_password.md` (mtime) for fresh creds, re-onboard with `python3 scripts/install_key.py`, retry
   - Still down → degraded mode (idea / research / eval-lab only) and ask the user to rotate the server
   - `local/address_and_password.md` is ALWAYS the source of truth for host/port/password — never assume yesterday's address still works
4. Read on demand: `docs/PROTOCOL.md`, `tail journal/events.jsonl`, `memory/failure_modes.md`

## 3. Roles — the Lead Is an Orchestrator

The Lead NEVER personally writes idea.md, model.py, config.py, feedback/*.md, or synthesis.md. Every role is dispatched to an independent subagent via the Task tool: one card = one subagent = one fresh context. Independent work launches in parallel.

| Role | Who | Produces | Lead verifies |
|---|---|---|---|
| **Researcher** | Subagent | SOTA analysis, architecture landscape, verified facts folded into STATE.md | Depth and specificity |
| **Idea Agent** | Subagent | `idea.md` + `novelty.json` + task card | Novelty gate passed |
| **Coding Agent** | Subagent | `model.py` + `config.py` + green smoke + card done | Smoke actually passed |
| **Executor** | **Lead only** | Server training + collect | Log contains `done status=` |
| **Feedback ×3+1** | Subagents in parallel | `feedback/{quant,qual,causal}.md` (+ `diagnostic.md` on failure) | Angles are distinct |
| **Synthesis** | Subagent | `synthesis.md` + hypothesis bookings + calibration table | Quality gate + confidence math correct |

**Lead-exclusive ownership**: tree.json status flips · STATE.md · journal · git operations · hypotheses.jsonl bookings (when no synthesis subagent is available).

Subagent prompt template:

    Read ~/cac_explore/AGENTS.md + STATE.md, then execute the <Role> loop for <card path>.
    Do NOT commit/push. Report back what you wrote and found.

If a subagent fails with network_error: retry once. On a second failure the Lead may perform the work directly, but MUST record the deviation in `journal/events.jsonl` and mark the node's synthesis.md with "Lead-booked due to subagent unavailability".

## 4. The Research Cycle (one pass per iteration)

### Step 0 · Research phase — before root bootstrap or when stuck
Dispatch websearch subagents IN PARALLEL on latest SOTA, unfamiliar concepts, error patterns. Fold findings into idea.md grounding sections and STATE.md verified facts.

### Step 1 · Root bootstrap — generation 0 only
Generate K=4 fundamentally different paradigms from `docs/research_direction.md`; each root explores a DIFFERENT corner of design space. Register all K in tree.json with `parent: null, status: "proposed"`.

### Step 2 · Dual selection — gen ≥ 1
    python code/selection/select_next.py parent              # best parent by quality×avail
    python code/selection/select_next.py hypo --parent <ID>  # Q_t hypothesis set

### Step 3 · Idea agent — mandatory multi-angle dispatch
Idea generation is always parallel multi-angle, never a single agent:
- ≥1 pure-mathematics lens (first principles: point processes, decision theory, equivariance, identifiability)
- ≥1 pure-physics lens (measurement / inverse problems: particle counting, deconvolution, shot noise, super-resolution)
- ≥1 champion-lineage agent building incrementally on the current best node — champion benefits are NEVER dropped while exploring disruption
- Optional lenses: counter-intuitive, low-cost/high-yield details, training dynamics

Zero-base lenses get NO champion anchoring, NO refuted-list foreclosure, NO minimal-experiment bias; each returns ONE sharpest proposal with mechanism + kill-or-confirm ladder. The Lead integrates and picks.

The idea agent then reads `memory/index.json` + parent synthesis.md and writes idea.md with 1–2 targeted changes from the parent (never a full redesign); each change maps to a specific hypothesis with pre-registered falsification criteria.

**Novelty gate (mandatory before tree registration):**
    python scripts/novelty_check.py --file <node>/idea.md   # stage 1: retrieval
Stage 2 is structural: a judge subagent compares design principles (not surface wording) against top matches and writes `novelty.json` {novel, most_similar_to, shared_principles, new_contribution}. A duplicate regenerates ONCE with explicit avoid-instructions; a second rejection kills the proposal.

### Step 4 · Coding agent
Writes model.py + config.py, then smoke-tests on the server. Only a green smoke completes the card. On smoke failure: diagnose, retry up to 2 times, else fail the card honestly (see §9).

### Step 5 · Executor (Lead only)
Launch real training via `scripts/run_node.sh`. Poll with SINGLE ssh commands, never loops. Early-stop if same-epoch is ≥+1.5 worse than parent at ep16+.

### Step 6 · Feedback — always 3 agents, +1 on failure
Quantitative + Qualitative + Causal always run, each reading the full node directory independently. **Diagnostic** additionally runs whenever a node FAILED, timed out, or was early-stopped: root-causes the failure and appends implementation notes to `memory/failure_modes.md`.
Lean path (clean early-stop meeting the pre-registered gate): minimum = Quantitative + Diagnostic; skipping even those requires a journal-documented deviation. Zero feedbacks is never acceptable.

### Step 7 · Synthesis
Consolidates feedbacks, deduplicates overlapping updates, resolves disagreements by specificity. Before booking anything, apply the **quality gate**:
1. Every new hypothesis passes `python scripts/check_hypothesis.py --text "..."` — malformed → revise or discard
2. Max 2 new hypotheses per node (K_synth=2); prefer updating existing ones over creating new
3. Contradiction remap: a proposal stating the opposite of an existing mechanism books as `contradicts` evidence instead of a new hypothesis
4. Misattribution check: evidence reasoning must match the hypothesis text, else remap to the right hyp_id or discard

After booking run `python scripts/calibration_report.py` and paste the bin table into synthesis.md.

### Closing trio — after every role returns
Update STATE.md (REWRITE the single session block, never append duplicates) → append a journal line (real UTC+8 timestamp; refs must be existing paths) → commit & push.

## 5. Hard Rules

1. Only local pushes; the server pulls
2. Large files never enter git
3. Task claiming = atomic rename
4. hypotheses.jsonl is append-only — corrections are new events, never edits
5. Remote tasks >1min go in tmux; never blocking sleep-loops
6. Smoke before real data
7. Read failure_modes.md before coding; append after incidents
8. Websearch when stuck (2+ failed attempts) or before designing (research mandate)
9. Never-idle: while the GPU runs, dispatch next Idea/Coding in parallel; polling = single ssh grep
10. Verify subagent claims against the actual filesystem / git log (hallucinations have occurred)
11. **Target stability**: the mission target changes ONLY via editing `docs/research_direction.md` at session start with a journal entry — mid-session drift invalidates early-stop bars and evidence weights
12. **Gate order**: no tree registration without the novelty gate; no booking without check_hypothesis; no session close without a calibration table in the latest synthesis
13. **Docs-sync on ops changes**: environment/ops changes (tools installed/removed, creds rotation, path/env changes, server quirks) are documented IMMEDIATELY in the affected docs (STATE gotchas, `memory/failure_modes.md`, cheat-sheet §7) and committed before proceeding — never deferred to session close

## 6. Documentation Budget

README ≤120 · AGENTS ≤180 · PROTOCOL ≤160 · STATE.md ≤60 · idea.md ≤80 · feedback ≤60 · synthesis.md ≤100

## 7. Server Cheat-Sheet

| Item | Value |
|---|---|
| Creds (source of truth) | `local/address_and_password.md` — check mtime EVERY session; after rotation run `python3 scripts/install_key.py` once |
| Connection | `ssh cac-server` (alias rewritten by install_key.py on rotation) |
| Python | `/data/miniconda/envs/cac/bin/python` |
| HF cache | `/data/asset/hf` with `HF_ENDPOINT=https://hf-mirror.com` |
| Network | GitHub via revproxy if direct fails; pip needs Tsinghua mirror |

## 8. Hypothesis Format & Memory Bank

    IF [choice] IN [scope], THEN [effect], BECAUSE [mechanism]. DISPROVED IF [criterion].

Confidence updates: η=0.20 · support: c←c+η·w·(1−c) · contradict: c←c−η·w·c · confirmed >0.75 · refuted <0.25.
Under η=0.20 a single strong contradiction cannot cross 0.25 — STATE.md's operational refuted list governs retries; ledger confidences are advisory (calibration_report prints the lag).

Tools: `scripts/check_hypothesis.py` (pre-booking gate) · `scripts/novelty_check.py` (pre-registration gate) · `scripts/calibration_report.py` (per-synthesis health check).

## 9. Recorded Deviations from arXiv:2604.12999

| Paper component | Local practice | Why |
|---|---|---|
| Coding error-recovery R_max=10 | Smoke-first + max 2 fix retries, else honest fail | tau_max budget + server queue make 10 retries unaffordable |
| Hyperparameter refinement F_max=5 | None — config authored once | Same budget reason; revisit if smoke-fail rate rises |
| Qualitative VLM heatmaps | Text-log qualitative when no dump exists | Eval-lab dumps only when needed |
| Embedding API redundancy filter | TF-IDF retrieval + LLM structural judge | No external embedding dependency |
