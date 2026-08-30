# AGENTS.md — Hypothesis-Driven Discovery

Implementation of [HypoExplore (arXiv:2604.12999)](https://arxiv.org/abs/2604.12999) adapted for FSC147 crowd counting. Deviations from the paper are explicit (§9), never silent.

**Mission**: ≤32M total params · same-parameter-class SOTA MAE on FSC147 test.
**Regime (since 2026-08-30, user directive)**: PARTIAL-FT backbone — middle layers (stages 1-2, hs_map 2/3) MAY be fine-tuned with differential LR (backbone 0.1× head). Innovation in pluggable head parts (§5.14) remains primary, but backbone mid-layer tuning is now in-scope to test intermediate-vs-final readout. Optimizer/loss/schedule are fixed (AdamW 1e-3 head / 1e-4 backbone, wd0.05, cosine, bs16, AMP, 30ep, MSE(+SmoothL1)). Unfreezing early stem (stage0) or all stages is out of scope. See README for champion and empirical lessons.

**Engine contract (frozen)**: `config.py` → `cfg=dict(...)`; `model.py` → `build_model(cfg)` → `forward(imgs,bboxes[,bboxes3])` → `{"density", optional "n_aux"}`. Engine loss = MSE(dens,gt_d)+w_cnt·L1(sum(dens),gt_c); **only `out["density"]`** feeds the loss/gradients. Asserts `params ≤ max_params_M`.

## 1. Operating Modes

Mode is set by the user at session start or mid-session. Switches are logged in the journal; the active mode lives in STATE.md's session block.

| Mode | Behavior |
|---|---|
| **Free-Research** (default) | Lead autonomously drives the cycle (§4) under all gates and hard rules |
| **User-Guided** | User directives override defaults, gates, cycle order; Lead executes then back-fills mandatory records (journal entry + tree/STATE honesty pass) |

History integrity never breaks in User-Guided mode: append-only ledgers stay append-only, large files stay out of git. A directive conflicting with this document wins within its scope.

## 2. Startup Sequence (mandatory, in order)

1. `cd ~/cac_explore && git pull --ff-only`
2. Read this document, then `STATE.md`. STATE.md holds exactly ONE session block; stale sections are archived to `journal/` first.
3. Preflight: `ssh -o ConnectTimeout=8 -o BatchMode=yes cac-server 'echo SERVER_OK'`
   - OK → proceed
   - Timeout → check `local/address_and_password.md` (mtime) for fresh creds; rerun `python3 scripts/install_key.py`; retry
   - Still down → degraded mode (idea/research/eval-lab only), ask user to rotate the server
   - `local/address_and_password.md` is the ALWAYS source of truth for host/port/password — never assume yesterday's address still works
4. Read on demand: `docs/PROTOCOL.md`, `tail journal/events.jsonl`, `memory/failure_modes.md`

## 3. Roles — the Lead Is an Orchestrator

The Lead NEVER personally writes idea.md, model.py, config.py, feedback/*.md, or synthesis.md. Every role is dispatched to an independent subagent via the Task tool: one card = one subagent = one fresh context. Independent work launches in parallel.

| Role | Who | Produces | Lead verifies |
|---|---|---|---|
| **Researcher** | Subagent | SOTA analysis, architecture landscape, facts folded into STATE.md | Depth and specificity |
| **Idea Agent** | Subagent | `idea.md` + `novelty.json` + task card | Novelty gate passed |
| **Coding Agent** | Subagent | `model.py` + `config.py` + green smoke + card done | Smoke passed + pluggability passed (§5.14) |
| **Executor** | **Lead only** | Server training + collect | Log contains `done status=` |
| **Feedback ×3+1** | Subagents in parallel | `feedback/{quant,qual,causal}.md` (+ `diagnostic.md` on failure) | Angles distinct |
| **Synthesis** | Subagent | `synthesis.md` + hypothesis bookings + calibration table | Quality gate + confidence math |

**Lead-exclusive ownership**: tree.json status flips · STATE.md · journal · git ops · hypotheses.jsonl bookings (when no synthesis subagent).

Subagent prompt template:

    Read ~/cac_explore/AGENTS.md + STATE.md, then execute the <Role> loop for <card path>.
    Do NOT commit/push. Report back what you wrote and found.

On network_error retry once. Second failure: Lead may do the work directly but MUST log the deviation in `journal/events.jsonl` and mark the node's synthesis.md "Lead-booked due to subagent unavailability".

## 4. Research Cycle (one pass per iteration)

### Step 0 · Research phase — before root bootstrap or when stuck
Dispatch websearch subagents IN PARALLEL on latest SOTA, unfamiliar concepts, error patterns. Fold findings into idea.md grounding and STATE.md verified facts.

### Step 1 · Root bootstrap — generation 0 only
Generate K=4 fundamentally different paradigms from `docs/research_direction.md`; each explores a DIFFERENT corner of design space. Register all in tree.json with `parent: null, status: "proposed"`.

### Step 2 · Dual selection — gen ≥ 1
    python code/selection/select_next.py parent              # best parent by quality×avail
    python code/selection/select_next.py hypo --parent <ID>  # Q_t hypothesis set

### Step 3 · Idea agent — mandatory multi-angle dispatch
Always parallel multi-angle, never a single agent:
- ≥1 pure-mathematics lens (first principles: point processes, decision theory, equivariance, identifiability)
- ≥1 pure-physics lens (measurement/inverse problems: particle counting, deconvolution, shot noise, super-resolution)
- ≥1 champion-lineage agent building incrementally on the best node — champion benefits NEVER dropped while exploring disruption
- Optional: counter-intuitive, low-cost/high-yield, training dynamics

Zero-base lenses get NO champion anchoring, NO refuted-list foreclosure, NO minimal-experiment bias; each returns ONE sharpest proposal with mechanism + kill-or-confirm ladder. The Lead integrates and picks.

The idea agent reads `memory/index.json` + parent synthesis.md and writes idea.md with 1–2 targeted changes from the parent (never a full redesign); each change maps to a hypothesis with pre-registered falsification criteria.

**Novelty gate (mandatory before tree registration):**
    python scripts/novelty_check.py --file <node>/idea.md   # stage 1: retrieval
Stage 2 is structural: a judge subagent compares design principles (not surface wording) against top matches and writes `novelty.json` {novel, most_similar_to, shared_principles, new_contribution}. A duplicate regenerates ONCE with avoid-instructions; second rejection kills the proposal.

### Step 4 · Coding agent
Writes model.py + config.py, then smoke-tests on the server. Only a green smoke completes the card. On smoke failure: diagnose, retry up to 2 times, else fail honestly (§9).

### Step 5 · Executor (Lead only)
Launch real training via `scripts/run_node.sh`. Poll with SINGLE ssh commands, never loops. Early-stop when at ep16+ and same-epoch train is ≥+1.5 worse than parent best.

### Step 6 · Feedback — always 3 agents, +1 on failure
Quantitative + Qualitative + Causal always run, each reading the full node directory independently. **Diagnostic** also runs whenever a node FAILED, timed out, or was early-stopped: root-causes the failure and appends notes to `memory/failure_modes.md`.
Lean path (clean early-stop meeting the pre-registered gate): minimum = Quantitative + Diagnostic; skipping even those requires a journal-documented deviation. Zero feedbacks is never acceptable.

### Step 7 · Synthesis
Consolidates feedbacks, deduplicates overlaps, resolves disagreements by specificity. Before booking, apply the **quality gate**:
1. Each new hypothesis passes `python scripts/check_hypothesis.py --text "..."` — malformed → revise or discard
2. Max 2 new hypotheses per node (K_synth=2); prefer updating existing over new
3. Contradiction remap: a proposal stating the opposite of an existing mechanism books as `contradicts` instead of a new hypothesis
4. Misattribution check: reasoning must match the hypothesis text, else remap or discard

After booking run `python scripts/calibration_report.py`; paste the bin table into synthesis.md.

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
8. Websearch when stuck (2+ failed attempts) or before designing
9. Never-idle: while the GPU runs, dispatch the next Idea/Coding in parallel; polling = single ssh grep
10. Verify subagent claims against the actual filesystem / git log (hallucinations have occurred)
11. **Target stability**: the mission target changes ONLY via `docs/research_direction.md` at session start with a journal entry — mid-session drift invalidates early-stop bars and evidence weights
12. **Gate order**: no tree registration without the novelty gate; no booking without check_hypothesis; no session close without a calibration table in the latest synthesis
13. **Docs-sync on ops changes**: environment/ops changes (tools, creds rotation, paths, server quirks) are documented IMMEDIATELY in the affected docs (STATE gotchas, `memory/failure_modes.md`, §7) and committed — never deferred to session close
14. **Pluggable-only architecture**: every component bolted onto the frozen backbone MUST be an independent, self-contained module. Components may serialize (串联) or parallelize (并联), but MUST NOT couple tightly to each other. No module passes its output into another module's internals; no gate depends on another module's state. Coupling is limited to (a) frozen-backbone features and (b) exemplar embeddings — the shared, stable interfaces. This guarantees single-switch ablations: toggling a component on/off must not require touching another. A design that cannot be cleanly ablated by removing one component is REJECTED before smoke.

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
| Runs | `/data/runs/<NODE>/` holds live training (best.pth, result.json, train.log); stale runs move to `/data/runs/archive_<date>/` — never delete the active lineage |

## 8. Hypothesis Format & Memory Bank

    IF [choice] IN [scope], THEN [effect], BECAUSE [mechanism]. DISPROVED IF [criterion].

Confidence: η=0.20 · support: c←c+η·w·(1−c) · contradict: c←c−η·w·c · confirmed >0.75 · refuted <0.25.
Under η=0.20 a single strong contradiction cannot cross 0.25 — STATE.md's operational refuted list governs retries; ledger confidences are advisory.

Tools: `scripts/check_hypothesis.py` (pre-booking gate) · `scripts/novelty_check.py` (pre-registration gate) · `scripts/calibration_report.py` (per-synthesis health check).

## 9. Recorded Deviations from arXiv:2604.12999

| Paper component | Local practice | Why |
|---|---|---|
| Coding error-recovery R_max=10 | Smoke-first + max 2 fix retries, else honest fail | tau_max budget + server queue make 10 retries unaffordable |
| Hyperparameter refinement F_max=5 | None — config authored once | Same budget reason; revisit if smoke-fail rate rises |
| Qualitative VLM heatmaps | Text-log qualitative when no dump exists | Eval-lab dumps only when needed |
| Embedding API redundancy filter | TF-IDF retrieval + LLM structural judge | No external embedding dependency |
