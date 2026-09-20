# AGENTS.md — Active Hypothesis Exploration for Crowd Counting

Implementation of **HypoExplore — Agentic Discovery with Active Hypothesis
Exploration** (Koo et al., [arXiv:2604.12999](https://arxiv.org/abs/2604.12999))
adapted for FSC147 class-agnostic counting. The paper's evolutionary machinery is the
load-bearing contract of this repo: **quality × availability parent selection
(Eq.2–4), dual Thompson-exploit + epistemic-explore hypothesis selection
(Eq.5–6), evidential confidence updating (Eq.1), and confirmation/refutation
thresholds**. We deviate from the paper ONLY where §9 of this file explicitly
records a deviation. Deviations that silently weaken the machinery are a hard
violation.

## Mission
**≤32M total params · same-parameter-class SOTA MAE on FSC147 test.**

## Standing regime (user directive — NOT negotiable)
- Backbone **frozen**, head-only training. Unfreezing is proven net-harmful
  (H0005 refuted, N0066/67) and out of scope.
- Innovation lives in the **pluggable head**: components bolt onto the frozen
  intermediate features hs(2,3) + the exemplar embedding ONLY — those are the
  single stable interface. A component that cannot be cleanly ablated by
  removing one switch (`use_<name>`) is **rejected before smoke**.
- Optimizer/loss/schedule invariant: AdamW, MSE(+w_cnt·L1), cosine, AMP.
- Every model must satisfy `build_model(cfg)` →
  `forward(imgs, bboxes[, bboxes3]) → {"density", ...}`; only `out["density"]`
  feeds the loss.

## The mechanism, and who may move it
The discovery mathematics lives in one place: `src/cac/expt/`. It is touched
only through the CLI:

| Gate | CLI | Guarded against |
|---|---|---|
| Parent selection (Eq.2–4) | `python scripts/discovery.py parent` | hand-picking a parent |
| Hypothesis selection (Eq.5–6) | `python scripts/discovery.py hypo <parent>` | shaping Q_t to pre-commit |
| Hypothesis authoring/format | `discovery validate` + `novelty_check.py` | malformed, unmeasured claims |
| Confidence (Eq.1) | `discovery evidence <hyp_id> --type <t> --strength <w> [--node n] [--note s]` — **never hand-computed**, append-only | gaming confidence by math |
| Tree state | `discovery` / `run_node` write info.json | silent info.json edits |
| Calibration | `discovery calibration` | confidence at test drift |
| Commit gate | `scripts/conformance.py` (must exit 0) | drift, phantom ids, lineage rot |

**No agent writes confidence numbers.** A confidence value is the *derived*
result of ledger evidence; neither the Lead nor any subagent may edit
`memory/hypotheses.jsonl` except by appending events through approved
commands. Editing ledger history, editing `info.json` by hand for status,
creating a child node dir without `discovery hypo`, or skipping conformance is
**gaming the machinery and is forbidden**.

## 1. Operating Modes
| Mode | Behavior |
|---|---|
| **Free-Research** (default) | Lead autonomously drives the cycle under all gates |
| **User-Guided** | User directives override defaults; Lead executes then back-fills journal + state |

History integrity never breaks: ledgers stay append-only, large files stay out
of git, conformance stays green before every commit.

## 2. Startup Sequence (mandatory, in order)
1. `git pull --ff-only`
2. Read this file, then `STATE.md` (exactly one session block).
3. Preflight `ssh -o ConnectTimeout=8 -o BatchMode=yes cac-server 'echo OK'`.
   On timeout → read `local/address_and_password.md` (mtime!), run
   `python scripts/install_key.py`, retry. (`local/` is gitignored; it is the
   always source of truth for host/port creds.)
4. Read `journal/events.jsonl` tail and `tree/` via `discovery tree`.
5. Run `scripts/conformance.py` — any prior session that left it red must be
   fixed before new work.

## 3. Roles — the Lead is an orchestrator, never a writer
The Lead NEVER personally authors `idea.md`, node `model.py`/`config.toml`,
`feedback/*.md`, or `synthesis.md`. Every role is dispatched to its own
subagent via the Task tool: **one card = one subagent = one fresh context =
one GPU card**. Independent work launches in parallel; the Lead verifies every
claim against the actual filesystem/git log (hallucinations have occurred).

| Role | Who | Produces | Lead verifies |
|---|---|---|---|
| **Researcher** | Subagent(s) | SOTA/mechanism facts → STATE.md | Depth, specific refs |
| **Idea Agent** | Subagent | `idea.md` + novelty `novelty.json` | Novelty gate green |
| **Coding Agent** | Subagent | `model.py` + `config.toml` (delta from parent) + green smoke | Smoke + pluggable rule |
| **Executor** | Lead + own card | `run_node` on server; collect result | result.json exists, honest |
| **Feedback ×3+1** | Subagents parallel | `feedback/{quant,qual,causal}.md` (+`diagnostic.md` on failure) | Distinct angles |
| **Synthesis** | Subagent | `synthesis.md` + K_SYNTH≤2 bookings | Confidence math + format gate |

Subagent prompt template:
    Read AGENTS.md + STATE.md, then execute the <Role> loop for <card>.
    Do NOT commit/push. Report exactly what files you wrote and with what contents.

No role merges: one subagent never wears two hats in one dispatch. If a
subagent's own claim references a file it wrote, the Lead re-reads that file
before accepting it.

## 4. Discovery Cycle (one pass per iteration)
1. **Select parent** `discovery parent` — deterministic max of score.
2. **Select hypotheses** `discovery hypo <parent>` — Q_t (≤2·K_HYPO) over
   uncertain hypotheses untested on this ancestry; child dir is created nested
   under the parent (the filesystem lineage IS the trajectory tree).
3. **Idea Agent** — MUST survey the web first: latest adjacent mechanisms
   (papers WITH code repos preferred), then think unconstrained from first
   principles to the essence of the problem — novelty-gate originality is the
   floor, not the ceiling; closed-door derivation alone is never acceptable.
   Always multi-angle (≥ pure-mathematics lens, ≥ champion-lineage lens, ≥ one
   counter-intuitive/low-cost lens). Each proposal is 1–2 targeted changes from
   the parent, each change maps to **exactly one testable hypothesis** with a
   pre-registered falsification criterion.
4. **Novelty gate** — `python scripts/novelty_check.py "<text>" [hyp_id]` must
   exit 0 (top TF-IDF sim < 0.82 AND no structural twin). Duplicate → regenerate
   once; second failure kills the proposal. (Local stand-in for the paper's
   embedding-API redundancy filter — §9.)
5. **Coding Agent** — delta from parent config/model only. Smoke first. Up to
   **3 fix retries** (R_max). After that, fail honestly (§9).
6. **Run** — `python scripts/run_node.py <node> [--budget-seconds 1800]`.
   Hard wall-clock ceiling τ_max = **1800 s**; on exceed, `_BudgetStop` halts
   and node is marked `timeout` (not `done`).
7. **Feedback** — Quantitative + Qualitative + Causal (3 agents); +Diagnostic on
   failure/timeout. Lean path allowed only for a clean early-stop that meets the
   pre-registered gate; zero feedbacks is never acceptable.
8. **Synthesis** — consolidate, dedupe, apply quality gate: (a) each booking
   passes format gate, (b) ≤ K_SYNTH=2 new hypotheses per node, (c) an
   opposite-of-existing books as `contradicts` on the existing id (never a new
   duplicate), (d) misattributed reasoning → remap or discard. Then run
   `discovery calibration` and paste the bin table into synthesis.md.
9. **Lab closure** — STATE.md session block → journal entry → `conformance` green
   → commit & push (server pulls; local pushes only).

## 5. Hard Rules (anti-laziness, anti-gaming)
1. Observed results are written as observed. Outcome shopping
   (re-running until a hypothesis fits) is **fabrication**.
2. `memory/hypotheses.jsonl` is append-only; corrections are new events. Never
   edit a line, never reorder `ts`, never delete an event.
3. No tree lines get status flips except via `run_node` (done/failed/timeout)
   or explicit discovery commands. Never hand-write a second node's `info.json`.
4. `conformance.py` must exit 0 **before every commit**. A red conformance is
   the top block.
5. Numbers you put into journal/synthesis/feedback must be verifiable from a
   `result.json`/log you or a subagent actually read — no recollections.
6. Never idle while the GPU runs: dispatch the next Idea/Coding in parallel;
   polling = a single ssh grep, never a sleep-loop.
7. A subagent `report` without file evidence is an unverified claim; verify or
   reject.
8. Doc drift is forbidden: environment/ops/creds changes land in the affected
   docs same-session (§7 cheat-sheet, STATE gotchas) — never deferred.
9. Anything longer than 1 min on the server runs in tmux; no blocking loops.
10. **No silent paper edits**: if you believe the paper's dynamics need a
    different constant or order, propose it to the user with evidence; do not
    just reprogram the machinery and continue (§9 only).
11. Refuted hypotheses are not retried silently. If a proposal re-encodes a
    refuted mechanism, it must book as `contradicts`-evidence on the existing
    id — and it must have a NEW falsifier, else it dies.
12. Every node run records `config_sha256` + `model_sha256` in `result.json`
    (runner does this) so "same config" claims are checkable.
13. A pluggable component is constructed AFTER all parent modules (append-only
    RNG order): shared-module init draws stay identical to the parent's, so a
    child differs from its parent only by the new component. The runner pins the
    data stream to `config.seed` (FSC147DataModule augment/seed wiring) and the
    model has zero dropout, so nothing else consumes RNG during training.
14. Journal events are appended ONLY via `scripts/journal.py` (single-line JSON
    + trailing newline enforced). Hand-appends glue objects onto one line and
    break conformance.
15. No closed-door ideas: every hypothesis must be preceded by a web survey of
    current adjacent mechanisms (code repos preferred). The survey informs;
    first-principles reasoning to the essence decides — "SOTA did it" is still
    not a mechanism (§6).

## 6. Hypothesis format & fidelity
```
IF [choice] IN [scope], THEN [measured effect], BECAUSE [mechanism ≥ sensible].
DISPROVED IF [falsification criterion with a number/comparison].
```
- Markers `IF IN THEN BECAUSE DISPROVED` must appear in order (machine-checked).
- Mechanism must identify a *reason the model improves*, not just "SOTA did it".
- Hedging ("maybe", "might") fails the gate. The falsifier must be a specific,
  testable bar (e.g. `final val MAE is not at least 0.30 lower ...`).
- Confidence: η=0.20 · support `c←c+η·w·(1−c)` · contradict `c←c−η·w·c` ·
  neutral logs but does NOT move c. Confirmed >0.75, refuted <0.25.
  These live ONLY in `src/cac/expt/`; agents never recompute them by hand.
- Falsifier bars are **semantics, not literals**: "≥0.30 lower" binds to the
  *live parent number under the current protocol*. If the protocol version
  changes between booking and verdict, re-instantiate the numeric line from the
  live parent and state BOTH numbers in the evidence note. Never edit the
  hypothesis text.
- Verdicts are **same-seed paired contrasts** (precision ~0.02 under the
  canonical harness): never compare absolute levels across seeds at the 0.30
  bar — level noise is ~±1 MAE.
- **Seed is FIXED at 20260830 for every run, forever** (user directive
  2026-09-20): all bookings, replicates and diagnostics run at this seed;
  `repro_run --set seed=` is FORBIDDEN without explicit user approval. No new
  seeds, no spread quantification, no cross-seed checks, ever.

## 7. Server cheat-sheet
| Item | Value |
|---|---|
| Connection | `ssh cac-server` (alias maintained by install_key.py) |
| Python | `/data/miniconda/envs/cac/bin/python` |
| HF cache | `/data/asset/hf`, offline-first (`hub.setup_hf_env()`: mirror endpoint, `HF_HUB_OFFLINE=1`; HF_* before heavy imports) |
| Data | `/data/dataset/FSC147` (VarV2 protocol: images_384_VarV2, gt_density_map_adaptive_384_VarV2, annotation_FSC147_384.json, Train_Test_Val_FSC_147.json) |
| Runs | node-local `run/latest/` on the server copy (best.pth, result.json, tensorboard) |
| GPU | single RTX3060 12 GB — one card per node run |
| Protocol (canonical) | augment=false, EMA eval, seeded loaders (seed=20260830 FIXED, never overridden) , cudnn deterministic · 32ep/1800s · every comparison same-seed vs the canonical baseline — pre-2026-09-19-evening runs are historical record only |
| GPU queue | one training tmux chain + one `has-session` watchdog for the next batch; when aborting, kill the WATCHDOG session first, then the chain (else the watchdog fires immediately); never `pkill -f` with a pattern that also appears in the killer's own command line |

## 8. Directory map
```
src/cac/expt/       the paper's machinery, single source (constants, node, hypothesis,
                    select, gates) — change only via §mechanism CLI & §10
src/cac/engine/     runner (smoke→budget, checksums)   src/cac/models/  champion + pl_module
src/cac/data/       FSC147 VarV2 datamodule            src/cac/calls/   best.pth, EMA
scripts/            discovery, conformance, novelty_check, run_node, install_key
tree/               THE trajectory tree (filesystem lineage = parent/child)
  N0001_champion/   seed root: info.json, idea.md, model.py, config.toml, result.json
memory/             hypotheses.jsonl (append-only) + index.json (rebuilt, not edited)
journal/            events.jsonl
docs/               research_direction.md (mission changes ONLY here, journaled)
configs/            read-only reference configs
```

## 9. Recorded deviations from arXiv:2604.12999
| Paper | Local | Why |
|---|---|---|
| Hand-written training loop | Lightning (smoke→budget) | same dynamics, less agent-written engine drift |
| Redundancy: embedding API + K_synth rejection | local TF-IDF (scripts/novelty_check.py) + structural judge | no external embedding dependency |
| Qualitative VLM heatmap reads | text-log qualitative analysis per node | no stored feature dumps at runtime |
| R_max=10 coding retries | ≤3 fix retries, then honest fail | τ_max budget + single-GPU queue |
| F_max=5 hyper-param refinements | phase 2; v1 authors config once | cost discipline |
| Reference model provided | N0001_champion (our proven artifact) | exploration starts from strength |
| Bootstrap K=5 random roots | K=1 certified seed root | see §9 "reference" above |
| Q_t composition on every test | when a NEW hypothesis is explicitly booked (`hypo --new/--book <parent>`), Q_t adjoins are capped at ONE compatible hypothesis and any co-composed pair sharing a head component (or config switch) is rejected at booking time | structurally infeasible / un-attributable joint compositions (N0006/N0007 case); selection scoring (Eq.5-6) untouched — `src/cac/expt/mechanisms.py` |

Everything else (selection, dual hypothesis selection, Eq.1 confidence,
confirmation/refutation thresholds, calibration, honesty gates, append-only
memory) is implemented as the paper specifies.

## 10. Changing the machinery itself
To change MATHLE constants, selection order, thresholds, or evidence types:
1. Edit ONLY `src/cac/expt/constants.py` + the corresponding module, with a
   journal entry citing the paper section being overridden.
2. Re-run `scripts/conformance.py` + `python -m tests`.
3. Commit separately as a mechanism change; never bundle with a node's run.

## 11. Don't-repeat register (settled mechanisms; ledger refs pruned 2026-09-20 reset)
- never: DDCA, extra spatial summaries/RGA, final-layer readout, backbone unfreeze.
- never: pre-condenser exemplar gating in ANY granularity (image-global channel
  gate, frozen-input control, per-exemplar scalar gate incl. clean solo,
  per-exemplar channel gate). Live directions were query/similarity-side and
  decoder-side — unproven, re-book fresh if wanted.
- frozen hs(2,3) readout + cross-attn condenser = load-bearing; don't quietly break them.