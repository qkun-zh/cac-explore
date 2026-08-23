# AGENTS.md — Multi-Agent Collaboration Protocol

**Mission**: a lightweight, innovative CAC model with ≤32M total parameters (frozen backbone included) and MAE < 16 on FSC147 test, discovered through HypoExplore-style hypothesis exploration.

**Hard constraint**: every candidate must contain a frozen backbone with HF/timm pretrained weights; only the counting head (+ adapters) trains. The engine already optimizes `requires_grad` params exclusively.

---

## 0. Startup Sequence (mandatory, in this order)

1. `cd ~/cac_explore && git pull --ff-only` (repo lives at `~/cac_explore`; if absent, clone from GitHub)
2. Read this document, then `STATE.md`
3. **Preflight — BEFORE any role work**: `ssh -o ConnectTimeout=8 -o BatchMode=yes cac-server 'echo SERVER_OK'`
   - prints SERVER_OK → normal operation below
   - times out → instance reclaimed. Enter **degraded mode** immediately and tell the user what you need (rent instance → paste creds into `local/address_and_password.md` → `bash scripts/onboard.sh`). Do NOT burn effort on server-dependent steps first — check STATE.md blockers too
4. Read on demand only: `docs/PROTOCOL.md`, tail of `journal/events.jsonl`, `memory/failure_modes.md`
5. `git pull` may bring updates to `docs/inspiration_from_GOD.txt` — read it at startup if changed (check: `git log -1 --format=%ci -- docs/inspiration_from_GOD.txt`)

**Who are you?** You are the **Lead** — an orchestrator, not a solo worker (user mandate, no exceptions): roles Idea / Coding / Feedback×4 / Synthesis MUST each run as an INDEPENDENT subagent via the Task tool — one claimed card = one fresh context; launch independent cards in parallel (e.g. Feedback×4 in one batch). Self-playing any of these roles is FORBIDDEN — the code author reviewing their own work defeats the redundancy check. Exception: **Executor stays Lead-only** — server training, tmux watching, collecting are stateful sequential work you do yourself. Division of labor: subagents read files, write ONLY their own outputs (`idea.md`, `model.py`/`config.py`, `feedback/*.md`, `synthesis.md`, own-card renames) and RETURN a report; the Lead exclusively owns tree.json, STATE.md, journal, hypotheses.jsonl bookings, and ALL git operations. Each subagent prompt must say: "Read `~/cac_explore/AGENTS.md` + `STATE.md`, then execute the `<Role>` loop for `<card path>`; do NOT commit/push; report back what you wrote and found."

**Degraded mode (no live server)** — allowed: Idea hat (bootstrap, selection, planning), Coding hat up to draft commit & push, doc/state repairs, web research. Blocked: `--smoke`, real training, collect, feedback-on-results, synthesis evidence booking. Surface the blocker to the user at session START, then do the allowed parts while waiting.

## 1. Work Loops (find your current hat)

### Idea Agent (dispatch: subagent)
1. Read `memory/index.json` + parent's `synthesis.md`
2. Pick parent + hypotheses per PROTOCOL §4. If selection finds no expandable node (fresh tree), run **root bootstrap**: create K=4 root nodes directly from `docs/research_direction.md`, register them in `tree/tree.json` with `parent: null, status: "proposed"`
3. Write `tree/nodes/<ID>/idea.md` (fixed sections per PROTOCOL §2; falsifiable claims)
4. Register node in `tree.json` (`status: "proposed"`); do NOT write hypotheses.jsonl — Synthesis books events after quality gate
5. Create task card `tasks/T####_pending_coding_<ID>.md`

### Coding Agent (dispatch: subagent)
1. Claim card by rename → read node's `idea.md` + `memory/failure_modes.md`
2. Write `model.py` (`build_model(cfg)`) + `config.py`; flip tree.json status to `"coded"`
3. Smoke self-check:
   - Local torch? test: `python -c "import torch"` 
   - If yes: `python code/engine/train.py --node_dir tree/nodes/<ID> --smoke --epochs 2`
   - If no: commit & push a draft first, then run the same command on the server via ssh
4. Only after green smoke: rename card `_done_`, push, create executor card

### Executor (Lead-only — server training, no subagent)
1. Push, then: `ssh cac-server 'cd /data/repo && git pull && bash scripts/run_node.sh <ID>'`; flip tree.json status to `"running"`
2. Watch: `ssh cac-server 'tmux capture-pane -t node_<ID> -p | tail -30'`
3. **Done signal**: log prints `[engine] done status=...` or result.json status ≠ "running". Then collect locally: `bash scripts/collect_node.sh <ID>`, set tree.json status to done/failed/timeout accordingly, commit

### Feedback Agents ×4 (dispatch: 4 subagents in parallel)
1. Claim cards; read node's `idea.md`, `model.py`, `config.py`, `result.json`, `train.log`
2. Write `feedback/<dimension>.md` per PROTOCOL §2. Scope note: engine saves no images/heatmaps — qualitative feedback works from train.log metrics and code reading

### Synthesis Agent (dispatch: subagent)
1. When 4 feedback files exist: dedupe updates, resolve contradictions, apply quality gate (7 dimensions: mechanistic, scoped, predictive, falsifiable, novel, transferable, actionable)
2. Write `synthesis.md`; book ALL memory events: `create` for new hypotheses, `evidence`/`revise` for existing ones → append to `memory/hypotheses.jsonl` → `python scripts/rebuild_index.py`
3. Update `tree.json`: status `"synthesized"` + scores/best_metric/tested_hypotheses
4. Update `STATE.md` next-steps; commit & push

### Closing Trio after EVERY dispatched role returns (Lead executes)
1. Update STATE.md (if stage changed) 2. Append one journal line 3. Commit & push

## 2. Numbering Conventions

Next free ID = max existing + 1, zero-padded 4 digits: nodes `N####_<slug>` (S-prefixed = smoke, excluded from scoring), task cards `T####_*`, hypotheses `H####`.

## 3. Hard Rules

0. **Inspiration check (user-specified, recurring)**: before EACH Idea Agent dispatch and each Synthesis round, run `git pull --ff-only` and re-read `docs/inspiration_from_GOD.txt`. The user updates it at any time with new direction hints — treat it as a live input, not a one-time read; fold new hints into hypothesis selection and STATE.md next-steps
0. **Never-idle principle (user-specified, permanent)**: the Lead MUST never block-wait on a GPU job. While any node is `running`, the Lead MUST keep the pipeline full in parallel: dispatch the next Idea/Coding subagent(s), run `websearch` grounding, prepare the next executor payload, or repair docs/state. Polling is `grep train.log | tail` in a single ssh — never a 20-minute `for i in seq` loop. A queued `coded` node must be ready to launch the instant the GPU frees. Idling with an empty pipeline is a protocol violation.
1. **Only local pushes**; server pulls. Artifacts return via collect script
2. **Large files never enter git** (datasets/checkpoints/logs stay under `/data/`)
3. Task claiming = atomic file rename; mutual exclusion via `mkdir locks/<name>`
4. `memory/hypotheses.jsonl` is **append-only**
5. Remote tasks >1 min must run in tmux; no bare SSH foreground hangs
6. New code passes `--smoke` before real data
7. Read `memory/failure_modes.md` before coding; append new pitfalls after incidents
8. **Web-grounding mandate (user-specified)**: when stuck — same bug survives 2 fix attempts, unfamiliar error/API behavior, design uncertainty — OR idea-drought (can't formulate falsifiable hypotheses / no expandable direction), you MUST run targeted `websearch` BEFORE the next attempt: search exact error text verbatim, method/paper names, SOTA numbers. Fold findings into `idea.md` grounding, `memory/failure_modes.md`, or STATE.md verified facts. Spinning in place is a protocol violation

## 4. Documentation Budget (anti-entropy)

Docs exist so any agent can resume fast — not as an archive. Violations compound into context-window exhaustion.

- **Line budgets**: README ≤120 · AGENTS ≤180 · PROTOCOL ≤160 · STATE.md ≤60 · idea.md ≤80 · feedback ≤60 each · synthesis.md ≤100
- **Single-home rule**: every fact lives in exactly one document; others link to it, never copy it
- **Rolling vs append-only**: STATE.md is rewritten in place (a snapshot, not a diary). History belongs ONLY in journal/hypotheses (append-only *data*, never read whole — always `tail`)
- **Pruning duty**: whoever's edit pushes a doc over budget must compress it in the same commit (merge superseded sections, delete resolved items). Outdated content is deleted, never annotated "(deprecated)"
- **No changelogs**: docs describe the present protocol; git history is the changelog
- **Node isolation**: agents read one node directory at a time; never bulk-read tree/nodes/

## 5. Server Cheat-Sheet

| Item | Value |
|---|---|
| Connection | `ssh cac-server` (alias auto-updated on rotation) |
| Persistence | only `/data`: repo/, dataset/FSC147/, runs/<ID>/, asset/ |
| Python | `/data/miniconda/envs/cac/bin/python` (torch 2.10.0+cu128, CUDA OK, RTX 3060 12GB) |
| Network | GitHub direct OK; pip needs `-i https://pypi.tuna.tsinghua.edu.cn/simple` |

## 6. Server Rotation Drill

Paste DeepLn's two lines verbatim into `local/address_and_password.md` (`ssh -p <port> root@<host>` then `<password>`), then run `bash scripts/onboard.sh`. Idempotent. If dataset lost: upload zip to `/data/dataset/`, unzip inside `FSC147/`, re-run.

Out-of-repo fixed deps (configured, don't touch): `~/.git-credentials`, `~/.ssh/id_ed25519`, commit email `qkun-zh@users.noreply.github.com`.

## 7. Hypothesis Format

```
IF [choice] IN [scope], THEN [effect], BECAUSE [mechanism]. DISPROVED IF [criterion].
```
Confidence: η=0.20, c∈[0.01,0.99], init 0.5; supports `c←c+η·w·(1−c)`, contradicts `c←c−η·w·c`; confirmed >0.75 / refuted <0.25.
