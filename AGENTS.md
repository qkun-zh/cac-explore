# AGENTS.md — Multi-Agent Collaboration Protocol

**Mission**: a lightweight, innovative CAC model with ≤32M parameters and MAE < 16 on FSC147 test. Approach it through HypoExplore-style hypothesis exploration, not single-shot heroic design.

---

## 0. Startup Sequence (mandatory for every agent / new session)

1. Read this document
2. `git pull --ff-only`
3. Read `STATE.md` (stage, verified facts, active tasks, next steps)
4. Read on demand: `docs/PROTOCOL.md` (file contracts), tail of `journal/events.jsonl` (recent events), `memory/failure_modes.md` (pitfalls)

---

## 1. Your Work Loop (find your role)

### Idea Agent
1. Read `memory/index.json`, `memory/failure_modes.md`, and the parent node's `synthesis.md`
2. Select parent node and hypothesis set per `docs/PROTOCOL.md §4`
3. Write `tree/nodes/<ID>/idea.md` (fixed sections, falsifiable claims; declare novelty against existing ideas)
4. Create task card `tasks/T####_pending_coding_<ID>.md`; commit & push

### Coding Agent
1. Claim the task card (rename it), read the node's `idea.md` plus **mandatory** `memory/failure_modes.md`
2. Write `model.py` (must expose `build_model(cfg)`) and `config.py` (see PROTOCOL §2 for cfg keys)
3. Smoke self-check (no dataset needed):
   ```bash
   python code/engine/train.py --node_dir tree/nodes/<ID> --smoke --epochs 2
   ```
   If torch is unavailable locally, run the same command on the server. **Do not push until the contract passes**
4. Rename task card to `_done_`, push; create the executor task card

### Executor (server side, triggered by Lead)
1. Locally: push, then launch on the server:
   ```bash
   ssh cac-server 'cd /data/repo && git pull && bash scripts/run_node.sh <ID>'
   ```
2. Watch progress: `ssh cac-server 'tmux capture-pane -t node_<ID> -p | tail -30'`
3. When finished, collect results locally and commit:
   ```bash
   bash scripts/collect_node.sh <ID>
   git add -A && git commit -m "result: <ID> status=..." && git push
   ```

### Feedback Agents ×4 (quantitative / qualitative / causal / diagnostic)
1. Claim your task card; read the node's `idea.md`, `model.py`, `config.py`, `result.json`, `train.log`
2. Write `tree/nodes/<ID>/feedback/<dimension>.md` (fixed structure in PROTOCOL §2, including hypothesis_updates list)

### Synthesis Agent
1. Once all four feedback files exist: merge & deduplicate updates, resolve contradictions, run the quality gate
2. Write `synthesis.md`; update confidence for each affected hypothesis using η=0.20 rules
3. Bookkeeping: append to `memory/hypotheses.jsonl` → run `python scripts/rebuild_index.py`
4. Update `tree/tree.json` (node status & scores) → update `STATE.md` → append journal → commit & push

### Closing Trio for EVERY role (non-negotiable)
1. Update `STATE.md`  2. Append one journal line  3. `git add -A && git commit && git push`

---

## 2. Hard Rules

1. **Only the local machine pushes**; the server only pulls. Experiment artifacts come back via `scripts/collect_node.sh` and are committed locally
2. **Large files never enter git**: datasets, checkpoints, full logs stay on the server under `/data/dataset`, `/data/runs`
3. Task claiming is atomic via file rename; mutually exclusive resources use `mkdir locks/<name>`, delete when done
4. `memory/hypotheses.jsonl` is **append-only; history lines are never rewritten**
5. Any remote task longer than ~1 minute must run inside tmux; no bare SSH foreground hangs
6. New code must pass `--smoke` before touching real data
7. Read `memory/failure_modes.md` before writing code; append new pitfalls after any incident

## 3. Server Cheat-Sheet

| Item | Value |
|---|---|
| Connection | `ssh cac-server` (alias mapped in local `~/.ssh/config`, auto-updated on rotation) |
| Persistence | only `/data`: `repo/` (clone), `dataset/FSC147/` (VarV2 full set), `runs/<ID>/`, `asset/` (scratch) |
| Python | `/data/miniconda/envs/cac/bin/python` (torch 2.10.0+cu128, CUDA OK, RTX 3060 12GB) |
| Network | GitHub reachable directly; pip MUST use `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| Data check | `/data/miniconda/envs/cac/bin/python scripts/check_data.py` must fully pass |

## 4. Server Rotation Drill (run every time a new instance is rented)

**The only thing you need to do**: paste DeepLn's two lines verbatim into `local/address_and_password.md` —
```
ssh -p <port> root@<host>
<password>
```
Then one command locally restores everything:
```bash
bash scripts/onboard.sh
```
(Automatically: parse creds → install pubkey → write ssh alias → clone repo if missing → init env → verify dataset. Idempotent, safe to re-run.)

Extra step if the dataset was lost: upload FSC147.zip to `/data/dataset/`, unzip inside the `FSC147/` directory, re-run onboard.sh.

**Out-of-repo fixed dependencies (already configured, do not touch)**: push credentials in `~/.git-credentials` (token backup in `local/github_token.txt`); SSH key at `~/.ssh/id_ed25519`; commit email must be `qkun-zh@users.noreply.github.com`.

## 5. Hypothesis Record Format

```
IF [architectural choice] IN [scope], THEN [predicted effect], BECAUSE [mechanism]. DISPROVED IF [falsification criterion].
```

Confidence update (η=0.20, c∈[0.01,0.99], initial 0.5): supports `c←c+0.20·w·(1−c)`; contradicts `c←c−0.20·w·c`. Verdicts: confirmed >0.75 / refuted <0.25 / uncertain otherwise.
