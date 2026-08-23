# cac-explore — Multi-Agent Discovery System for Lightweight CAC Counting

## Mission

> **Build an innovative Class-Agnostic Counting (CAC) model that achieves MAE ≤ 4 on the FSC147 test split.**
>
> **No architecture restrictions** — full backbone fine-tuning allowed, any parameter budget, any paradigm. Must be innovative and hit the essence of the counting problem.

This is a hard target beyond current public methods. The Lead does not write models directly — models are produced by the **multi-agent hypothesis-exploration loop** defined in this repository. Any agent taking over can losslessly resume after reading only three files: this README, `AGENTS.md`, and `STATE.md`.

Reference baseline: `tree/nodes/S0001_smoke/` (0.01M-param toy network, val MAE 46.7 @ 2 epochs) — proves the pipeline works end-to-end; not comparable to the target.

## System Architecture

Replicates [HypoExplore](https://arxiv.org/abs/2604.12999): **one Git repository = the shared file system of all agents**. All state lives in files, never in any agent's context window.

```
┌───────────────────────────────────┐
│ Local WSL Debian                  │  Lead + Idea/Coding/Feedback/Synthesis agents
│ ~/cac_explore                     │  ★ the ONLY machine with push rights
└──────────┬────────────────────────┘
           │ git push / pull
┌──────────▼────────────────────────┐
│ GitHub: qkun-zh/cac-explore       │  shared bus (public repo)
└──────────┬────────────────────────┘
           │ git pull (server is read-only)
┌───────────────────────────────────┐
│ DeepLn rented GPU server          │  the only place with a GPU (RTX 3060 12GB)
│ /data/repo + /data/runs           │  tmux training; artifacts collected back via SSH
└───────────────────────────────────┘
```

## Directory Map (what every file is for)

| Path | Purpose |
|---|---|
| `AGENTS.md` | **Protocol**: startup sequence, per-role work loops, hard rules, server cheat-sheet, rotation drill |
| `STATE.md` | **Current situation snapshot**: stage, verified facts, active tasks, next steps. Read second |
| `docs/PROTOCOL.md` | File contract details: required structure of every file in a node dir, research-cycle formulas, state machine |
| `docs/research_direction.md` | Research direction memo: CAC landscape, FSC147 data protocol, candidate technical routes |
| `docs/arXiv-2604.12999_HypoExplore_summary.txt` | Distilled paper summary — the framework we replicate |
| `code/engine/train.py` | **The single training entry point**, shared by all nodes. Contract: reads node's model/config, writes result.json |
| `code/data/fsc147.py` | FSC147 VarV2 dataset loader (precomputed density maps, sum-preserving resize, exemplar-box parsing) |
| `code/selection/select_next.py` | Trajectory-tree expansion policy: parent choice (quality×avail), hypothesis choice (Thompson sampling + epistemic value) |
| `scripts/run_node.sh` | [Server] launch training for a node in tmux (session `node_<ID>`, 30-min wall clock via engine flag) |
| `scripts/collect_node.sh` | [Local] pull result.json + train.log tail from server into the node directory |
| `scripts/check_data.py` | Dataset sanity check (split sizes, shapes, count conservation) |
| `scripts/bootstrap_remote.sh` | [Server] idempotent one-shot environment init |
| `scripts/install_key.py` | [Local] after instance rotation: install SSH pubkey & rewrite connection alias (reads `local/address_and_password.md`) |
| `scripts/onboard.sh` | [Local] **one-command server onboarding**: creds → key → repo → env → data verification |
| `scripts/rebuild_index.py` | Rebuild index.json from hypotheses.jsonl (self-repair if snapshot corrupts) |
| `scripts/sync.sh` | Local convenience: pull --autostash, commit pending changes, push |
| `scripts/revproxy.py` | Spare: local socks proxy for the server (unused by default; GitHub is reachable directly) |
| `local/` | **Machine-local secrets (gitignored, NEVER committed)**: `address_and_password.md` server credentials, `github_token.txt` token backup |
| `tasks/_template.md` | Task-card template; `T####_pending_*.md` unclaimed, **rename to `*_claimed_*` to claim** |
| `journal/events.jsonl` | Global audit stream (append-only): who did what when |
| `memory/hypotheses.jsonl` | **Hypothesis memory bank** (append-only; history lines are never rewritten) |
| `memory/index.json` | Rebuildable snapshot of the bank: per-hypothesis confidence/status/evidence log |
| `memory/failure_modes.md` | **Pitfall list**: Coding agents must read before writing code; must append after any incident |
| `tree/tree.json` | Trajectory tree T: parent/child links + status/best_metric/score, maintained by Synthesis |
| `tree/nodes/<ID>/` | One self-contained directory per experiment: idea → code → result → feedback ×4 → synthesis |

Node ID convention: `S0001_smoke` is the smoke-test node; production nodes start from `N0002_<short-name>`.

## Quick Start for a New Agent

```bash
git pull --ff-only             # 1. sync to latest
cat AGENTS.md STATE.md         # 2. protocol + situation (then docs/PROTOCOL.md if needed)
tail -5 journal/events.jsonl   # 3. what happened recently
```

Then follow your role's work loop in `AGENTS.md`.
