# PROTOCOL — File Contracts & Node Lifecycle

## 1. Sync Topology

```
Local ~/cac_explore ──push/pull──> GitHub <──pull── Server /data/repo
 (Lead + subagents edit)                          (runs training)
        ▲                                              │
        └──────── scripts/collect_node.sh (SSH/SCP) ───┘
```

**Single-writer rule**: only the local machine pushes. The server's clone is read-only; experiment artifacts are written under `/data/runs/` and collected back by the script above.

## 2. Node Directory Contract `tree/nodes/<ID>/`

ID format: `N####_<short-name>` for production nodes (`S0001_smoke` is the smoke test).

| File | Author | Content |
|---|---|---|
| `idea.md` | Idea Agent | Fixed sections: `## Title`, `## Motivation & Intuition`, `## Architecture Spec` (core_ideas / core_blocks / network_structure / tunable_aspects / invariants), `## Proposed Hypotheses` (each with falsification), `## Delta vs Parent`, `## Novelty Statement` |
| `model.py` | Coding Agent | Must expose `build_model(cfg) -> nn.Module`; inputs `[B,3,H,W]` + bboxes `[B,4]`; output dict containing `density`. **density may be low-resolution** (e.g. S/8): the engine bilinearly upsamples it to GT size with sum conservation during training; evaluation counts via density sums, which are resolution-independent |
| `config.py` | Coding Agent | Must define `cfg = dict(...)`. **Only required key: `input_size`.** Commonly used optional keys: `epochs, batch_size, lr, weight_decay, eta_min, amp, smoke(default False), max_params_M(default 32), loss_count_weight(default 0.3), data_root(default /data/dataset/FSC147), num_workers(default 4)`. Free to add more |
| `result.json` | Collected from Executor | `{node, status: running\|success\|failed\|timeout, metrics:{mae,rmse,best_mae,...}, timing:{train_seconds,epochs_done}, diagnostics:{oom,instability,smoke,params_M,...}, run_dir, ts}` — rewritten after every epoch while training |
| `feedback/quantitative.md` etc. ×4 | Feedback Agents | Each has fixed structure: `## reasoning` / `## actionable_feedback` / `## hypothesis_updates` (list items: hypothesis_id, evidence_type∈supports/contradicts/neutral, strength∈[0,1], reasoning); diagnostic exists only for failures/timeouts |
| `synthesis.md` | Synthesis Agent | Deduplicated merged updates, quality-gate verdict (7 dimensions), booking list |
| `train.log` | collect script | Tail copy (≤500 lines) of the full server log |

## 3. Global State Files

- `tree/tree.json`: `nodes: {<ID>: {parent, children[], status(proposed|coded|running|done|failed|timeout|synthesized), best_metric, train_seconds, quality, avail, score}}`, maintained by Synthesis.
- `memory/hypotheses.jsonl`: one event per line: `{ts, type(create|evidence|revise), hyp_id, text?, evidence_type?, strength?, source_node?}`.
- `memory/index.json`: `{_meta, hypotheses: {<hyp_id>: {text, confidence, n_tested, status(confirmed|refuted|uncertain), tags[], log[]}}}` — a materialized snapshot of the jsonl, rebuildable via `scripts/rebuild_index.py`.
- `journal/events.jsonl`: audit stream, append-only.

## 4. Research Cycle (standard flow per node)

1. **Pick parent**: `python code/selection/select_next.py parent`
   `score = λ_parent·quality + (1−λ_parent)·avail`, where
   `quality = λ_acc·Acc_norm + (1−λ_acc)·(1 − min(τ,τ_max)/τ_max)` with λ_acc=0.85, λ_parent=0.60, τ_max=30min
2. **Pick hypotheses**: `select_next.py hypo --parent <ID>`
   Exploitation set = Thompson sampling Beta(α,β) (prior (1,1), w=1), top-2; exploration set = epistemic value `1−|2c−1|`, top-2; deduped union ≤4
3. Idea → Coding → redundancy check (compare against existing idea.md for mechanism duplication)
4. Executor on server: `bash scripts/run_node.sh <ID>` (tmux session `node_<ID>`; wall-clock timeout 30min enforced by engine)
5. Local: `bash scripts/collect_node.sh <ID>`
6. Feedback ×4 (parallel claimable task cards) → Synthesis (dedup, contradiction resolution, quality gate, η=0.20 confidence updates)
7. Update tree.json / index.json / STATE.md / journal; commit & push

## 5. Smoke Mode

All new code passes `--smoke` first: synthetic random data, 2 epochs, CPU-capable. Verifies the model/config contract before any real-data run:

```bash
python code/engine/train.py --node_dir tree/nodes/<ID> --smoke --epochs 2
```
