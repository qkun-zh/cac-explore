# cac — Active Hypothesis Exploration for Crowd Counting

HypoExplore ([arXiv:2604.12999](https://arxiv.org/abs/2604.12999)) applied to
FSC147 crowd counting. The trajectory tree IS a directory tree: every branch in
`tree/` (physical nesting) is the lineage of one architecture family, rooted at
the certified champion **N0001_champion** (MAE 19.647 / RMSE 74.05 / 31.32M).

`tree/` (topology + `info.json`) and `memory/hypotheses.jsonl` (append-only
ledger) are the only ground truth; `src/cac/expt/` is the paper's decision
rules implemented as code. Read **AGENTS.md first**.

## Quickstart (local, no GPU needed)
    pip install -e .           # python 3.13
    python scripts/conformance.py          # must print CONFORMANCE OK
    python scripts/discovery.py tree       # ASCII tree
    python scripts/discovery.py validate   # format-gate hypothesis ledger
    python tests/test_all.py 2>/dev/null || python -m pytest tests

## One evolution step (each new branch is one iteration)
    python scripts/discovery.py parent                 # Eq.2–4 parent
    python scripts/discovery.py hypo <parent>          # Eq.5–6 child + hypothesis set
    python scripts/novelty_check.py "<new hypothesis>" # novelty gate
    ... edit tree/<child>/config.toml + model.py (delta from parent only) ...
    python scripts/conformance.py                      # green before commit
    # on cac-server:
    python scripts/run_node.py <child>                 # smoke → budget (τ_max 1800s)
    python scripts/discovery.py calibration            # bin table

## Layout
- `src/cac/expt/` — constants, node (tree), hypothesis (ledger+Eq.1), select
  (Eq.2–6), gates (calibration + conformance).
- `src/cac/engine|models|data|calls/` — runner (smoke→budget, checksums),
  champion counter, Lightning pl_module, FSC147 datamodule, best/EMA callbacks.
- `scripts/` — discovery, conformance, novelty_check, run_node, install_key.
- `tree/` — trajectory tree; `memory/` — ledger + index; `journal/` — events;
  `docs/research_direction.md` — mission (changes only there, journaled).

## Reproducibility
A node reruns bit-identical under its `config.toml` seed (training seeds
torch/numpy/random, seeded loader + worker_init). Every `result.json` records
`config_sha256`/`model_sha256`.