# GRAPH_GUIDE - NeuG read/write manual (one page)

NeuG is a SQLite graph at `neug.db` (gitignored; seed with `python3 scripts/seed_db.py`).
Tables: `nodes(id,type,data)`, `edges(src,rel,dst)`.
Node types: `model | hypo | ban`. Edge types: `CHILD_OF | TESTS | SUPPORTS | CONTRADICTS`.

## The three queries an agent needs (one call each)

```bash
python3 scripts/graph.py live-parent          # best model + next bars + all bans (start here)
python3 scripts/graph.py untested --parent Nxxxx   # hypos not tested in that subtree yet
python3 scripts/graph.py bans                 # all BanRule nodes
python3 scripts/graph.py tree                 # ASCII lineage
```

## Writes (only entry points; direct SQL/DB edits fail check.py)

```bash
python3 scripts/graph.py new-hypo --text "IF ... IN ... THEN ... BECAUSE ... DISPROVED IF ... 0.30 ..."
python3 scripts/graph.py new-model --parent N0029 --hypo H0001 --switch mass_sharpen --params 0
python3 scripts/graph.py evidence --hypo H0001 --model N0030 --mae 25.10 --type support --weight 0.85 --note "..."
python3 scripts/graph.py ban --rule "never: <mechanism>" --ref H0001
```

## Naming (search-friendly, fixed prefixes)

- Models: `Nxxxx` (zero-padded, auto-increment from N0029). Hypos: `Hxxxx`. Bans: `Bxxxx`.
- Model `switch` = the CLI flag name (no `--`); new-model refuses switches not
  declared in `models/cdino/tf_pipeline/args.py`. Root champion switch=null.
- Edges carry no payload; all metrics live on `model` nodes
  (`subset_mae`, `full_mae`, `params`, `status`, `note`).

## Evidence rules (decide --type before running)

- `support`: `subset_mae <= next_bar_subset` at the time of the run (bar cleared).
- `contradict`: `subset_mae > next_bar_subset` (bar missed / got worse).
- `neutral`: run uninterpretable (crash/NaN before any metric). Not for bad results.
- `--type` is required; unknown model/hypo IDs are rejected.
- Optional `--full-mae` records a full-val-1286 number (champion rows only).

## Confidence (code-owned, never hand-computed)

eta=0.20. support: `c += eta*w*(1-c)`. contradict: `c -= eta*w*c`.
confirmed > 0.75, refuted < 0.25. Computed inside `graph.py evidence` only.
`live-parent` picks min subset_mae with 3dp tie-break toward the shallower node
(no real gain => parent stays live).
