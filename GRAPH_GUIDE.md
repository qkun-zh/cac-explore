# GRAPH_GUIDE — NeuG read/write manual (one page)

NeuG is a SQLite graph at `neug.db` (gitignored). Tables: `nodes`, `edges`.
Node types: `model | hypo | ban`. Edge types: `CHILD_OF | TESTS | SUPPORTS | CONTRADICTS`.

## The three queries an agent needs (one call each)

```bash
python scripts/graph.py live-parent   # best model + next bars + all bans (start here)
python scripts/graph.py untested --parent N0029   # hypos not yet tested under parent ancestry
python scripts/graph.py bans          # all BanRule nodes
```

## Writes (only entry points; direct SQL/DB edits fail check.py)

```bash
python scripts/graph.py new-hypo --text "IF ... IN ... THEN ... BECAUSE ... DISPROVED IF ..."
python scripts/graph.py new-model --parent N0029 --hypo H0001 --switch use_foo --params 0
python scripts/graph.py evidence --hypo H0001 --model N0030 --mae 25.1 --type contradict --weight 0.85 --note "..."
python scripts/graph.py ban --rule "never: <mechanism>" --ref H0001
python scripts/graph.py tree          # ASCII lineage
```

## Naming (search-friendly, fixed prefixes)

- Models: `Nxxxx` (zero-padded, auto-increment). Hypos: `Hxxxx`. Bans: `Bxxxx`.
- Switches: `use_<lowercase_name>`. Edges carry no payload; all metrics live on `model` nodes.

## Confidence (code-owned, never hand-computed)

eta=0.20. support: `c += eta*w*(1-c)`. contradict: `c -= eta*w*c`.
confirmed > 0.75, refuted < 0.25. Computed inside `graph.py evidence` only.
