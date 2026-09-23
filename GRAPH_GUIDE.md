# GRAPH_GUIDE - NeuG read/write manual (one page)

NeuG is a Cypher graph at `neug.db` (gitignored directory; seed with `python3 scripts/seed_db.py`).
Requires `pip install neug`. Schema: node tables `Node(id,type,data)`, `Meta(k,v)`;
rel table `Edge(rel)`.
Node types: `model | hypo | ban`. Edge types: `CHILD_OF | TESTS | SUPPORTS | CONTRADICTS`.
Closed families (switches/keywords/bans) live in `scripts/closed_families.json`.

## The three queries an agent needs (one call each)

```bash
python3 scripts/graph.py live-parent          # best model + next bars + all bans (start here)
python3 scripts/graph.py untested --parent Nxxxx   # hypos with no evidence edge in that subtree
python3 scripts/graph.py bans                 # all BanRule nodes
python3 scripts/graph.py tree                 # ASCII lineage
python3 scripts/diag.py --log <run.log>       # density-bin MAE share + worst15 (required in step 1)
```

## Writes (only entry points; direct SQL/DB edits fail check.py)

```bash
python3 scripts/graph.py new-hypo --text "IF ... IN ... THEN ... BECAUSE ... DISPROVED IF ... 0.30 ..."
python3 scripts/graph.py new-model --parent N0029 --hypo H0001 --switch num_exemplars --params 0
# evidence: scp the server log first; MAE/n/type are PARSED, never hand-typed
scp cac-server:/data/repro/logs/Nxxxx_sub286.log /tmp/
python3 scripts/graph.py evidence --hypo H0001 --model N0031 --log /tmp/Nxxxx_sub286.log --weight 0.85 --note "..."
# crash / no metric (terminal; confidence unchanged):
python3 scripts/graph.py fail --hypo H0001 --model N0031 --reason failed|timeout --note "..."
python3 scripts/graph.py ban --rule "never: <mechanism>" --ref H0001
```

## Naming (search-friendly, fixed prefixes)

- Models: `Nxxxx` (zero-padded, auto-increment from N0029). Hypos: `Hxxxx`. Bans: `Bxxxx`.
- Model `switch` = the CLI flag name (no `--`); new-model refuses switches not
  declared in `models/cdino/tf_pipeline/args.py` and switches in closed_families.json.
  Root champion switch=null.
- Edges carry no payload; all metrics live on `model` nodes
  (`subset_mae`, `full_mae`, `params`, `status`, `note`, `evidence_type`, `evidence_bar`).

## Evidence rules (mechanically derived from the log)

- `evidence` requires `--log` (a real scp'd file). It parses EXTENDED_METRICS for
  MAE and n_images; n must equal 286; a Traceback/no-metric log is rejected
  (use `fail` instead).
- `type` is not an argument: support iff `mae <= parent.subset_mae - 0.30`,
  else contradict. Bar and type are stored on the model node.
- One evidence per model: status=done refuses a second call. `--weight` in (0,1],
  confidence clamped to [0,1]. Requires a TESTS edge (booked via new-model).
- `fail --reason failed|timeout` closes a crashed run without moving confidence.
- A model left `status=open` fails `check.py` (no commit until evidence/fail).

## Confidence (code-owned, never hand-computed)

eta=0.20. support: `c += eta*w*(1-c)`. contradict: `c -= eta*w*c`.
confirmed > 0.75, refuted < 0.25. Computed inside `graph.py evidence` only.
`live-parent` picks min subset_mae with 3dp tie-break toward the shallower node
(no real gain => parent stays live).

## Closed families

`scripts/closed_families.json` holds banned switches, hypo keywords, and the
full ban list. `new-hypo` rejects keywords; `new-model` rejects switches;
`check.py` requires every ban rule to exist as a node (auto-added on any
graph.py call if missing).
