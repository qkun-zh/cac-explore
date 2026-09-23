# cac v2 - Training-Free Class-Agnostic Counting

Mission: training-free FSC147 counting, subset286 MAE <= 20.0.
Start point: N0029 TF champion (CountingDINO + DINOv3 ConvNeXt-T, full-val 26.485, subset286 25.818).

## Rules for agents (enforced by scripts/check.py, not by trust)

1. Run `python3 scripts/check.py` before every commit. Red = stop, fix first.
   `.git/hooks/pre-commit` runs it automatically (install: `python3 scripts/install_hooks.py`).
2. Never edit `neug.db` directly. Use `python3 scripts/graph.py <cmd>` only.
3. Never invent node IDs. IDs come from `graph.py new-model` / `new-hypo`.
4. Source: Python (`*.py`) + Bash (`*.sh`) only. Data: `*.json`/`*.toml`/`*.lock`.
   Docs: exactly the three .md files at repo root. Anything else fails check.py.
5. Docs are pure ASCII English (any non-ASCII char in them fails check.py).
   CJK anywhere in code/comments also fails check.py.
6. Every experiment runs on subset286 only. Full val 1286 only on explicit user request.
7. Required model `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` must stay in the
   inference graph (check.py verifies it in models/**.py). Total stack <= 64M params
   (check.py verifies every model node). Training-free only: no gradient updates,
   no fit-on-val-GT for board numbers.
8. A model node must leave `status=open` (evidence or fail) before any commit.
   `evidence --log` parses the server log for MAE and auto-judges support/contradict
   against the bar; you never hand-type MAE or support/contradict yourself.
9. One override flag per run. `run_variant.sh` rejects >1 flag and all closed-family
   / champion-constant keys. Closed families live in `scripts/closed_families.json`
   and are also ban nodes (auto-seeded).

## First session (fresh clone)

```bash
python3 scripts/seed_db.py       # installs pre-commit + creates neug.db: N0029 + bans
python3 scripts/check.py         # must print CHECK OK
python3 scripts/graph.py live-parent   # parent + bars + bans in one call
```

## One evolution step

```bash
# 1. context + diagnosis (diag drives where to aim; 200+ bin is the known lever)
python3 scripts/graph.py live-parent
python3 scripts/graph.py untested --parent <live_parent>
#   scp the champion/candidate log or per-image csv first, then:
python3 scripts/diag.py --log /tmp/champion_sub286.log
# 2. book hypothesis (format + closed-keyword + novelty gates run inline)
python3 scripts/graph.py new-hypo --text "IF <choice> IN <scope>, THEN <measured effect>, BECAUSE <mechanism>. DISPROVED IF <bar with a number>."
# 3. implement BEFORE booking the node: declare --<flag_name> (str2bool/int,
#    default off) in models/cdino/tf_pipeline/args.py + validate in validate_args();
#    gate the logic with getattr(config, "<flag_name>", False). One switch only.
#    new-model refuses switches not in args.py and closed-family switches.
# 4. book child node. --switch IS the CLI flag name (without --).
python3 scripts/graph.py new-model --parent <live_parent> --hypo Hxxxx --switch <flag_name>
# 5. run on server (sync first: bash scripts/sync.sh); exactly one override flag
bash scripts/sync.sh
ssh cac-server 'bash /data/cac/models/run_variant.sh Nxxxx --<flag_name> True'
# 6. pull the log, then record evidence (MAE/type/bar all come from the log):
scp cac-server:/data/repro/logs/Nxxxx_sub286.log /tmp/
python3 scripts/graph.py evidence --hypo Hxxxx --model Nxxxx --log /tmp/Nxxxx_sub286.log --note "..."
#    support    iff parsed_mae <= parent - 0.30 (bar computed inside evidence)
#    contradict iff parsed_mae >  bar
#    crash/no metric -> use fail --reason failed|timeout (confidence unchanged)
#    node stays open -> check.py fails; evidence or fail is mandatory before commit
python3 scripts/check.py         # green again, then commit
```

## Worked example (concrete; num_exemplars is open and declared in args.py)

```bash
python3 scripts/graph.py new-hypo --text "IF capping num_exemplars IN exemplar aggregation, THEN subset286 MAE drops by at least 0.30, BECAUSE fewer noisy tail exemplars reduce ROI-mass error on dense images. DISPROVED IF subset286 MAE is not at least 0.30 lower than parent 25.818."
# prints Hxxxx (next auto id); never invent or reuse an id
python3 scripts/graph.py new-model --parent N0029 --hypo Hxxxx --switch num_exemplars --params 0
# prints Nxxxx; gate passes because --num_exemplars is in args.py and not a closed switch
python3 scripts/check.py
bash scripts/sync.sh
ssh cac-server 'bash /data/cac/models/run_variant.sh Nxxxx --num_exemplars 8'
scp cac-server:/data/repro/logs/Nxxxx_sub286.log /tmp/
python3 scripts/graph.py evidence --hypo Hxxxx --model Nxxxx --log /tmp/Nxxxx_sub286.log --note "..."
python3 scripts/check.py
```

## Model code

Runnable champion lives in `models/` (pulled from server `/data/cdino_run`):
- `models/cdino/` - pipeline package (entry, args, backbones, postprocess, src).
- `models/champion.json` - winning flags (N0029: MWEx + ADEI t1.5d1m8 + fs=0.5).
- `models/run_champion.sh` - reproduce the champion exactly.
- `models/run_variant.sh <NODE_ID> [flags...]` - champion flags + one override.
- `scripts/sync.sh` - sync this repo to the canonical server dir `/data/cac`
  (previous server tree archived once at `/data/cac_v1`).
- `scripts/diag.py` - density-bin MAE share + worst15 from a log or per-image csv.
- `scripts/closed_families.json` - closed switches/keywords/bans (source of truth).

Node switch convention: `switch` on a graph node equals the argparse flag name;
new-model refuses switches not declared in `args.py` and switches listed in
`closed_families.json`. Champion root N0029 has switch=null (flags fixed in
run_champion.sh).

See GRAPH_GUIDE.md for query reference, PROTOCOL.md for eval protocol.
