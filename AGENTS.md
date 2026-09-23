# cac v2 - Training-Free Class-Agnostic Counting

Mission: training-free FSC147 counting, subset286 MAE <= 20.0.
Start point: N0029 TF champion (CountingDINO + DINOv3 ConvNeXt-T, full-val 26.485, subset286 25.818).

## Rules for agents (enforced by scripts/check.py, not by trust)

1. Run `python3 scripts/check.py` before every commit. Red = stop, fix first.
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

## First session (fresh clone)

```bash
python3 scripts/seed_db.py       # creates neug.db (gitignored): N0029 + 6 bans
python3 scripts/check.py         # must print CHECK OK
python3 scripts/graph.py live-parent   # parent + bars + bans in one call
```

## One evolution step

```bash
# 1. context
python3 scripts/graph.py live-parent
python3 scripts/graph.py untested --parent <live_parent>
# 2. book hypothesis (format + novelty gate run inline)
python3 scripts/graph.py new-hypo --text "IF <choice> IN <scope>, THEN <measured effect>, BECAUSE <mechanism>. DISPROVED IF <bar with a number>."
# 3. implement BEFORE booking the node: declare --<flag_name> (str2bool/int,
#    default off) in models/cdino/tf_pipeline/args.py + validate in validate_args();
#    gate the logic with getattr(config, "<flag_name>", False). One switch only.
#    new-model refuses any switch not yet declared in args.py.
# 4. book child node. --switch IS the CLI flag name (without --).
python3 scripts/graph.py new-model --parent <live_parent> --hypo Hxxxx --switch <flag_name>
python3 scripts/check.py         # must print CHECK OK (also verifies flag exists)
# 5. run on server (sync first: bash scripts/sync.sh)
bash scripts/sync.sh
ssh cac-server 'bash /data/cac/models/run_variant.sh Nxxxx --<flag_name> True'
# 6. record evidence. --type is REQUIRED:
#    support    iff subset_mae <= that run's next_bar (cleared the bar)
#    contradict iff subset_mae >  bar (missed the bar / got worse)
#    neutral    only when the run is uninterpretable (crash before any metric)
python3 scripts/graph.py evidence --hypo Hxxxx --model Nxxxx --mae <v> --type support|contradict --note "..."
python3 scripts/check.py         # green again, then commit
```

## Worked example (concrete; tta_flip is already declared in args.py)

```bash
python3 scripts/graph.py new-hypo --text "IF tta_flip IN density readout, THEN subset286 MAE drops by at least 0.30, BECAUSE averaging horizontal-flip TTA cancels orientation-sensitive peak jitter in the density map. DISPROVED IF subset286 MAE is not at least 0.30 lower than parent 25.818."
# prints Hxxxx (next auto id, e.g. H0002); never invent or reuse an id
python3 scripts/graph.py new-model --parent N0029 --hypo Hxxxx --switch tta_flip --params 0
# prints Nxxxx (next auto id, e.g. N0031); gate passes because --tta_flip is in args.py
python3 scripts/check.py
bash scripts/sync.sh
ssh cac-server 'bash /data/cac/models/run_variant.sh Nxxxx --tta_flip 1'
# read final MAE from /data/repro/logs/Nxxxx_sub286.log, then record evidence:
#   support    iff subset_mae <= that run's next_bar (cleared the bar)
#   contradict iff subset_mae >  bar (missed the bar)
#   neutral    only if the run crashed before any metric
python3 scripts/graph.py evidence --hypo Hxxxx --model Nxxxx --mae <v> --type support --note "..."
python3 scripts/check.py
```

## Model code

Runnable champion lives in `models/` (pulled from server `/data/cdino_run`):
- `models/cdino/` - pipeline package (entry, args, backbones, postprocess, src).
- `models/champion.json` - winning flags (N0029: MWEx + ADEI t1.5d1m8 + fs=0.5).
- `models/run_champion.sh` - reproduce the champion exactly.
- `models/run_variant.sh <NODE_ID> [flags...]` - champion flags + your overrides.
- `scripts/sync.sh` - sync this repo to the canonical server dir `/data/cac`
  (previous server tree archived once at `/data/cac_v1`).

Node switch convention: `switch` on a graph node equals the argparse flag name;
new-model refuses any switch not declared in `args.py`. Champion root N0029 has
switch=null (its flags are fixed in run_champion.sh).

See GRAPH_GUIDE.md for query reference, PROTOCOL.md for eval protocol.
