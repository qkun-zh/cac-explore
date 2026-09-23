# cac v2 — Training-Free Class-Agnostic Counting

Mission: training-free FSC147 counting, subset286 MAE <= 20.0.
Start point: N0029 TF champion (CountingDINO + DINOv3 ConvNeXt-T, full-val 26.485, subset286 25.818).

## Rules for agents

1. Run `python scripts/check.py` before every commit. Red = stop, fix first.
2. Never edit `neug.db` directly. Use `python scripts/graph.py <cmd>` only.
3. Never invent node IDs. IDs come from `graph.py new-model` / `new-hypo`.
4. Only two languages: Python (`*.py`) and Bash (`*.sh`). Other source extensions fail `check.py`.
5. All docs in English. Any non-ASCII prose outside code comments fails `check.py`.
6. Every experiment runs on subset286 only. Full val 1286 only on explicit user request.
7. Required model `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` must stay in the inference graph. Total stack <= 64M params.
8. Training-free only: no gradient updates, no fit-on-val-GT for board numbers.

## One evolution step

```bash
python scripts/graph.py live-parent        # one call: parent + bars + bans
python scripts/graph.py new-hypo --text "..."   # format + novelty gate inline
python scripts/graph.py new-model --parent Nxxxx --hypo Hxxxx
# ... edit models/<id>.py (single use_<name> switch) ...
python scripts/check.py                    # must print CHECK OK
# ... run on server, then:
python scripts/graph.py evidence --hypo Hxxxx --model Nxxxx --mae <v> --note "..."
```

See GRAPH_GUIDE.md for query reference, PROTOCOL.md for eval protocol.
