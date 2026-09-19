# N0001_champion — bootstrap seed root

The canonical champion from the authoring lab (/home/qkun/cac_explore/baseline),
N0054_xscale_exemplar lineage. This node is the **single** starting point of
every evolution lineage in `tree/`. It is not a proposal: it is the migrated
ground truth whose ablation history seeds `memory/hypotheses.jsonl`.

- Frozen DINOv3-ConvNeXt-Tiny, intermediate readout hs(2,3) (not final layer).
- Pluggable CountingHead: FineFuser → ExemplarEncoder (cross-attn transformer,
  XScale coarse multi-scale exemplar summary) → cross-attn Condenser →
  DensityDecoder; GCA global-count aux.
- Proven-effective values: AdamW 1e-3, wd 0.08, cosine eta_min 1e-6, bs16, AMP,
  384px, 32 epochs (+ later diversity), seed 20260830.
- FSC147 val: **MAE 19.647 / RMSE 74.05 / 31.32M params**.

## Invariant carried by every descendant
- Backbone frozen; head-only training; parameter assert ≤ 32M.
- `build_model(cfg)` contract: forward(imgs, bboxes[, bboxes3]) -> {"density", ...}.
- Only `out["density"]` feeds the loss.
- Components bolt onto the frozen features + exemplar embedding only
  (single-switch ablatable; §wrap coupling rules of the repo AGENTS.md).

## Why it is the seed
The paper's evolution starts from a provided reference; our reference is the
strongest artifact we have, with an 8-hypothesis evidence ledger migrated so the
exploration never re-litigates settled questions (GCA+, XScale+, frozen
intermediate readout; DDCA−, RGA/extras−, unfreeze−, final-layer readout−).