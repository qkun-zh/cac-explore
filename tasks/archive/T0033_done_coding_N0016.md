# T0033 — pending_coding N0016_dino_seqcount

**Node**: `tree/nodes/N0016_dino_seqcount` (parent `N0010_dino_multilayer_long`, champion val MAE 21.53)
**Claim**: atomic rename `_pending_coding_` -> `_claimed_`; after green smoke rename to `_done_` (AGENTS §1 Coding loop).

## Scope
Implement the SeqCount paradigm node per its `idea.md`. **REQUIRES ENGINE EXTENSION** (precedent: Huber loss was added to train.py for N0011):

1. Extend `code/engine/train.py` with cfg flag `paradigm:"seq"`:
   - Targets: adaptive-pool gt density map to NxN (`N=14`, avg_pool kernel 28->14), round, clamp to `K-1` (`K=64`) -> int64 `[B,196]`
   - Train loss: `F.cross_entropy(logits.view(-1,K), targets.view(-1))`
   - Eval: teacher-forcing OFF — autoregressive decode from Start until End or L=196 steps; `pred_count = logits.argmax(-1).sum(1)` vs gt counts; log MAE/RMSE as usual
   - Regression path MUST stay byte-identical when the flag is absent
2. Write node's `model.py`: `build_model(cfg)` — champion encoder stack verbatim (DINOv2-S reg4 features_only out_indices=(6,11) @392 + scalar layer-gate + area-prompt + adapter768, frozen) -> Linear 768->256 memory -> causal Transformer decoder 4L/d=256/ffn=512/heads=4/norm_first + word/Start/End embeddings -> seq_logits `[B,L,K]`
3. Write node's `config.py`: `paradigm:"seq"`, input_size=392, N=14, K=64, epochs/lr per parent family
4. Flip tree.json status -> `coded`

## Smoke gate (MANDATORY — covers BOTH paradigms)
```
python code/engine/train.py --node_dir tree/nodes/N0010_dino_multilayer_long --smoke --epochs 2   # regression champion unchanged
python code/engine/train.py --node_dir tree/nodes/N0016_dino_seqcount      --smoke --epochs 2   # new seq mode
```
Read `memory/failure_modes.md` before coding. Lead owns all git operations.
