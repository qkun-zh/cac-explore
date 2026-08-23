# idea.md — N0016_dino_seqcount

## Title
SeqCount paradigm transplant: autoregressive discrete count-token generation on the frozen DINOv2 champion encoder stack

## Motivation & Intuition
- Champion N0010 (val MAE 21.53) has an unsolved failure mode: RMSE/MAE=3.63x catastrophic outliers. Density regression with implicit Gaussian-kernel GT mismatches object scale, and high-count images dominate MSE gradients — every prior tail fix failed (H0018 neutral, H0020 refuted, H0022 untested).
- `docs/inspiration_from_GOD.txt` §1–3 (SeqCount, read 2026-08-23): convert CAC from density-map regression to sequence generation. Patch-level integer counts become discrete tokens; a causal Transformer decoder generates the row-major sequence; sum = total count. Validated on FSC-147/FSCD-LVIS/UAVVIC/TRANCOS/CARPK/PUCPR+.
- Imported mechanisms: (a) per-patch integer classification removes kernel/scale mismatch entirely; (b) causal self-attention models neighbor-count correlations (paper ablation: independent per-patch prediction significantly worsens MAE/RMSE); (c) cross-attention focuses each token on its own image region, mitigating objects split across patch borders.
- User directive (journal 2026-08-23): "若要尝试，seqcount的范式应该排在前列" — SeqCount is FIRST in the paradigm-attempt queue.

## Architecture Spec
core_ideas: keep the proven champion encoder VERBATIM (H0014/H0017 confirmed); change only head + loss + inference paradigm (regression -> generation).

core_blocks:
1. Encoder (verbatim N0010): frozen DINOv2-S reg4, timm features_only, out_indices=(6,11), 392px -> dual taps 784 tokens each; scalar layer-gate fusion; exemplar area-prompt; adapter768. ~21M frozen.
2. Memory projection: Linear 768->256 over fused tokens; serves as cross-attention memory (K/V).
3. Serialization: counting grid N=14 aligned to token grid -> 14x14=196 tokens, each covers 2x2 DINOv2 patches (adaptive_avg_pool 28->14 on fused map); row-major flatten; special Start/End tokens; vocab K=64 integer count tokens, targets clamped to K-1=63 (mitigates SeqCount's OOV limitation §5.1).
4. Decoder: causal Transformer decoder, 4 layers, d=256, ffn=512, heads=4, norm_first, dropout=0.1; input = [Start] + word_embedding(count token) + positional encoding; masked self-attn + cross-attn to projected encoder tokens; LM head -> K logits per position.
5. Loss/inference: train = cross-entropy mean over all 196 positions; eval = teacher-forcing OFF, autoregressive from Start until End or L=196 steps, pred_count = sum of per-step argmax.

network_structure: [B,3,392,392]+bboxes -> frozen encoder(gate+prompt+adapter) -> tokens[784,768] -> proj -> mem[784,256]; decoder inputs [Start,c1..c195] <-> xattn(mem) -> logits[B,196,64].

tunable_aspects: N in {10,14,20}; K in {32,64}; decoder depth {2,4} and width {192,256}; embedding/decoder lr multiplier.

invariants: backbone frozen; <=32M total params; fixed row-major order; NO Gaussian kernels anywhere; seq-mode output contract = {"seq_logits":[B,L,K]} (engine-seq consumes it; regression "density" contract untouched).

Params: 21M frozen + gate/prompt/adapter ~0.6M + decoder 4L ~3.2M + embeddings <0.2M ~= 25M <= 32M OK

## Proposed Hypotheses
- H0024: IF SeqCount-style sequence generation replaces density regression IN FSC147 on the champion encoder stack, THEN val MAE <= 19.0 AND RMSE/MAE < 3.3, BECAUSE discrete per-patch classification avoids Gaussian-kernel scale mismatch and causal context models neighbor correlations. DISPROVED IF MAE > 21.53 OR engine-seq mode fails smoke twice.

## Delta vs Parent (N0010_dino_multilayer_long)
- Same encoder stack, resolution, optimizer family; NEW: engine `paradigm:"seq"` mode — integer targets built by adaptive-pooling gt density to NxN + round + clamp(K-1); CE loss; autoregressive argmax-sum eval (engine extension required, precedent: Huber added for N0011).
- Head swapped: adapter+MLP density head -> projection + 4-layer causal decoder; inference switched from density summation to sequence generation.
- Accepted risks: vocab saturation on extreme patches (clamped at 63); AR latency ~196 steps (val-only, acceptable); high-count long tail deferred to mosaic/augreg merge (SeqCount+ recipe, cf. N0013 line).

## Novelty Statement
First paradigm-level (not mechanism-level) change in this search: regression -> discrete sequence generation on a category-agnostic frozen ViT substrate. Tests whether champion-encoder gains transfer across task formulations and directly attacks the RMSE-tail problem that H0018/H0020 could not fix.
