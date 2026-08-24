# Idea — N0027_norm_flip_swa (parent: N0021_dino_partialft, champion val MAE 20.44 @23.26M)
Data-hygiene retrain: recipe otherwise IDENTICAL to champion (40ep bs8 lr1e-3 cosine AMP,
partial FT blocks10-11 @lr×0.1, dropout 0.1) — only the changes below differ. Verified facts:
fsc147.py:68 emits imgs/255 only (NO ImageNet norm → frozen DINOv2 blocks 0-9 consume
off-distribution input); fsc147.py:24 augment flag exists with flip logic :63-66 but
train.py:137 never passes it; engine keeps no weight average; result.json headline is
last-epoch evaluate(), not best (best_mae IS logged — synthesis must read that column).

## Change A — input normalization (H0037, model-level)
Normalize imgs INSIDE model.forward with timm default ImageNet stats applied to the /255
tensor: x = (x − mean)/std, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225) (registered
buffers; node-local — shared fsc147.py untouched, so every eval script feeding /255 keeps
working against the new ckpt automatically).
**H0037**: IF ImageNet norm THEN val best MAE ≤19.94 (=20.44−0.5) BECAUSE frozen-layer input
statistics realign with DINOv2 pretraining. DISPROVED IF MAE ∈ [20.14, 20.74].
lr MAY need retune (norm changes grad scale into the frozen stack) — deliberately NOT here:
measuring hygiene under the frozen recipe is the point. One follow-up arm (lr_mult ∈
{0.05,0.2}) only if disproof lands in that noise band.

## Change B — horizontal-flip aug (H0038, train split only)
Enable the EXISTING flip path: engine make_loaders passes augment=cfg["augment"] to the train
DS (train.py:137); p=0.5 flips img+density+bbox consistently (fsc147.py:63-66).
Honesty: ONE run cannot attribute flip separately from norm — the single training arm carries
BOTH changes. **H0038 (joint form)**: IF norm+flip THEN joint-arm best MAE ≤19.74 BECAUSE
mirror symmetry doubles effective train data at zero cost and counts are flip-invariant.
DISPROVED IF joint MAE >20.44 (worse than parent). Flip-term attribution via a norm-only
second arm runs ONLY IF the joint arm wins; otherwise moot.

## Change C — SWA-lite (H0039, ENGINE piggyback, zero extra training time)
Running uniform average of TRAINABLE params over epoch ends E14–E28 (CPU accumulator, no VRAM
cost); saved as swa.pth in run_dir + ONE post-loop evaluate() with averaged weights; metrics
gain {"swa_mae","swa_rmse"} beside best_mae. Motivation: champion late tail-drift (val RMSE
79.8→83.1 over E26–40 while train loss fell) + synthesis rec for EMA-type stabilizers.
**H0039**: IF SWA(E14–E28) THEN swa-MAE < best-single-MAE − 0.2 BECAUSE parameter averaging
selects the flat-minimum basin and cancels tail-drift oscillation. DISPROVED IF gap < 0.2.

## Free rider — dual-res eval@448 (optional, flagged)
cfg["dual_res_eval"]=True (this node only): engine builds one extra val loader @448 and
evaluate()s it at each epoch end, logging mae448/rmse448 into the running result.json. ~10
lines, evaluate() signature untouched; enables H0036-style routing readout on ANY new ckpt.
Default OFF for all other nodes; drop this rider entirely if it pushes the engine diff past
~15 lines — SWA takes priority.

## Run protocol & decision table
Single training run: cfg = champion config + {"augment": true, "swa_start": 14, "swa_end":
28, "dual_res_eval": true}. Same seed discipline; σ≈±0.3 noise floor.
- joint MAE ≤19.74 → H0038(joint) PASS; launch norm-only attribution arm for flip term.
- 19.74 < MAE ≤19.94 → H0037 PASS alone; flip unresolved pending attribution arm.
- 19.94 < MAE < 20.14 → inconclusive noise zone: book weak evidence (w≤0.5), no verdict.
- MAE ∈ [20.14, 20.74] → H0037 DISPROVED (hygiene gain absent beyond noise).
- MAE >20.74 → strong disproof of A and of B(joint); inspect swa_mae/mae448 for salvage.

## Risks
- Norm shift moves features wholesale → transient slower head fit under unchanged cosine;
  accepted, that shift-in-convergence IS the measured quantity.
- SWA window assumes 40ep (parent best @E25 sits mid-window); if run stops <28, average over
  available [14, ep_last] and mark truncated in swa.pth metadata.
- Flip↔prompt interaction: flipped bbox must still cover the object — sign check (S−x2,S−x1)
  is implemented (:66) but assert once in smoke.
