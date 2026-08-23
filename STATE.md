# STATE — Current Situation

**Stage**: gen-0 COMPLETE — all 4 roots done+synthesized; best N0005 32.66 · Selection for gen-1 next
**Blockers**: none — revproxy alive; all smokes green

## Verified Facts (do not re-learn the hard way)
- Engine contract: single box [B,4] S-space; low-res density OK; <32M total; MSE+0.3·L1 count
- Root results (val): N0005 swin-prompt 32.66 @271s/28.2M · N0003 convnext-xattn 34.26 @430s/16.9M
  · N0004 effnet-gate 40.37 @535s/3.65M · N0002 dinov2-cosine 42.05 @318s/22.2M
- Hypothesis bank: H0008 0.585 & H0004 0.58 (confirmed mechanisms: implicit prompt, cross-attn)
  H0009 0.42 + H0007 0.42 (refuted: implicit-worse, small-backbone-ok) · H0011 0.50 untested top-lever
- timm traps logged in memory/failure_modes.md (img_size, tags ra_in1k, channels-last BHWC, out_indices range)
- Time budget: real nodes used only 15–30% of τ_max=1800s → children can run 25–40 epochs

## Next Steps (in order)
1. `select_next.py parent` + `hypo --parent` → pick gen-1 parent + hypotheses
2. Idea hat: child node merging winners — DINOv2-S/Swin tokens + prompt/cross-attn + box-area channel (H0011) + ≥25ep
3. Standard loop to synthesis; iterate toward MAE<16

## Active Tasks
- T0001–T0012 all done (gen-0 complete)
