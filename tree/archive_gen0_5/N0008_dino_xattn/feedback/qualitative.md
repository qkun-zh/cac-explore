# feedback/qualitative.md — N0008_dino_xattn

## reasoning
Code reading: basis maps are dots of query-emitted vectors with L2-NORMALIZED tokens (cos-like), softmax-mixed. At init, basis≈0 so density≈0 and MSE gradient pushes basis up from zero — but mem_proj+attention outputs drift fast under lr=1e-3, so queries chase a moving memory representation: the oscillation pattern matches exactly. N0007's adapter head had no such moving-target problem because tokens enter the head directly without a jointly-trained projection being consumed by attention.

## actionable_feedback
- Lower lr AND freeze mem_proj for first phase would decouple the moving target; within contract, lr cut alone should suffice.
- Consider stop-gradient on memory for epoch 1 (model-side trick) if instability persists.

## hypothesis_updates
- H0015: contradicts, strength 0.45 (qualitative concurs with optimization-confound reading).
