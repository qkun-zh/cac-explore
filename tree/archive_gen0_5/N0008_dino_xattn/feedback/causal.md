# feedback/causal.md — N0008_dino_xattn

## reasoning
Causal attribution: NOT the merge idea but its initialization dynamics. Evidence: (1) identical decoder family converged fine on conv features (N0003); (2) divergence appeared immediately when memory became self-supervised-ViT tokens whose scale distribution differs sharply; (3) late-phase recovery was steady and un-collapsed, i.e., the landscape has a good basin nearby. Prediction: stable-lr retry lands between N0007 (27.65) and the true merged potential (<25.5); if it again fails to beat 27.65, the cross-attn value does not transfer to ViT-token grids and H0015 should be refuted properly.

## actionable_feedback
- N0009 = same architecture, lr 2.5e-4, layers 1, K 4, epochs 30. Decision rule pre-registered: <=25.5 supports H0015; >27.65 refutes transfer.

## hypothesis_updates
- H0015: contradicts, strength 0.50 (pending stable-retry adjudication).
