# Title: Tiny Exemplar Density Baseline (smoke test)

## Motivation & Intuition
Minimal node for validating the full pipeline (contract, tmux, collection, commit). A plain convolutional density head; accuracy is not a goal.

## Architecture Spec
- core_ideas: 4-layer conv downsampling to 1/8 resolution; single-channel density output; softplus for non-negativity
- core_blocks: Conv3x3-BN-ReLU ×2 → Conv3x3-s1 → 1x1 output
- network_structure: input [B,3,S,S] + bbox (exemplar content ignored in this node, placeholder only) → density [B,1,S/8,S/8]
- tunable_aspects: width (16), depth
- invariants: <0.5M params; CPU-runnable; no external weights

## Proposed Hypotheses
- IF softplus instead of relu is used as the density output IN tiny density networks, THEN training is smoother with no dead zone, BECAUSE zero-gradient dead units have proportionally larger impact on very small networks. DISPROVED IF smoke loss goes NaN or fails to decrease.

## Delta vs Parent
Root node (no parent). Result is for pipeline acceptance only; not scored in the trajectory tree.

## Novelty Statement
Zero novelty — intentionally conservative.
