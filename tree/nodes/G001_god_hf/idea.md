# Idea — G001_god_hf (HF DINOv3 + GOD OT, 576 piles token-direct, prompt-A, fully frozen backbone)
Parent: none (zero-base, GOD lineage). Implements `docs/inspiration_from_GOD.md` faithfully: sand piles S={(w_j,p_j)}_{j=1..M}, pits G={g_i}_{i=1..N} capacity 1, Lot = αΣπ||p-g||² + βΣmax(0,1-R)² + γΣs², Lrep = λΣ w_j w_k exp(-||p-p||²/2σ²). M=576 token-direct (384/16=24×24). Prompt-A gated mass for CAC.

## Change vs champion (N0021/N0027)
- Backbone: `facebook/dinov3-vits16-pretrain-lvd1689m` via `AutoModel.from_pretrained` + `AutoImageProcessor(size=384)` (HF stack, `load_dataset("isentropic/FSC147")` compliant; local mirror `/data/dataset/FSC147` same content for speed)
- Per-token pile head: `w_j = softplus(MLP_w(T_j)) * gate`, `p_j = grid_center + tanh(MLP_p(T_j))*8`, `gate=sigmoid(α_sim·cos(T,proto)+β_sim)`, proto = RoI mean of 3 exemplar boxes on token grid. No extra decoder, step-0 grid ≡ density baseline.
- Loss: **only GOD** (no MSE). Transport via single-step entropy+dustbin soft assignment (1-iter Sinkhorn micro-adjust for 576×3000 efficiency): `logits=[-d/ε, 0_dustbin] → softmax → π=w·P, s=w·P_dustbin, R=Σπ`. Lot + Lrep end-to-end differentiable, log-domain stable.
- Optim: `transformers AdamW` (HF), backbone frozen (killer stability), only prompt+heads train.

## Hypotheses
H_GOD_A: IF per-token GOD-OT replaces MSE IN FSC147 THEN val MAE ≤19.9 BECAUSE transported-mass readout Σ(w-s) anchors per-object mass (fixes exemplar-box 0.03–0.19) and repulsion prevents center collapse. DISPROVED IF MAE ≥20.44 or loss NaN.

Micro-adjusts vs doc for engineering: (1) single-step dustbin softmax vs full Sinkhorn (batch 576×3000 tractable, retains ε), (2) coordinates /S归一, σ=median_box/S, (3) L2 unbalanced kept, ε=0.05.

Params: DINOv3-S 21.6M frozen + heads+prompt ~0.5M → ~22.1M ≤32M.
Engine: `scripts/train_god_hf.py` (HF processor/dataset path, not old engine).
