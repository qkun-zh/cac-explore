# Idea — N0033_salf_moe_frozen (parent: N0026_res_sweep, frozen ConvNeXt-Tiny champion)

Champion: cac_d ConvNeXt-Tiny frozen (384px, MAE 18.33 TEST / 20.4 val) — FineFuser(h2@1/8+h3@1/16)+Condenser+DensityDecoder, 28.74M total.

## Changes vs parent (2 minimal, backbone FROZEN per instruction)
**A. SALF-lite: spatially-adaptive 4-stage fusion (H0049, ~0.8M).** Expose hs1@1/4(96ch), hs2@1/8(192), hs3@1/16(384), hs4@1/32(768) → 1×1+GN→128 each → align to H/8 (96→48 down×2, 24↑×2, 12↑×4 bilinear). Concat 512 → 1×1→128 GN+GELU → 1×1→4 logits per-location → softmax → weighted sum → 128ch fused @48. Replaces FineFuser's fixed cat+conv. Addresses DCA-MoE finding: sparse regions need shallow detail, dense need deep semantics — fixed fusion is spatially invariant.

**B. DR-MoE-lite: density-routed multi-RF experts (H0050, ~0.6M) + balance loss.** After SALF, router 1×1(128→32 GN+GELU →1×1→3) → softmax per location. Experts: E1 3×3 (local), E2 depthwise 3×3 d=2 +1×1 (mid), E3 depthwise 7×7 +1×1 (large). Output = Σ w_i·E_i(x). Add aux L_balance = 0.01·CV²(load), CV=std/mean of router mean weights. Then interpolate ×2 →96 + Condenser(128→64) + Decoder same as champion.

Both γ=1 step-0 not zero — but SALF gate init uniform (0 logits) → step0 ≈ mean fusion; MoE router uniform → step0 ≈ mean experts. Honest delta. Param total est 28.7+1.4=30.1M <32M. Frozen backbone untouched (requires_grad False, eval).

## Hypotheses
**H0049** IF SALF per-location 4-stage weighting IN frozen ConvNeXt THEN val best MAE ≤19.3 (parent 20.4 -1.1) BECAUSE spatial adaptation picks shallow for sparse / deep for dense. DISPROVED IF MAE ≥20.1 or gate entropy <0.3 (collapse to one stage).
**H0050** IF MoE 3-RF soft routing + balance THEN tail [500+) MAE drops ≥20% vs parent without bulk [0,50) degradation >0.5 BECAUSE receptive field adapts to density. DISPROVED IF tail drop <10% or expert collapse (one w mean >0.7).

## Gates & protocol
40ep 384px frozen, AdamW 1e-3 cosine, batch16 (12GB), AMP. Early-stop if ep16+ gap +1.5 vs parent. Kill-or-confirm: H0049 via val MAE + gate weight stats; H0050 via bucket MAE + router histogram. Blacklist kept: full-FT, hires-out, norm+flip alone (no repeat).
