# N0065_scale_embed — Champion + BMNet+ scale embedding, 128 epochs

**Parent**: N0054_xscale_exemplar (19.647)
**Hypothesis H0092**: scale-aware exemplar embedding via relative bbox size improves cross-scale invariance.

**Design**: Keep GCA+XScale head unchanged, add learned scale embedding (20 bins, BMNet+ formula: rel_scale = w/img_w*0.5 + h/img_h*0.5, quantized 0.5/scale_bins) to ExemplarEncoder output. 20*256=5120 params, total 31.33M.

**Why**: BMNet+ val 15.74 uses scale embedding; smallest <32M SOTA with code. Port least-invasive idea to frozen head.

**Kill**: <19.647 CONFIRM, >=19.647 or nan/divergence DISPROVED.

**Deviation**: User-Guided per AGENTS §1, Lead-direct (subagent unavailable), 128ep + 36h resume.

**Result**: 81/128ep best 20.429@E28 (+0.78), 10 nan @E64-73 →62 MAE collapse. H0092 refuted.
