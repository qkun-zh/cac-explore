# N0001_champion — synthesis

The seed root's synthesis consolidates the migrated ablation history from
`cac_explore` (each finding cross-referenced to its originating node). This is
the authoritative reading of what is settled; `memory/hypotheses.jsonl` carries
the same facts as scored evidence and is what the selection machinery consumes.

## Settled (do not re-litigate)
- **GCA (global-count aux) is positive** (~1.6 over no-GCA; N0051).
  Kept in champion. H0001 supports 1.0 + 0.8.
- **XScale coarse multi-scale exemplar summary is positive** (+0.95 over
  no-XScale; N0054). Kept. H0002 supports 1.0 + 0.8.
- **Frozen intermediate readout hs(2,3) is the load-bearing interface**
  (+~7 vs final-layer readout; N0054 vs N0068 2x2). H0003 supports 1.0,1.0,1.0
  (main), 0.5. Comparable position to the increasing-depth prior.
- **Cross-attention condenser is load-bearing** (N0057/58/59 all negative on
  swaps; held). H0004 supports 1.0 + 0.8.
- **Backbone unfreeze is net-harmful** (+6..+9; N0066/67). H0005 contradicts
  1.0,1.0,1.0,0.8 — refuted.
- **DDCA (dilated context branch) is negative** (+1.8; N0052/53). H0006
  contradicts 1.0 + 0.8.
- **Final-layer readout (vs intermediate) is negative** (+~7; N0068). H0007
  contradicts 1.0,1.0,0.8.
- **Extra spatial summaries / regional aux are negative** (key/fine/max +1.2..+3.2,
  RGA +0.85; N0055/56/60, N0053). H0008 contradicts 1.0 + 0.8.

## Open (the actual evolutionary frontier)
- Multiresolution (392/518) — never empirically proven; config marked off.
- SWA, tail_reweight, dual_res_eval — recorded as untested hypotheses, not
  shipped switches.
- Anything not in baseline/config.toml: not yet explored. New hypotheses may
  attach ONLY above frozen features + exemplar embedding (pluggable rule).

## Calibration caveat
Seed confidence derived from migration weights, not from our calibration loop;
first dynamic epochs will populate the bin table.