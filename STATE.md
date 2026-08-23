# STATE — Current Situation

**Stage**: gen-4 RUNNING — champion N0010 21.53 · N0013 augreg tracking ahead of champion trajectory · N0012/N0011 refuted
**Blockers**: none — pipeline full (never-idle rule in AGENTS §3.0)

## Verified Facts (do not re-learn)
- Champion recipe: frozen DINOv2-S reg4 dual taps(6,11) scalar gate + area-prompt + adapter768 + MLP head, 392px, 40ep, count-w1.0 → **21.53** @1275s/23.11M
- N0011 per-token gate+Huber REFUTED 26.68 (+24%); H0019 0.415/H0020 0.42 — MLP head is enough on strong features
- N0012 highres518 EARLY-STOPPED ep18 best 26.03; H0023 contradicts 0.455 w/ timeout+bs confound (retry needs grad-accum+full schedule before final refutation); H0017 0.645
- N0013 augreg tracking AHEAD of champion: E13 best 24.38 vs parent E13 25.50 — photometric+bbox jitter works so far
- Early-stop rule (user): if same-epoch trajectory ≥ +1.5 worse than parent at ep16+, kill to save GPU
- Engine now supports loss_function=huber (unused going forward); 23 hypotheses in bank

## Next Steps (in order)
1. Poll N0013 single-shot; collect when done; feedback×3+synthesis subagents
2. Queue order when GPU frees: N0014 highres+augreg merge → N0016 4-tap multiscale → N0015 672 extreme-res
3. If augreg confirms (<21), gen-5 merges it with longer schedule / count-calibration (H0022)

## Active Tasks
- T0028 pending_executor N0013 (RUNNING) · T0030 N0014 queued · T0032 N0015 queued · T0034 N0016 queued
EOF_MARKER_NOT_NEEDED
