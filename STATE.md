# STATE — Session 2026-08-29 (Lead=qkun-local)

**Mode**: Free-Research (autonomous). User directive: combine PoM (2604.06129) + ParTY (2603.09611) pluggable modules autonomously, no questions.
**Preflight**: creds ROTATED earlier today (port 44387, host gxkkqyad0izmmwnlsnow.deepln.com) → install_key.py rerun → SERVER_OK. Server engine has +5-line periodic-ckpt hotfix (save_every, default off) — benign, engine train.py is server source-of-truth.

## Champion (frozen regime, UNCHANGED)
**N0054_xscale_exemplar** (GCA + XScale) val MAE **19.647** / RMSE 74.05 · 31.32M · LOCKED DELIVERABLE. Recipe: AdamW 1e-3, wd0.05, cosine, bs16, AMP, 30ep @384, MSE+SmoothL1. use_gca=True, use_ddca=False, use_xscale=True, xscale_size=3.

## Active node: N0059_pom_morph (RUNNING, 30ep)
**Pre-registered falsifier of H0081.** Replace producer self-attention with 2× PoM-PolyMorpher (2604.06129 eq.3: token-averaged 2nd-order moments H + per-token sigmoid gate, norm_first residual path identical to N0054), 49 tokens KEPT, ParTY part-pool EXCLUDED (it was N0058's confound), capacity matched (D=352: head 3.522M vs N0054 3.505M, total 31.34M). Single switch use_pom; server smoke GREEN (use_pom=False restores exact 31.32M/3.50M).
**H0082** IF param-matched non-attention operator replaces exemplar producer self-attn in frozen N0054 THEN val MAE < 19.647 BECAUSE cross-token order (not self-attention) is load-bearing. DISPROVED IF ≥20.40.
**Gates**: CONFIRM <19.40 (H0081 disproved) · TIE 19.40–19.90 (operator exonerated; N0058 = capacity+pool) · KILL ≥19.90 or ep16+ same-epoch ≥ N0054+1.5. 2nd seed mandatory only if first <19.40.
Launched 09:21 via tmux node_N0059_pom_morph (run_node.sh git-pull hangs server-side → direct tmux).

## Frozen-regime NEGATIVE table (30ep @384)
| Node | Axis | Delta |
|---|---|---|
| N0054 GCA+XScale | — | **19.647 CHAMPION** |
| N0055 XScale-Key | info-add 2K keys | +1.19 |
| N0056 XFine | info-add extra scale | +3.06 |
| N0057 cond-matcher | consumer swap | +1.43 |
| N0058 PMOM | producer swap (part-pool+−42%cap) | +2.17 same-ep / +3.50 floor |

N0059 decides whether operator-swap failure = attention-is-load-bearing (law) or = N0058's capacity+pooling (operator exonerated).

## Server gotchas
- run_node.sh `git pull` hangs on server: launch tmux directly (export PATH=/data/miniconda/bin).
- tmux libtinfo warning non-fatal. python /data/miniconda/envs/cac/bin/python; HF /data/asset/hf + hf-mirror.
- Sync via scp; `local/` gitignored.

## Queue
1. ✅ N0059: idea-finalize+novelty (stage1 0.474→stage2 NOVEL) → register → code+local smoke → scp+server smoke GREEN → launched (09:21).
2. Poll curve vs N0054 same-epoch; early-stop ep16+ ≥+1.5; if TIE band run late-drop E18-27 parity check.
3. Feedback + Diagnostic + synthesis + calibration + H0082/H0081 ledger + tree flip + commit/push.