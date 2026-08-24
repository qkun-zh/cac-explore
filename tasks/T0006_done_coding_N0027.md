# T0006_coding_N0027_norm_flip_swa

- status: done               # pending -> claimed_<agent> -> done | cancelled
- finished: 2026-08-24T10:15:00+0800
- created: 2026-08-24T09:57:00+0800
- role: coding
- node: tree/nodes/N0027_norm_flip_swa
- inputs: tree/nodes/N0027_norm_flip_swa/idea.md + memory/failure_modes.md + tree/nodes/N0021_dino_partialft/{model.py,config.py} (champion — copy base) + code/engine/train.py + code/data/fsc147.py
- outputs: tree/nodes/N0027_norm_flip_swa/{model.py,config.py} (champion copy + changes A/B) + engine patch (augment passthrough, SWA-lite, optional dual-res eval) + smoke green on server + flip card done
- notes: SHARED-code discipline: engine patch must be backward compatible, ALL new behavior behind cfg flags defaulted OFF. Do NOT touch code/data/fsc147.py or champion node files. Real 40ep run belongs to Executor/Lead — you stop at green smoke. Nothing committed/pushed.

SPEC:
1. model.py (node dir, copy champion): add registered buffers IMAGENET mean=(0.485,0.456,0.406),
   std=(0.229,0.224,0.225); at top of forward: imgs = (imgs - mean)/std (imgs arrive /255 from
   fsc147.py:68). NOTHING else changes (freeze mask, param_groups, head, prompts intact).
2. config.py (node dir, copy champion cfg): add augment=True, swa_start=14, swa_end=28,
   dual_res_eval=True, dual_res_size=448. All other keys byte-identical to champion.
3. Engine patch (code/engine/train.py, defaults preserve old nodes exactly):
   a) make_loaders: train DS gets augment=bool(cfg.get("augment", False)).
   b) SWA-lite: if cfg swa_start/swa_end set — at epoch ends in [start,end] accumulate CPU
      clones of every p.data with requires_grad=True; after loop: uniform-average, save
      swa.pth {epoch_window, model: averaged_state}, load into model, evaluate() ONCE, add
      {"swa_mae","swa_rmse"} to final write_result. Clamp window to actual epochs run
      (truncated runs: average over available range, record "truncated": true).
   c) OPTIONAL dual-res eval (skip if total engine diff would exceed ~15 LOC): if
      cfg["dual_res_eval"], build one extra val loader @dual_res_size before loop; at each
      epoch end evaluate() it and log mae448/rmse448 in the running write_result + print.
      evaluate() signature UNCHANGED. Inference-only; if OOM risk, bs4 for the 448 loader.
   d) result.json headline stays last-epoch (comparable across nodes); best_mae already
      carried — synthesis reads best_mae/swa_mae. No headline surgery in this patch.
4. Smoke FIRST on server (ssh cac-server), env HF_HOME=/data/asset/hf
   HF_ENDPOINT=https://hf-mirror.com, /data/miniconda/envs/cac/bin/python:
   code/engine/train.py --node_dir tree/nodes/N0027_norm_flip_swa --smoke
   Verify: normalized forward finite; SWA clamps to short smoke schedule and writes swa.pth +
   swa_mae; dual-res loader builds; regression: rerun ONE prior-node smoke (e.g. champion
   config) proving flags-off path unchanged. Then one REAL-DATA val-batch sanity (finite dens
   sums, flipped bbox covers object — assert sign S-x2,S-x1 once). Long steps go in tmux with
   completion marker; never blocking loops.
5. Report back: unified diff summary (engine), node files written, exact smoke output lines,
   confirmation nothing committed. Claims will be verified against filesystem (hallucination rule).

EVIDENCE (coding agent, 2026-08-24):
- Node files: model.py (champion copy + ImageNet buffers in_mean/in_std @ forward top; nothing else), config.py (champion byte-identical + augment/swa_start=14/swa_end=28/dual_res_eval/dual_res_size=448). NOTE: dropout stays 0.1 per champion cfg/idea.md (prompt said 0.15 — cfg value wins).
- Engine diff (+36/-3): L137 augment passthrough (default False); SWA-lite (~19 LOC: CPU accumulator of trainable params E14-28, uniform avg → swa.pth {epoch_window,truncated,model} → load_state_dict(strict=False) → one evaluate() → diag swa_mae/swa_rmse/swa_epoch_window/swa_truncated); dual-res rider 10 LOC (val@448 bs4, final-epoch eval only, diag mae448/rmse448); headline untouched. fsc147.py + champion node untouched.
- Smoke N0027 --smoke --epochs 16 (server): `[normdbg] post-norm range=[-2.118,2.640]` (debug print verified then removed+resynced, md5 2a65ea3e) · `params_M: 23.11` · `[engine] swa window=[14,16] MAE=2.012 RMSE=2.529` · diagnostics `"swa_mae":2.0121,"swa_truncated":true` · swa.pth written (18.4MB) · status=success.
- Regression N0021 --smoke --epochs 2 (flags-off): status=success, no swa/dual lines, params=23.11M — identical to N0027 ⇒ norm adds zero params (STATE's 23.26M was a different accounting).
- Real-data sanity (temp script, deleted after): val sums finite>0 · flip@i sign_ok=True count_preserved=True inside=True · 448 loader fwd grid [2,1,32,32] finite.
- Nothing committed/pushed locally or on server; real 40ep NOT launched (Executor's).
