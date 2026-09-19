# N0001_champion — feedback (migration notes)

- `feedback/quant.md`: original val MAE 19.647 / RMSE 74.05, 31.32M params,
  32 epochs. Historic run predates seeding — reproducible artifact is a fresh
  rerun with config.seed=20260830.
- `feedback/causal.md`: interface contract proven load-bearing: frozen hs(2,3)
  intermediate readout + exemplar embedding; unfreeze/readout-final hurt.
- Operational: on cac-server, conda env `cac`, HF cache /data/asset/hf,
  HF_ENDPOINT=hf-mirror.com; data at /data/dataset/FSC147 (VarV2 protocol).
- GPU note: the original deep-epoch probes nearest winner; keep 32ep minimum,
  early-stop bar = same-epoch ≥+1.5 worse than parent best at ep16+.