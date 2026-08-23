# failure_modes — implementation pitfalls & practices (l_t(h))

> Mandatory reading before Coding/ops work. The Diagnostic step appends after every incident.
> Format: `- <failure mode>: <practice> (source)`

## Environment & Dependencies

- Renaming a conda env via mv+sed over bin/: ELF binaries with embedded path strings get byte-misaligned by sed and segfault. Renaming = create a new env and reinstall. (setup)
- Installing torch without pinning: local driver is CUDA 12.4; latest PyPI torch (2.13/cu130) is major-incompatible. Pin `torch==2.10.0` (PyPI default wheel IS the cu128 build — verifiable in metadata requires_dist). (setup)
- pip without `-i` Tsinghua mirror: nvidia deps fall back to direct pypi.org at 75KB/s and stall. Always add `-i https://pypi.tuna.tsinghua.edu.cn/simple`. (setup)
- Old pip + pysocks proxy raises urllib3 PoolKey TypeError: don't fix pip; download wheels directly with `curl --socks5-hostname`. (setup)

## File Operations

- pkill/pgrep -f pattern matches its own command line: the remote shell kills itself and the whole command silently vanishes (root cause of two "no output" incidents). Use `[u]nzip`-style bracket patterns or exact process names. (setup)
- Deleting/moving while a writer is active: rm'd pip's in-flight /tmp/pip-unpack-* wheels (killed an install); mv raced a running unzip producing a mangled layout. Always pgrep for active writers before any cleanup. (setup)
- Long inline SSH commands that drop mid-flight half-execute silently: anything >30s goes in tmux with a completion marker file (e.g. `.EXTRACT_OK`); verify by marker + counts, never by "no error seen". (setup)
- Multi-line Python nested inside ssh single quotes hangs on quoting: put such scripts in the repo (`scripts/*.py`) instead of inline. (setup)

## Git / Accounts

- GitHub push rejected on email privacy: commits must use `qkun-zh@users.noreply.github.com`. (setup)

## Engine / Contracts

- Model density resolution vs GT mismatch: models may emit low-res density (standard for counting); the engine upsamples to GT size with sum conservation, so node code must NOT assume shapes match. Evaluation always uses density sums. (S0001)
- FSC147 JSON ids carry `.jpg`; exemplar boxes are in original-image coordinates scaled by annotation W/H — not by the on-disk image size. (S0001)
- HF/timm ViTs are registered at a fixed img_size (DINOv2 = 518): other inputs assert-fail in patch_embed unless `timm.create_model(..., dynamic_img_size=True)`. (N0002)
- Fresh server env lacks timm: install with `pip -i https://pypi.tuna.tsinghua.edu.cn/simple timm` before any pretrained node. (N0002)
- Server→GitHub direct is flaky: run local `scripts/revproxy.py` (reverse SOCKS on remote :1081) and set `git -C /data/repo config http.proxy socks5h://127.0.0.1:1081`. Requires the local HTTP proxy (172.18.80.1:57777) alive; check PID first. (N0002)
- Shell-launch trap: in `a && b; tmux ...` the tmux part still runs when `a` fails — a stale-repo smoke can launch without env exports. Chain everything with `&&`, put exports INSIDE the tmux command string. (N0002)

## Training Dynamics

- Frozen-backbone heads converge slowly at 10 epochs (N0002 MAE still descending at ep10, 317s of 1800s used): budget ≥20 epochs or accept under-converged baselines. (N0002)
- select_next.py acc_norm inverted: best_metric is MAE (lower=better) but (val-lo)/(hi-lo) ranked the WORST node first. Fixed to (hi-val)/(hi-lo). Always sanity-check selection output against known best node. (gen-0)
