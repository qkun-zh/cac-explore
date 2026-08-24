#!/usr/bin/env python3
"""scripts/eval_readout_lab.py — N0025 eval-only readout lab on the frozen N0021 champion.

--dump: one val pass @392 bs8 AMP. Champion architecture is imported from
tree/nodes/N0021_dino_partialft (single source of truth) and weights come from
/data/runs/N0021_dino_partialft/best.pth. Writes per-image JSONL rows:
  {img_id, N_gt, N_hat_raw, N_hat_ttnorm, N_hat_trimmed_05/10/20,
   box_gains[3], box_integrals[3], g_applied, g_skipped}
All 3 exemplar boxes are read directly from annotation_FSC147_384.json
(box_examples_coordinates) and rescaled exactly like code/data/fsc147.py
(sx=S/ann_W, sy=S/ann_H). Box integral = sum of rho over cells whose centers fall
inside the box (grid 28x28 at S=392). TT-Norm: skipped iff every integral < 1e-6,
else g = clip(median_k 1/max(integral_k,1e-6), 0.2, 5), N' = g * N_raw.
Trimmed arms drop the top {0.5%,1%,2%} hottest cells before summing.

--analyze DUMP.jsonl: OFFLINE CPU-only (no torch). Prints overall MAE/RMSE table,
count-stratified bucket breakdown with SSE shares, split-half cross-fitted isotonic
(IsotonicRegression(out_of_bounds="clip"), identity-regularized outside p10-p90 of
the fit half, BOTH directions) and explicit H0033/H0034 verdicts. Verdicts use
cross-fit numbers only; full-val refit for test deployment is gated on split-half
stability and reported separately.
"""
import argparse
import importlib.util
import json
import math
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE21 = os.path.join(REPO, "tree", "nodes", "N0021_dino_partialft")
DEFAULT_CKPT = "/data/runs/N0021_dino_partialft/best.pth"
DEFAULT_DATA_ROOT = "/data/dataset/FSC147"
TRIM_FRACS = (("N_hat_trimmed_05", 0.005), ("N_hat_trimmed_10", 0.01), ("N_hat_trimmed_20", 0.02))
ROW_KEYS = ["img_id", "N_gt", "N_hat_raw", "N_hat_ttnorm",
            "N_hat_trimmed_05", "N_hat_trimmed_10", "N_hat_trimmed_20",
            "box_gains", "box_integrals"]
ARMS = (("raw", "N_hat_raw"), ("ttnorm", "N_hat_ttnorm"),
        ("trim0.5%", "N_hat_trimmed_05"), ("trim1%", "N_hat_trimmed_10"),
        ("trim2%", "N_hat_trimmed_20"))
BUCKETS = ((0.0, 25.0), (25.0, 75.0), (75.0, 200.0), (200.0, 500.0), (500.0, math.inf))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def boxes_in_S_space(ann, S):
    sx, sy = S / float(ann["W"]), S / float(ann["H"])
    out = []
    for corners in ann["box_examples_coordinates"][:3]:
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        out.append((min(xs) * sx, min(ys) * sy, max(xs) * sx, max(ys) * sy))
    return out


def box_integrals(rho_flat, G, S, boxes):
    import torch
    ps = S / G
    centers = (torch.arange(G, dtype=torch.float64) + 0.5) * ps
    cy = centers.view(G, 1).expand(G, G).reshape(-1)
    cx = centers.view(1, G).expand(G, G).reshape(-1)
    rho = rho_flat.double()
    vals = []
    for (x0, y0, x1, y1) in boxes:
        m = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
        vals.append(float((rho * m).sum()))
    return vals


def tt_gain(integrals):
    if all(i < 1e-6 for i in integrals):
        return None
    return max(0.2, min(5.0, statistics.median(1.0 / max(i, 1e-6) for i in integrals)))


def trimmed_sum(rho_flat, frac):
    import torch
    k = int(rho_flat.numel() * frac)
    if k <= 0:
        return float(rho_flat.sum())
    return float(torch.sort(rho_flat, descending=True).values[k:].sum())


def cmd_dump(args):
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, os.path.join(REPO, "code"))
    from data.fsc147 import FSC147Density, collate_density

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_mod = load_module(os.path.join(NODE21, "config.py"), "n0025_champ_cfg")
    model_mod = load_module(os.path.join(NODE21, "model.py"), "n0025_champ_model")
    cfg = dict(getattr(cfg_mod, "cfg"))
    model = model_mod.build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    meta = {k: ckpt[k] for k in ("epoch", "best_mae") if isinstance(ckpt, dict) and k in ckpt}
    print(f"[lab] loaded {args.ckpt} meta={meta} device={device}", flush=True)

    root = args.data_root or cfg.get("data_root", DEFAULT_DATA_ROOT)
    S = int(cfg.get("input_size", 392))
    bs = int(cfg.get("batch_size", 8))
    va = FSC147Density(root, S, "val")
    loader = DataLoader(va, batch_size=bs, shuffle=False, num_workers=int(cfg.get("num_workers", 4)),
                        collate_fn=collate_density, pin_memory=True)
    with open(os.path.join(root, "annotation_FSC147_384.json")) as f:
        anno = json.load(f)

    n_total = len(va) if args.limit is None else min(args.limit, len(va))
    print(f"[lab] val imgs={len(va)} running_on={n_total} S={S} bs={bs} amp={args.amp and device.type == 'cuda'}",
          flush=True)

    use_amp = args.amp and device.type == "cuda"
    idx = 0
    sse_raw = sse_tt = 0.0
    abs_err_raw = 0.0
    with open(args.out, "w") as fout, torch.no_grad():
        for bi, batch in enumerate(loader):
            if idx >= n_total:
                break
            imgs = batch["imgs"].to(device)
            bboxes = batch["bboxes"].to(device)
            counts = batch["counts"].float()
            with torch.autocast("cuda", enabled=use_amp):
                out = model(imgs, bboxes)
            dens = out["density"].float().cpu()
            B, _, G, _ = dens.shape
            for j in range(B):
                if idx >= n_total:
                    break
                img_id = va.ids[idx]
                rho = dens[j, 0].flatten()
                raw = float(rho.sum())
                integrals = box_integrals(rho, G, S, boxes_in_S_space(anno[img_id], S))
                g = tt_gain(integrals)
                ttnorm = raw if g is None else g * raw
                row = {
                    "img_id": img_id,
                    "N_gt": float(counts[j]),
                    "N_hat_raw": raw,
                    "N_hat_ttnorm": ttnorm,
                    "box_gains": [1.0 / max(i, 1e-6) for i in integrals],
                    "box_integrals": integrals,
                    "g_applied": g,
                    "g_skipped": g is None,
                }
                for key, frac in TRIM_FRACS:
                    row[key] = trimmed_sum(rho, frac)
                fout.write(json.dumps(row) + "\n")
                abs_err_raw += abs(raw - row["N_gt"])
                sse_raw += (raw - row["N_gt"]) ** 2
                sse_tt += (ttnorm - row["N_gt"]) ** 2
                idx += 1
            if (bi + 1) % 10 == 0:
                print(f"[dump] batch {bi + 1}/{math.ceil(n_total / bs)} rows={idx} "
                      f"runMAE_raw={abs_err_raw / idx:.3f}", flush=True)

    mae_raw = abs_err_raw / max(idx, 1)
    rmse_raw = math.sqrt(sse_raw / max(idx, 1))
    print(f"[dump] DONE rows={idx} MAE_raw={mae_raw:.4f} RMSE_raw={rmse_raw:.4f} SSE_tt={sse_tt:.1f}",
          flush=True)

    if args.smoke:
        validate_dump(args.out, n_total)
    print(f"[lab] dump written to {args.out}")
    return args.out


def validate_dump(path, expected_rows):
    seen = 0
    nan_fields = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            missing = [k for k in ROW_KEYS if k not in r]
            assert not missing, f"row {seen} missing keys {missing}"
            assert len(r["box_integrals"]) == 3 and len(r["box_gains"]) == 3, f"row {seen} box arrays"
            for k in ROW_KEYS[1:]:
                v = r[k]
                vals = v if isinstance(v, list) else [v]
                nan_fields += sum(1 for x in vals if isinstance(x, float) and not math.isfinite(x))
            recomputed = tt_gain(r["box_integrals"])
            expect = r["N_hat_raw"] if recomputed is None else recomputed * r["N_hat_raw"]
            assert abs(expect - r["N_hat_ttnorm"]) < 1e-3 * max(1.0, abs(expect)), f"row {seen} ttnorm mismatch"
            seen += 1
    assert seen == expected_rows, f"rows {seen} != expected {expected_rows}"
    print(f"SMOKE_OK rows={seen} schema=OK nonfinite={nan_fields} "
          f"tt_skipped={sum(1 for l in open(path) if json.loads(l)['g_skipped'])}")
    assert nan_fields == 0, "non-finite values in dump"


def cmd_analyze(args):
    import numpy as np
    from sklearn.isotonic import IsotonicRegression

    with open(args.analyze) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    n = len(rows)
    gt = np.array([r["N_gt"] for r in rows], dtype=float)

    def mae(rs, key):
        return sum(abs(r[key] - r["N_gt"]) for r in rs) / len(rs)

    def rmse(rs, key):
        return math.sqrt(sum((r[key] - r["N_gt"]) ** 2 for r in rs) / len(rs))

    def sse(rs, key):
        return sum((r[key] - r["N_gt"]) ** 2 for r in rs)

    print("=" * 72)
    print(f"N0025 READOUT LAB ANALYSIS  rows={n}  (offline CPU, leak-safe)")
    print("=" * 72)
    print("\n-- Overall --")
    print(f"{'arm':<10} {'MAE':>9} {'RMSE':>10} {'dMAE vs raw':>12}")
    base_mae = mae(rows, "N_hat_raw")
    for name, key in ARMS:
        print(f"{name:<10} {mae(rows, key):>9.3f} {rmse(rows, key):>10.3f} "
              f"{mae(rows, key) - base_mae:>+12.3f}")

    print("\n-- Count-stratified MAE by N_gt bucket --")
    hdr = "bucket".ljust(14) + "".join(f"{name:>10}" for name, _ in ARMS) + f"{'n':>7}"
    print(hdr)
    bucket_rows = {}
    for lo, hi in BUCKETS:
        sub = [r for r in rows if lo <= r["N_gt"] < hi]
        bucket_rows[(lo, hi)] = sub
        line = f"[{lo:g},{hi:g})".ljust(14) + "".join(f"{mae(sub, k):>10.3f}" for _, k in ARMS) + f"{len(sub):>7}"
        print(line)

    total_sse = {name: sse(rows, key) for name, key in ARMS}
    print("\n-- Count-stratified SSE share (%) by N_gt bucket --")
    print("bucket".ljust(14) + "".join(f"{name:>10}" for name, _ in ARMS) + f"{'n':>7}")
    for (lo, hi), sub in bucket_rows.items():
        shares = "".join(f"{100.0 * sse(sub, k) / max(total_sse[name], 1e-9):>10.2f}" for name, k in ARMS)
        print(f"[{lo:g},{hi:g})".ljust(14) + shares + f"{len(sub):>7}")

    order = sorted(range(n), key=lambda i: rows[i]["img_id"])
    halfA = [rows[i] for i in order[0::2]]
    halfB = [rows[i] for i in order[1::2]]

    def fit_regularized(rs, key):
        X = np.array([r[key] for r in rs], dtype=float)
        Y = np.array([r["N_gt"] for r in rs], dtype=float)
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(X, Y)
        lo, hi = float(np.percentile(X, 10)), float(np.percentile(X, 90))
        band = (X >= lo) & (X <= hi)
        rel_dev = float(np.mean(np.abs(iso.predict(X[band]) - X[band])) / max(float(np.mean(X[band])), 1e-9))

        def fn(x):
            xv = float(x)
            return float(iso.predict([xv])[0]) if lo <= xv <= hi else xv

        return fn, lo, hi, rel_dev

    def crossfit(key):
        res = {}
        for tag, fit_half, eval_half in (("A->B", halfA, halfB), ("B->A", halfB, halfA)):
            fn, lo, hi, rel_dev = fit_regularized(fit_half, key)
            pre = mae(eval_half, key)
            post = sum(abs(fn(r[key]) - r["N_gt"]) for r in eval_half) / len(eval_half)
            res[tag] = (pre, post, pre - post, rel_dev, (lo, hi))
        grid = sorted({round(x, 4) for tag in res for x in np.percentile(
            [r[key] for r in (halfA if tag == "A->B" else halfB)], np.linspace(10, 90, 17))})
        both = [x for x in grid if res["A->B"][4][0] <= x <= res["A->B"][4][1]
                and res["B->A"][4][0] <= x <= res["B->A"][4][1]]
        return res, both

    print("\n-- Split-half isotonic (cross-fit only; identity outside p10-p90 of fit half) --")
    arm_res = {}
    for name, key in ARMS[:2]:
        res, overlap_grid = crossfit(key)
        arm_res[name] = res
        for tag in ("A->B", "B->A"):
            pre, post, delta, rel_dev, band = res[tag]
            print(f"{name:<8} fit{tag}: pre={pre:.3f} post={post:.3f} improvement={delta:+.3f} "
                  f"(fit-half curve rel-dev-from-identity={rel_dev:.4f}, band=[{band[0]:.1f},{band[1]:.1f}])")

    raw_res = arm_res["raw"]
    imp_AB = raw_res["A->B"][2]
    imp_BA = raw_res["B->A"][2]
    id_AB = raw_res["A->B"][3] <= 0.05
    id_BA = raw_res["B->A"][3] <= 0.05

    gains = [tt_gain(r["box_integrals"]) for r in rows]
    active = [(r, g) for r, g in zip(rows, gains) if g is not None]
    skip_frac = 1.0 - len(active) / n
    near_ident = sum(1 for _, g in active if abs(g - 1.0) < 0.01) / max(len(active), 1)
    mae_tt = mae(rows, "N_hat_ttnorm")

    print("\n-- TT-Norm gain stats --")
    print(f"images={n} skipped(all integrals<1e-6)={int(round(skip_frac * n))} ({100 * skip_frac:.1f}%) "
          f"|g-1|<0.01 among active={int(round(near_ident * len(active)))}/{len(active)} ({100 * near_ident:.1f}%) "
          f"median(g)={statistics.median(g for _, g in active):.4f}" if active else
          f"images={n} skipped={n} (all integrals < 1e-6)")

    print("\n-- H0033 verdict (TT-Norm per-image exemplar gain) --")
    print(f"MAE raw={base_mae:.3f}  MAE ttnorm={mae_tt:.3f}  delta={mae_tt - base_mae:+.3f}  "
          f"(pass bar <=19.5; fail bar >20.64; inert-gain fail bar >50% |g-1|<0.01)")
    if mae_tt <= 19.5:
        v33 = "PASS"
    elif mae_tt > 20.64 or near_ident > 0.5:
        v33 = "FAIL"
    else:
        v33 = "INCONCLUSIVE"
    why33 = (f"ttnorm MAE {mae_tt:.3f} vs bars [19.5 pass / 20.64 fail]; "
             f"inert-gain share {100 * near_ident:.1f}% vs 50% bar; skip share {100 * skip_frac:.1f}%")
    print(f"VERDICT H0033: {v33} — {why33}")

    print("\n-- H0034 verdict (split-half isotonic recalibration, raw readout) --")
    m_imp = min(imp_AB, imp_BA)
    identity_like = id_AB or id_BA
    print(f"improvement A-fit->B={imp_AB:+.3f}  B-fit->A={imp_BA:+.3f}  min={m_imp:+.3f}  "
          f"(pass bar >=0.6; fail bar <0.3; identity-fail: curve within +/-5% of identity)")
    if m_imp >= 0.6 and not identity_like:
        v34 = "PASS"
    elif m_imp < 0.3 or identity_like:
        v34 = "FAIL"
    else:
        v34 = "INCONCLUSIVE"
    why34 = (f"min held-out improvement {m_imp:+.3f} vs bars [<0.3 fail / >=0.6 pass]; "
             f"curve~identity(A,B)=({id_AB},{id_BA})")
    print(f"VERDICT H0034: {v34} — {why34}")

    print("\n-- Deployment gate (full-val isotonic refit onto test) --")
    gate_stable = imp_AB >= 0.3 and imp_BA >= 0.3
    print(f"stability: both directions improve >=0.3 -> {gate_stable} "
          f"(A->B {imp_AB:+.3f}, B->A {imp_BA:+.3f}); "
          f"fitted curves ~identity: A={raw_res['A->B'][3]:.4f}, B={raw_res['B->A'][3]:.4f}")
    if gate_stable and not identity_like:
        decision = "SANCTIONED: refit isotonic on FULL val for test deployment"
    else:
        decision = "BLOCKED: ship raw counts on test"
    print(decision)
    print("NOTE: verdicts above use split-half CROSS-FIT numbers only; val labels never touch test "
          "beyond this sanctioned map.")
    print("=" * 72)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dump", action="store_true")
    p.add_argument("--analyze", metavar="DUMP.jsonl")
    p.add_argument("--smoke", action="store_true", help="limit to ~20 imgs + validate JSONL schema/no-NaN")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=os.path.join(REPO, "tree/nodes/N0025_eval_readout/dump_val.jsonl"))
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--data-root", default=None)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    args = p.parse_args()
    if args.smoke and args.limit is None:
        args.limit = 20
    if args.dump:
        cmd_dump(args)
    elif args.analyze:
        cmd_analyze(args)
    else:
        p.error("choose --dump or --analyze DUMP.jsonl")


if __name__ == "__main__":
    main()
