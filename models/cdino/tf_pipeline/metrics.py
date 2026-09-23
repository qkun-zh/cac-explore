"""Extended error metrics, density bins, worst-15 dump."""
import numpy as np


def extended_metrics(predictions, targets):
    p = np.asarray(predictions, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    e = p - t
    ae = np.abs(e)
    ape = ae / (t + 1e-8)
    ext = {
        'n': int(t.size),
        'MAE': float(ae.mean()),
        'RMSE': float(np.sqrt((e ** 2).mean())),
        'MADiff_bias': float(e.mean()),
        'MedAE': float(np.median(ae)),
        'StdAE': float(ae.std()),
        'P50_abs': float(np.percentile(ae, 50)),
        'P75_abs': float(np.percentile(ae, 75)),
        'P90_abs': float(np.percentile(ae, 90)),
        'P95_abs': float(np.percentile(ae, 95)),
        'P99_abs': float(np.percentile(ae, 99)),
        'Max_abs': float(ae.max()),
        'Max_err_signed': float(e[np.argmax(ae)]),
        'n_over': int((e > 0).sum()),
        'n_under': int((e < 0).sum()),
        'n_exact': int((e == 0).sum()),
        'pct_over': float(100.0 * (e > 0).mean()),
        'pct_under': float(100.0 * (e < 0).mean()),
        'MAPE': float(100.0 * ape[t > 0].mean()) if (t > 0).any() else float('nan'),
        'MedAPE': float(100.0 * float(np.median(ape[t > 0]))) if (t > 0).any() else float('nan'),
        'n_gt0': int((t <= 0).sum()),
        'n_absle1': int((ae <= 1).sum()),
        'n_absle5': int((ae <= 5).sum()),
        'n_absle10': int((ae <= 10).sum()),
        'n_err_ge50': int((ae >= 50).sum()),
        'n_err_ge100': int((ae >= 100).sum()),
        'corr_pt': float(np.corrcoef(t, p)[0, 1]) if t.size > 1 else float('nan'),
        'gt_mean': float(t.mean()),
        'pred_mean': float(p.mean()),
        'gt_sum': float(t.sum()),
        'pred_sum': float(p.sum()),
        'ratio_sum': float(p.sum() / (t.sum() + 1e-8)),
    }
    return p, t, e, ae, ape, ext


def density_bins(t, e, ae, ape, metrics=None):
    bins = [(0, 10), (10, 30), (30, 80), (80, 200), (200, 10 ** 9)]
    bin_stats = []
    for lo, hi in bins:
        m = (t >= lo) & (t < hi)
        if m.sum() == 0:
            continue
        row = {
            'bin': f'{lo}-{hi if hi < 10 ** 9 else "inf"}',
            'n': int(m.sum()),
            'MAE': float(ae[m].mean()),
            'Bias': float(e[m].mean()),
            'MedAE': float(np.median(ae[m])),
            'MAPE': float(100.0 * ape[m & (t > 0)].mean()) if (m & (t > 0)).any() else float('nan'),
        }
        bin_stats.append(row)
        if metrics is not None:
            metrics[f'MAE_bin_{row["bin"]}'] = row['MAE']
            metrics[f'Bias_bin_{row["bin"]}'] = row['Bias']
    return bin_stats


def write_worst15(per_image_path, ids_out, t, p, e, ae, ape):
    order = np.argsort(-ae)[:15]
    worst_path = per_image_path.replace('.csv', '_worst15.csv')
    with open(worst_path, 'w') as wf:
        wf.write('idx,filename,gt,pred,err,abs_err,ape\n')
        for i in order:
            wf.write(
                f'{ids_out[i] if i < len(ids_out) else i},{ids_out[i] if i < len(ids_out) else ""},'
                f'{t[i]:.6f},{p[i]:.6f},{e[i]:.6f},{ae[i]:.6f},{ape[i]:.6f}\n'
            )
            print(f'WORST idx={i} file={ids_out[i]} gt={t[i]:.1f} pred={p[i]:.1f} err={e[i]:.1f}', flush=True)
    print(f'worst15_path={worst_path}', flush=True)
    return worst_path
