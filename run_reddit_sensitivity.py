"""
Hyperparameter sweep for RAPO on Reddit -- same situation as MovieLens's
first pass: default config (carried over from movielens's winning values,
untested on THIS domain) did not reach significance (t=1.06, p=0.295).
This sweep tests whether stronger/differently-tuned deformation helps.
Reuses run_reddit.run_one_seed directly.
"""

import json
import numpy as np
from scipy import stats
import run_reddit as R
from reddit_env import RedditEnv

GRID = [
    dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3, w_G=0.0, w_H=2.5, psi_min=0.20),  # original default
    dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3, w_G=0.0, w_H=4.0, psi_min=0.10),
    dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3, w_G=0.0, w_H=6.0, psi_min=0.05),
    dict(lam=0.10, alpha=1.0, delta=0.97, eta=0.7, tau=0.2, w_G=0.0, w_H=6.0, psi_min=0.05),
    dict(lam=0.20, alpha=1.0, delta=0.93, eta=0.4, tau=0.4, w_G=0.0, w_H=3.5, psi_min=0.15),
]
N_SEEDS = 30  # sweep pass -- lower N than the final confirmation run, same as movielens


def main():
    print("Building environment...", flush=True)
    env = RedditEnv()
    print(flush=True)

    base_rows = [R.run_one_seed(seed, "baseline", env) for seed in range(N_SEEDS)]
    base_auc = np.array([r["AUC_R"] for r in base_rows])
    base_rep = np.array([r["rep_harm_total"] for r in base_rows])
    print(f"Baseline AUC-R: {np.nanmean(base_auc):.3f} +/- {np.nanstd(base_auc,ddof=1):.3f}")
    print(f"Baseline absolute replay harm: {base_rep.mean():.3f} +/- {base_rep.std(ddof=1):.3f}\n", flush=True)

    summary = []
    for i, cfg in enumerate(GRID):
        R.CONDITION_KWARGS["rapo"] = cfg
        rows = [R.run_one_seed(seed, "rapo", env) for seed in range(N_SEEDS)]
        rapo_auc = np.array([r["AUC_R"] for r in rows])
        rapo_rep = np.array([r["rep_harm_total"] for r in rows])

        valid = ~(np.isnan(base_auc) | np.isnan(rapo_auc))
        b, r_ = base_auc[valid], rapo_auc[valid]
        reduction_pct = (b.mean() - r_.mean()) / b.mean() * 100
        t = stats.ttest_rel(b, r_)

        # absolute replay-harm comparison -- the metric that actually
        # matters here, see diagnose_reddit_backfire.py: AUC-R's ratio
        # structure can look flat/worse even when absolute replay harm is
        # genuinely lower, because RAPO also suppresses its own
        # exposure-phase harm (shrinking AUC-R's denominator too)
        abs_reduction_pct = (base_rep.mean() - rapo_rep.mean()) / base_rep.mean() * 100
        t_abs = stats.ttest_rel(base_rep, rapo_rep)

        entry = dict(cfg=cfg, rapo_AUC_R_mean=float(rapo_auc.mean()),
                     reduction_pct=float(reduction_pct),
                     t_stat=float(t.statistic), p_value=float(t.pvalue),
                     rapo_rep_harm_mean=float(rapo_rep.mean()),
                     abs_reduction_pct=float(abs_reduction_pct),
                     t_stat_abs=float(t_abs.statistic), p_value_abs=float(t_abs.pvalue))
        summary.append(entry)
        print(f"cfg {i} (w_H={cfg['w_H']}, psi_min={cfg['psi_min']}, lam={cfg['lam']}): "
              f"AUC-R reduction={reduction_pct:.1f}% (t={t.statistic:.2f}, p={t.pvalue:.4f})  |  "
              f"ABSOLUTE replay-harm reduction={abs_reduction_pct:.1f}% "
              f"(t={t_abs.statistic:.2f}, p={t_abs.pvalue:.4f})", flush=True)

    with open("results_reddit_sensitivity.json", "w") as f:
        json.dump(dict(baseline_AUC_R_mean=float(base_auc.mean()), sweep=summary), f, indent=2)
    print("\nSaved results_reddit_sensitivity.json")
    print("\nLook for: reduction_pct clearly positive AND p < 0.05.")


if __name__ == "__main__":
    main()
