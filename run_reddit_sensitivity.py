import json
import numpy as np
from scipy import stats
import run_reddit as R
from reddit_env import RedditEnv

GRID = [
    dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3, w_G=0.0, w_H=2.5, psi_min=0.20),
    dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3, w_G=0.0, w_H=4.0, psi_min=0.10),
    dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3, w_G=0.0, w_H=6.0, psi_min=0.05),
    dict(lam=0.10, alpha=1.0, delta=0.97, eta=0.7, tau=0.2, w_G=0.0, w_H=6.0, psi_min=0.05),
    dict(lam=0.20, alpha=1.0, delta=0.93, eta=0.4, tau=0.4, w_G=0.0, w_H=3.5, psi_min=0.15),
]
N_SEEDS = 30


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

        abs_reduction_pct = (base_rep.mean() - rapo_rep.mean()) / base_rep.mean() * 100
        t_abs = stats.ttest_rel(base_rep, rapo_rep)

        # THE FIX: aggregate util_deformed/util_undeformed into ReplayRet
        # per swept config -- this was computed by run_one_seed all along
        # (returned in each row) but never aggregated/saved by this sweep
        # script, so fig_utility_safety.pdf had no utility axis to plot.
        # Same aggregation pattern already used correctly in run_reddit.py's
        # main().
        total_util_def = sum(r["util_deformed"] for r in rows)
        total_util_undef = sum(r["util_undeformed"] for r in rows)
        replay_ret = total_util_def / total_util_undef if total_util_undef > 0 else float('nan')

        entry = dict(cfg=cfg, rapo_AUC_R_mean=float(rapo_auc.mean()),
                     reduction_pct=float(reduction_pct),
                     t_stat=float(t.statistic), p_value=float(t.pvalue),
                     rapo_rep_harm_mean=float(rapo_rep.mean()),
                     abs_reduction_pct=float(abs_reduction_pct),
                     t_stat_abs=float(t_abs.statistic), p_value_abs=float(t_abs.pvalue),
                     ReplayRet=float(replay_ret))
        summary.append(entry)
        print(f"cfg {i} (w_H={cfg['w_H']}, psi_min={cfg['psi_min']}, lam={cfg['lam']}): "
              f"AUC-R reduction={reduction_pct:.1f}% (t={t.statistic:.2f}, p={t.pvalue:.4f})  |  "
              f"ABSOLUTE replay-harm reduction={abs_reduction_pct:.1f}% "
              f"(t={t_abs.statistic:.2f}, p={t_abs.pvalue:.4f})  |  ReplayRet={replay_ret:.3f}", flush=True)

    with open("results_reddit_sensitivity.json", "w") as f:
        json.dump(dict(baseline_AUC_R_mean=float(np.nanmean(base_auc)), sweep=summary), f, indent=2)
    print("\nSaved results_reddit_sensitivity.json")


if __name__ == "__main__":
    main()
