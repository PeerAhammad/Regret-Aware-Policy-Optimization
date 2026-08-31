"""
Hyperparameter sweep for RAPO on MovieLens -- the first real run showed
RAPO's default config (carried over unchanged from the synthetic testbed)
did NOT clearly suppress harm (AUC-R: baseline 1.16 vs rapo 1.21, not
significant). This sweep tests whether stronger deformation actually
helps, now that harm registers correctly and cross-region escape is real.

Reuses run_movielens.run_one_seed directly. Writes
results_movielens_sensitivity.json. Also reports AUC-R as primary (RAG
saturates to 1.000 in this domain for the same reason it did in
safety_gym -- 200 steps is enough that SOME harmful hit is near-certain
regardless of condition, so peak-harm doesn't discriminate; cumulative
harm does).

v2 fixes (both found only after this file had already been run for real):
  1. reduction_pct was computed as a PER-SEED ratio (diff / that seed's
     own baseline AUC-R) and then averaged. This is numerically unstable
     -- a single seed with a near-zero baseline AUC-R blows the ratio
     toward +/-infinity and corrupts the average, even though the
     underlying simulation is fine. This is the exact bug that produced a
     real "-9,313,725,683%" artifact on Reddit data in analyze_reddit.py;
     same fix applied here: aggregate-MEANS reduction%, not a per-seed
     ratio averaged afterward.
  2. ReplayRet (utility) was never aggregated into the sweep summary, even
     though run_one_seed returns util_deformed/util_undeformed for every
     row -- so fig_utility_safety.pdf had no utility axis to plot for this
     domain. Fixed by aggregating it here, same pattern as run_movielens.py's
     main() already does correctly.
  Also renamed reduction_pct_mean -> reduction_pct for consistency with
  the Reddit sweep script's field names (a mismatch here silently broke
  the plotting script's field lookup).
"""

import json
import numpy as np
from scipy import stats
import run_movielens as M

GRID = [
    dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3, w_G=0.0, w_H=1.2, psi_min=0.35),  # original default
    dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3, w_G=0.0, w_H=2.5, psi_min=0.20),
    dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3, w_G=0.0, w_H=4.0, psi_min=0.10),
    dict(lam=0.10, alpha=1.0, delta=0.97, eta=0.7, tau=0.2, w_G=0.0, w_H=4.0, psi_min=0.10),
    dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.1, w_G=0.0, w_H=6.0, psi_min=0.05),
]
N_SEEDS = 20


def main():
    print("Building environment (shared across all configs)...", flush=True)
    env = M.MovieLensEnv(top_n=M.TOP_N, n_regions=M.N_REGIONS,
                          top_k_neighbors=M.TOP_K_NEIGHBORS, seed=0)

    base_rows = [M.run_one_seed(seed, "baseline", env) for seed in range(N_SEEDS)]
    base_auc = np.array([r["AUC_R"] for r in base_rows])
    print(f"Baseline AUC-R: {np.nanmean(base_auc):.3f} +/- {np.nanstd(base_auc,ddof=1):.3f}\n", flush=True)

    summary = []
    for i, cfg in enumerate(GRID):
        M.CONDITION_KWARGS["rapo"] = cfg
        rows = [M.run_one_seed(seed, "rapo", env) for seed in range(N_SEEDS)]
        rapo_auc = np.array([r["AUC_R"] for r in rows])

        valid = ~(np.isnan(base_auc) | np.isnan(rapo_auc))
        b, r_ = base_auc[valid], rapo_auc[valid]
        # FIX 1: aggregate-means reduction%, not a per-seed ratio averaged
        # afterward (see module docstring for why).
        reduction_pct = (b.mean() - r_.mean()) / b.mean() * 100
        t = stats.ttest_rel(b, r_)

        # FIX 2: aggregate util_deformed/util_undeformed into ReplayRet,
        # same pattern already used correctly in run_movielens.py's main().
        total_util_def = sum(r["util_deformed"] for r in rows)
        total_util_undef = sum(r["util_undeformed"] for r in rows)
        replay_ret = total_util_def / total_util_undef if total_util_undef > 0 else float('nan')

        entry = dict(cfg=cfg, rapo_AUC_R_mean=float(rapo_auc.mean()),
                     reduction_pct=float(reduction_pct),
                     t_stat=float(t.statistic), p_value=float(t.pvalue),
                     ReplayRet=float(replay_ret))
        summary.append(entry)
        print(f"cfg {i} (w_H={cfg['w_H']}, psi_min={cfg['psi_min']}, lam={cfg['lam']}): "
              f"AUC-R={rapo_auc.mean():.3f}, reduction={reduction_pct:.1f}%, "
              f"t={t.statistic:.2f}, p={t.pvalue:.4f}, ReplayRet={replay_ret:.3f}", flush=True)

    with open("results_movielens_sensitivity.json", "w") as f:
        json.dump(dict(baseline_AUC_R_mean=float(np.nanmean(base_auc)), sweep=summary), f, indent=2)
    print("\nSaved results_movielens_sensitivity.json")
    print("\nLook for: reduction_pct clearly positive AND p < 0.05. If NOTHING in this")
    print("grid gets there, the mechanism may need more than hyperparameter tuning --")
    print("worth flagging back rather than picking the least-bad option and moving on.")


if __name__ == "__main__":
    main()
