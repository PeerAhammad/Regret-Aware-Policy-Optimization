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
    print(f"Baseline AUC-R: {base_auc.mean():.3f} +/- {base_auc.std(ddof=1):.3f}\n", flush=True)

    summary = []
    for i, cfg in enumerate(GRID):
        M.CONDITION_KWARGS["rapo"] = cfg
        rows = [M.run_one_seed(seed, "rapo", env) for seed in range(N_SEEDS)]
        rapo_auc = np.array([r["AUC_R"] for r in rows])

        valid = ~(np.isnan(base_auc) | np.isnan(rapo_auc))
        b, r_ = base_auc[valid], rapo_auc[valid]
        reduction_pct = (b - r_) / np.maximum(b, 1e-9) * 100
        t = stats.ttest_rel(b, r_)

        entry = dict(cfg=cfg, rapo_AUC_R_mean=float(rapo_auc.mean()),
                     reduction_pct_mean=float(reduction_pct.mean()),
                     t_stat=float(t.statistic), p_value=float(t.pvalue))
        summary.append(entry)
        print(f"cfg {i} (w_H={cfg['w_H']}, psi_min={cfg['psi_min']}, lam={cfg['lam']}): "
              f"AUC-R={rapo_auc.mean():.3f}, reduction={reduction_pct.mean():.1f}%, "
              f"t={t.statistic:.2f}, p={t.pvalue:.4f}", flush=True)

    with open("results_movielens_sensitivity.json", "w") as f:
        json.dump(dict(baseline_AUC_R_mean=float(base_auc.mean()), sweep=summary), f, indent=2)
    print("\nSaved results_movielens_sensitivity.json")
    print("\nLook for: reduction_pct clearly positive AND p < 0.05. If NOTHING in this")
    print("grid gets there, the mechanism may need more than hyperparameter tuning --")
    print("worth flagging back rather than picking the least-bad option and moving on.")


if __name__ == "__main__":
    main()
