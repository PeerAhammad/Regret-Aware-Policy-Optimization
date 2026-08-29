"""
Focused rerun of the single best candidate from run_movielens_sensitivity.py
(cfg1: w_H=2.5, psi_min=0.2, lam=0.15 -- t=1.19, p=0.25 at N=20, the only
config with the right sign and the least-bad p-value) at much higher N to
check whether the suppression effect is real-but-underpowered or genuinely
absent. Reuses run_movielens.run_one_seed directly -- no new logic.
"""

import numpy as np
from scipy import stats
import run_movielens as M

CFG = dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3, w_G=0.0, w_H=2.5, psi_min=0.20)
N_SEEDS = 60


def main():
    print("Building environment...", flush=True)
    env = M.MovieLensEnv(top_n=M.TOP_N, n_regions=M.N_REGIONS,
                          top_k_neighbors=M.TOP_K_NEIGHBORS, seed=0)

    print(f"Running {N_SEEDS} seeds for baseline and rapo (cfg={CFG})...", flush=True)
    base_rows = [M.run_one_seed(seed, "baseline", env) for seed in range(N_SEEDS)]
    M.CONDITION_KWARGS["rapo"] = CFG
    rapo_rows = [M.run_one_seed(seed, "rapo", env) for seed in range(N_SEEDS)]

    base_auc = np.array([r["AUC_R"] for r in base_rows])
    rapo_auc = np.array([r["AUC_R"] for r in rapo_rows])
    valid = ~(np.isnan(base_auc) | np.isnan(rapo_auc))
    b, r_ = base_auc[valid], rapo_auc[valid]

    print(f"\nn = {len(b)}")
    print(f"baseline AUC-R: {b.mean():.3f} +/- {b.std(ddof=1):.3f}")
    print(f"rapo AUC-R:     {r_.mean():.3f} +/- {r_.std(ddof=1):.3f}")

    t = stats.ttest_rel(b, r_)
    print(f"\nPaired t-test: t = {t.statistic:.4f}, p = {t.pvalue:.6g}")

    reduction_pct = (b.mean() - r_.mean()) / b.mean() * 100
    print(f"Reduction in mean AUC-R: {reduction_pct:.2f}%")

    if t.pvalue < 0.05 and t.statistic > 0:
        print("\n--> Significant suppression at N=60. This config is a real candidate.")
    else:
        print("\n--> Still not significant. This is a genuine finding: the effect is")
        print("    weak/marginal (or absent) in this domain as currently designed --")
        print("    worth reporting honestly rather than as a clean suppression result.")


if __name__ == "__main__":
    main()
