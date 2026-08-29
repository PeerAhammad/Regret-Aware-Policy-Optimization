"""
Hyperparameter sensitivity sweep for RAPO on MovieLens. Reuses
run_movielens.run_one_seed directly against a shared env -- no hardcoded
numbers, no separate reimplementation. Writes results_movielens_sensitivity.json.
"""

import json
import numpy as np
from rapo_core import TraceScarMemory
import run_movielens as M

GRID = [
    dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3, w_G=0.0, w_H=0.8, psi_min=0.25),
    dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3, w_G=0.0, w_H=1.2, psi_min=0.35),
    dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3, w_G=0.0, w_H=1.6, psi_min=0.35),
    dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3, w_G=0.0, w_H=2.0, psi_min=0.25),
]
N_SEEDS = 20


def run_config(cfg, env):
    rows = []
    for seed in range(N_SEEDS):
        M.CONDITION_KWARGS["rapo"] = cfg
        row = M.run_one_seed(seed, "rapo", env)
        rows.append(row)
    auc = np.array([r["AUC_R"] for r in rows])
    return float(np.nanmean(auc)), rows


def main():
    print("Building environment...", flush=True)
    env = M.MovieLensEnv(top_n=M.TOP_N, n_regions=M.N_REGIONS,
                          top_k_neighbors=M.TOP_K_NEIGHBORS, seed=0)

    # baseline (stationary) AUC-R for reduction% reference, same env/seeds
    base_rows = [M.run_one_seed(seed, "baseline", env) for seed in range(N_SEEDS)]
    base_auc = np.nanmean([r["AUC_R"] for r in base_rows])
    print(f"Baseline AUC-R: {base_auc:.3f}\n", flush=True)

    summary = []
    for cfg in GRID:
        auc_mean, rows = run_config(cfg, env)
        reduction = (base_auc - auc_mean) / base_auc * 100 if base_auc > 0 else float('nan')
        summary.append(dict(cfg=cfg, AUC_R_mean=auc_mean, harm_reduction_pct=reduction))
        print(f"w_H={cfg['w_H']}, psi_min={cfg['psi_min']} -> "
              f"AUC-R={auc_mean:.3f}, reduction={reduction:.2f}%", flush=True)

    with open("results_movielens_sensitivity.json", "w") as f:
        json.dump(dict(baseline_AUC_R=float(base_auc), sweep=summary), f, indent=2)
    print("\nSaved results_movielens_sensitivity.json")


if __name__ == "__main__":
    main()
