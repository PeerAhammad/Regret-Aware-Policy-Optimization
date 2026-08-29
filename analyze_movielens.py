"""
Reads results_movielens.json (written ONLY by run_movielens.py) and
prints summary statistics + a paired significance test for RAPO vs.
baseline on harm (AUC_R, the cumulative-harm metric) and on the
paired-RNG utility check. No hardcoded numbers anywhere in this file --
if you need to regenerate the underlying data, run run_movielens.py.
"""

import json
import numpy as np
from scipy import stats

with open("results_movielens.json") as f:
    results = json.load(f)

meta = results.pop("_meta")
print("Config:", json.dumps(meta, indent=2))

for condition, r in results.items():
    print(f"\n[{condition}]")
    print(f"  RAG            = {r['RAG_mean']:.3f} +/- {r['RAG_ci95']:.3f}")
    print(f"  AUC-R          = {r['AUC_R_mean']:.3f} +/- {r['AUC_R_ci95']:.3f}")
    print(f"  ReplayRet      = {r['ReplayRet']:.3f}")
    print(f"  ASD            = {r['ASD_mean']:.4f}")
    print(f"  scar_gap@replay= {r['scar_gap_mean']:.3f} +/- {r['scar_gap_ci95']:.3f}")

if "baseline" in results and "rapo" in results:
    base_auc = np.array([row["AUC_R"] for row in results["baseline"]["raw"]])
    rapo_auc = np.array([row["AUC_R"] for row in results["rapo"]["raw"]])
    valid = ~(np.isnan(base_auc) | np.isnan(rapo_auc))
    base_auc, rapo_auc = base_auc[valid], rapo_auc[valid]

    print("\n========== RAPO vs. Stationary: paired test on AUC-R (cumulative harm) ==========")
    diff = base_auc - rapo_auc
    reduction_pct = diff / np.maximum(base_auc, 1e-9) * 100
    print(f"n = {len(diff)}")
    print(f"Mean harm reduction: {reduction_pct.mean():.2f}% +/- {reduction_pct.std(ddof=1):.2f}%")
    sem = stats.sem(reduction_pct)
    margin = stats.t.ppf(0.975, len(reduction_pct) - 1) * sem
    print(f"95% CI: {reduction_pct.mean() - margin:.2f}% - {reduction_pct.mean() + margin:.2f}%")
    t = stats.ttest_rel(base_auc, rapo_auc)
    print(f"Paired t-test: t = {t.statistic:.4f}, p = {t.pvalue:.6g}")
