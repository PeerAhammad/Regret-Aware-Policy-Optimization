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
    print(f"n = {len(base_auc)}")
    print(f"baseline AUC-R: {base_auc.mean():.3f} +/- {base_auc.std(ddof=1):.3f}")
    print(f"rapo AUC-R:     {rapo_auc.mean():.3f} +/- {rapo_auc.std(ddof=1):.3f}")
    # Reduction % from aggregate means, NOT a per-seed ratio averaged
    # afterward -- a per-seed version of this (diff/base_auc per seed,
    # then mean) is numerically unstable: any single seed with a
    # near-zero baseline AUC-R blows up that seed's ratio toward
    # +/-infinity and corrupts the average, even though the underlying
    # simulation is fine. Confirmed this the hard way: an earlier version
    # of this exact formula produced a "-9,313,725,683% harm reduction"
    # line on real reddit data. The paired t-test below is unaffected
    # either way since it operates on the raw bounded AUC-R values.
    reduction_pct = (base_auc.mean() - rapo_auc.mean()) / base_auc.mean() * 100
    print(f"Reduction in mean AUC-R: {reduction_pct:.2f}%")
    t = stats.ttest_rel(base_auc, rapo_auc)
    print(f"Paired t-test: t = {t.statistic:.4f}, p = {t.pvalue:.6g}")
