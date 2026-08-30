"""
Reads results_reddit.json (written ONLY by run_reddit.py) and runs a
paired significance test for RAPO vs. baseline on AUC-R. Same pattern as
analyze_movielens.py -- no hardcoded numbers.
"""

import json
import numpy as np
from scipy import stats

with open("results_reddit.json") as f:
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

    print("\n========== RAPO vs. Stationary: paired test on AUC-R ==========")
    print(f"n = {len(base_auc)}")
    print(f"baseline AUC-R: {base_auc.mean():.3f} +/- {base_auc.std(ddof=1):.3f}")
    print(f"rapo AUC-R:     {rapo_auc.mean():.3f} +/- {rapo_auc.std(ddof=1):.3f}")
    reduction_pct = (base_auc.mean() - rapo_auc.mean()) / base_auc.mean() * 100
    print(f"Reduction in mean AUC-R: {reduction_pct:.2f}%")
    t = stats.ttest_rel(base_auc, rapo_auc)
    print(f"Paired t-test: t = {t.statistic:.4f}, p = {t.pvalue:.6g}")

    if t.pvalue < 0.05 and t.statistic > 0:
        print("--> Significant.")
    else:
        print("--> NOT significant on AUC-R. See the ABSOLUTE replay-harm test below --")
        print("    AUC-R is a ratio against each condition's OWN exposure-phase harm,")
        print("    and RAPO suppresses exposure too, which can make the ratio misleading")
        print("    even when absolute replay harm is genuinely lower (confirmed via")
        print("    diagnose_reddit_backfire.py). The absolute comparison below is the")
        print("    more direct test of 'does RAPO reduce replay harm vs. no intervention'.")

    # --- Absolute replay-harm comparison (the more direct cross-condition
    # test -- see diagnose_reddit_backfire.py for why AUC-R's ratio
    # structure can be misleading here) ---
    if "rep_harm_total" in results["baseline"]["raw"][0]:
        base_rep = np.array([row["rep_harm_total"] for row in results["baseline"]["raw"]])
        rapo_rep = np.array([row["rep_harm_total"] for row in results["rapo"]["raw"]])
        print("\n========== RAPO vs. Stationary: paired test on ABSOLUTE replay harm ==========")
        print(f"n = {len(base_rep)}")
        print(f"baseline replay harm (mean count over {meta.get('T_replay','?')} steps): "
              f"{base_rep.mean():.3f} +/- {base_rep.std(ddof=1):.3f}")
        print(f"rapo replay harm:     {rapo_rep.mean():.3f} +/- {rapo_rep.std(ddof=1):.3f}")
        abs_reduction_pct = (base_rep.mean() - rapo_rep.mean()) / base_rep.mean() * 100
        print(f"Reduction in mean absolute replay harm: {abs_reduction_pct:.2f}%")
        t_abs = stats.ttest_rel(base_rep, rapo_rep)
        print(f"Paired t-test: t = {t_abs.statistic:.4f}, p = {t_abs.pvalue:.6g}")
        if t_abs.pvalue < 0.05 and t_abs.statistic > 0:
            print("--> Significant reduction in ABSOLUTE replay harm.")
    else:
        print("\n(results_reddit.json doesn't have rep_harm_total -- rerun run_reddit.py")
        print(" with the updated version to get the absolute-harm comparison.)")

if "baseline" in results and "rapo_off_replay" in results:
    base_auc = np.array([row["AUC_R"] for row in results["baseline"]["raw"]])
    off_auc = np.array([row["AUC_R"] for row in results["rapo_off_replay"]["raw"]])
    valid = ~(np.isnan(base_auc) | np.isnan(off_auc))
    base_v, off_v = base_auc[valid], off_auc[valid]
    t2 = stats.ttest_rel(off_v, base_v)
    print(f"\n========== RAPO-off@rep vs. Stationary (causal-isolation check) ==========")
    print(f"off_replay AUC-R: {off_v.mean():.3f}, baseline AUC-R: {base_v.mean():.3f}")
    print(f"Paired t-test (off_replay > baseline): t = {t2.statistic:.4f}, p = {t2.pvalue:.6g}")

    if "rep_harm_total" in results["baseline"]["raw"][0] and "rep_harm_total" in results["rapo_off_replay"]["raw"][0]:
        base_rep = np.array([row["rep_harm_total"] for row in results["baseline"]["raw"]])
        off_rep = np.array([row["rep_harm_total"] for row in results["rapo_off_replay"]["raw"]])
        t2_abs = stats.ttest_rel(off_rep, base_rep)
        print(f"\n-- same check on ABSOLUTE replay harm --")
        print(f"off_replay: {off_rep.mean():.3f} +/- {off_rep.std(ddof=1):.3f}   "
              f"baseline: {base_rep.mean():.3f} +/- {base_rep.std(ddof=1):.3f}")
        print(f"Paired t-test (off_replay > baseline): t = {t2_abs.statistic:.4f}, p = {t2_abs.pvalue:.6g}")

if "baseline" in results and "pm_variant" in results:
    base_rep = np.array([row["rep_harm_total"] for row in results["baseline"]["raw"]])
    pm_rep = np.array([row["rep_harm_total"] for row in results["pm_variant"]["raw"]])
    t3 = stats.ttest_rel(base_rep, pm_rep)
    reduction_pm = (base_rep.mean() - pm_rep.mean()) / base_rep.mean() * 100
    print(f"\n========== PM-variant vs. Stationary: ABSOLUTE replay harm ==========")
    print(f"pm_variant: {pm_rep.mean():.3f} +/- {pm_rep.std(ddof=1):.3f}   "
          f"baseline: {base_rep.mean():.3f} +/- {base_rep.std(ddof=1):.3f}")
    print(f"Reduction: {reduction_pm:.2f}%")
    print(f"Paired t-test: t = {t3.statistic:.4f}, p = {t3.pvalue:.6g}")

    if "rapo" in results:
        rapo_rep = np.array([row["rep_harm_total"] for row in results["rapo"]["raw"]])
        t4 = stats.ttest_rel(pm_rep, rapo_rep)
        print(f"\n-- PM-variant vs. RAPO (does persistent scar H beat fast-memory-only?) --")
        print(f"pm_variant: {pm_rep.mean():.3f}   rapo: {rapo_rep.mean():.3f}")
        print(f"Paired t-test (pm > rapo): t = {t4.statistic:.4f}, p = {t4.pvalue:.6g}")
