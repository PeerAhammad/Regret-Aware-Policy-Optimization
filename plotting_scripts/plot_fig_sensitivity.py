"""
Generates fig_sensitivity.pdf (suppression vs. individual hyperparameter)
from the SAME real sweep JSON files as plot_fig_utility_safety.py.

Same caveat: needs your real results_movielens_sensitivity.json /
results_reddit_sensitivity.json / results_graph_diffusion_sweep.json in
this folder. Run the sweep script(s) again if you don't have them saved.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sources = [
    ("results_movielens_sensitivity.json", "MovieLens-25M", "#1f77b4"),
    ("results_reddit_sensitivity.json", "Reddit Cascades", "#d62728"),
    ("results_graph_diffusion_sweep.json", "Graph Diffusion", "#2ca02c"),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
panel_params = ["w_H", "psi_min", "lam"]

any_found = False
for path, label, color in sources:
    if not os.path.exists(path):
        print(f"  (skipping {label}: {path} not found)")
        continue
    any_found = True
    with open(path) as f:
        data = json.load(f)
    rows = data.get("sweep", data.get("results", []))

    for ax, param in zip(axes, panel_params):
        xs, ys = [], []
        for r in rows:
            cfg = r.get("cfg", {})
            val = cfg.get(param)
            reduction = r.get("abs_reduction_pct", r.get("reduction_pct",
                         r.get("harm_reduction_pct")))
            # abs_reduction_pct checked FIRST: for Reddit, reduction_pct
            # (AUC-R-based) is known unreliable (ratio-instability
            # artifact -- see diagnose_reddit_backfire.py) and produces
            # nonsensical negative values even when real suppression is
            # happening. Always prefer the absolute-harm field when present.
            if val is not None and reduction is not None:
                xs.append(val)
                ys.append(reduction)
        if xs:
            # sort by x for a clean line
            pairs = sorted(zip(xs, ys))
            xs_sorted, ys_sorted = zip(*pairs)
            ax.plot(xs_sorted, ys_sorted, "o-", label=label, color=color)

if any_found:
    for ax, param in zip(axes, panel_params):
        ax.set_xlabel(param)
        ax.set_ylabel("Harm reduction (%)")
        ax.set_title(f"Sensitivity to {param}")
        ax.legend(fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig("fig_sensitivity.pdf")
    fig.savefig("fig_sensitivity_preview.png", dpi=100)
    print("\nSaved fig_sensitivity.pdf")
else:
    print("\nNo sweep JSON files found in this folder. Copy the real files here first:")
    for path, label, _ in sources:
        print(f"  {path}  (from the {label} sweep)")
