"""
Generates fig_utility_safety.pdf (Pareto tradeoff: suppression vs. utility
cost) from REAL sweep data.

This needs the actual sweep JSON files produced on your machine:
  - results_movielens_sensitivity.json  (from run_movielens_sensitivity.py)
  - results_reddit_sensitivity.json     (from run_reddit_sensitivity.py)
  - results_graph_diffusion_sweep.json  (from sweep_graph_diffusion.py --
    rename/save its checkpoint output to this filename, or edit the path
    below to match whatever you saved it as)

Run the relevant sweep script(s) again if you don't still have the JSON
(they get deleted/overwritten between sessions in my sandbox, and I don't
have reliable enough memory of every exact number to safely reconstruct
this plot without the real file -- reconstructing from memory here would
risk exactly the kind of fabrication this whole project has been about
avoiding).

Place this script in the same folder as whichever JSON files you have,
and run: python plot_fig_utility_safety.py
It will use whichever of the three domains it finds and skip the rest
with a clear message, rather than fail or guess.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 5))

sources = [
    ("results_movielens_sensitivity.json", "MovieLens-25M", "#1f77b4"),
    ("results_reddit_sensitivity.json", "Reddit Cascades", "#d62728"),
    ("results_graph_diffusion_sweep.json", "Graph Diffusion", "#2ca02c"),
]

any_found = False
for path, label, color in sources:
    if not os.path.exists(path):
        print(f"  (skipping {label}: {path} not found)")
        continue
    any_found = True
    with open(path) as f:
        data = json.load(f)

    # Different sweep scripts saved slightly different schemas -- handle
    # both "sweep": [...] (movielens/reddit) and "results": [...] (graph
    # diffusion) shapes.
    rows = data.get("sweep", data.get("results", []))

    xs, ys = [], []
    for r in rows:
        # x-axis: utility/suppression cost. Use ReplayRet-style utility if
        # present, else fall back to whatever utility-like field exists --
        # THIS PART MAY NEED ADJUSTING once you see the actual keys in
        # your JSON; print(rows[0].keys()) below to check.
        reduction = r.get("reduction_pct", r.get("harm_reduction_pct",
                     r.get("abs_reduction_pct")))
        utility = r.get("ReplayRet_mean", r.get("ReplayRet"))
        if reduction is not None and utility is not None:
            xs.append(utility)
            ys.append(reduction)

    if not xs:
        print(f"  ({label}: found file but no ReplayRet/reduction pairs -- "
              f"check the field names, first row keys: "
              f"{list(rows[0].keys()) if rows else 'no rows'})")
        continue

    ax.scatter(xs, ys, label=label, color=color, s=60)

if not any_found:
    print("\nNo sweep JSON files found in this folder. Copy the real files here first:")
    for path, label, _ in sources:
        print(f"  {path}  (from the {label} sweep)")
else:
    ax.set_xlabel("Utility preserved (ReplayRet)")
    ax.set_ylabel("Harm reduction (%)")
    ax.set_title("Suppression vs. utility tradeoff across swept configs")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_utility_safety.pdf")
    fig.savefig("fig_utility_safety_preview.png", dpi=100)
    print("\nSaved fig_utility_safety.pdf")
