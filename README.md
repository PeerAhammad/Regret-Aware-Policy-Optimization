# RAPO / RSD — MovieLens-25M pipeline (rebuilt)

This replaces the previous MovieLens code entirely. Reasons, concretely:

1. **The old `MovieLensPolicy.recommend()` returned the same static top-10
   movies on every call**, regardless of history — there was no real
   `x_t -> x_{t+1}` transition, just a fixed list with one random
   "controversial" swap-in per step. Fixed: `movielens_env.py` now builds a
   genuine item-item similarity kernel (genre-cosine, top-K neighbors) and
   the walk genuinely depends on the current item, matching how the graph
   diffusion and Safety Gym domains work.

2. **The old code drew a NEW random controversial movie every single
   step**, in both Exposure and Replay — so RSD's core requirement
   ("reintroduce the SAME stimulus at Replay") was never actually
   satisfied. Fixed: one controversial seed item (`stimulus_item`) is
   chosen once, deterministically, when the environment is built, and
   reused identically for both Exposure and Replay resets.

3. **Utility was computed three different, incompatible ways** across old
   scripts (normalized-by-max in some, raw popularity sum in others, and a
   third version behind a script that wasn't in the project at all — the
   generator for `movielens_trials.json` was missing). Fixed: ONE utility
   definition (paired-RNG comparison of raw popularity counts visited on a
   held-out benign walk), used everywhere, in the one script
   (`run_movielens.py`) that produces `results_movielens.json`.

4. **A pasted headline number (utility change ≈ −1.30%, p≈9×10⁻¹⁶) did not
   match what `utility_statistics.py` actually outputs when run** (real
   output: −0.90% ± 3.37%, p=0.245, not significant). I could not trace
   the source of the mismatched numbers — the generating script wasn't in
   the project. This pipeline avoids the failure mode structurally: there
   is exactly one script that computes results and one that reads them,
   so there's nothing to fall out of sync.

5. **First real-data run came back with `RAG`/`AUC-R` = nan and
   `scar_gap` = exactly 0.000 for every condition** — harm never fired,
   at all, in any of the 80 (20 seeds × 4 conditions) runs. Root cause:
   harm was originally defined as "landed on a movie individually tagged
   `controversial`," but the walk only navigates genre-cosine similarity,
   and "controversial" has no structural relationship to genre — so the
   walk essentially never reached another controversial-tagged item.
   Switched to region-level harm (landed in `r_harm`, the stimulus item's
   own genre cluster), matching how graph_diffusion/safety_gym define
   harm. That surfaced a SECOND bug: `r_harm` was computed as the mode
   region across all 126 controversial items, which doesn't necessarily
   match the specific stimulus item's own region — confirmed directly (a
   walk starting at the stimulus item explored region 3; the
   independently-computed mode region was 2; harm still never fired).
   Fixed by tying `r_harm` directly to the actual stimulus item's own
   region. Verified against mock data that harm now fires and `scar_gap`
   comes out positive for every deforming condition.

## Files

- `movielens_env.py` — environment: candidate pool (top 3000 most-rated
  movies), genre-cosine item-item kernel, KMeans genre regions (R=75,
  matching the paper's stated region count for this domain),
  tag-based (`tags.csv`, `tag=="controversial"`) stimulus selection.
- `run_movielens.py` — the ONLY script that writes `results_movielens.json`.
  Same 4 conditions (baseline / pm_variant / rapo_off_replay / rapo), same
  deterministic per-condition seeding (an explicit dict, NOT `hash()` —
  `hash()` on strings is randomized per-process by default and silently
  broke reproducibility across machines in an earlier domain; caught and
  fixed there, avoided here from the start), same paired-RNG utility check
  used in graph_diffusion and safety_gym.
- `analyze_movielens.py` — reads `results_movielens.json` ONLY. No
  hardcoded numbers anywhere.
- `run_movielens_sensitivity.py` — hyperparameter sweep, reuses
  `run_movielens.run_one_seed` directly rather than reimplementing the
  experiment a second time.
- `check_movielens.py`, `inspect_movielens_tags.py` — kept from the
  original project; harmless read-only data inspection scripts.

## Setup

Place the real dataset at:
```
data/ml-25m/movies.csv
data/ml-25m/ratings.csv
data/ml-25m/tags.csv
```
(all three are part of the standard ml-25m.zip download; `tags.csv` is
required now since the controversial-movie stimulus comes from it).

Requires `pandas`, `numpy`, `scipy`, `scikit-learn`:
```
pip install pandas numpy scipy scikit-learn
```

Run:
```
python run_movielens.py
python analyze_movielens.py
```

`ratings.csv` (25M rows) is processed in chunks — safe for the full file,
no need to pre-filter it.

## What to check when you run this for real

- **Region count**: `n_regions` in the printed config should be close to
  75 (it's `min(75, n_candidate_items - 1)`, so only lower if the KMeans
  input is somehow degenerate).
- **`scar_gap@replay` must be nonzero (and positive) for `rapo`/
  `rapo_off_replay`/`pm_variant`.** If it's exactly 0.000 for every seed,
  harm never fired — this exact failure happened on the first real run
  and is now fixed (see point 5 above), but if it recurs, check
  `env.r_harm` really does equal `env.region_of[env.stimulus_item]`
  (should be true by construction now) and that harm is actually being
  hit during Exposure (add a print of `harm_exp.sum()` per seed).
- **`RAPO-off@rep`'s harm should land close to `baseline`'s** (the causal-
  isolation signature seen in both other domains) — if it doesn't, look at
  `deform_decay`/`deform_replay` flags in `run_one_seed` first.
- I have NOT run this against the real 25M-row dataset — only against a
  tiny synthetic mock of the same file schema, to verify the code runs
  end-to-end without crashing/producing nans. The actual headline numbers
  (RAG, AUC-R, ReplayRet) need to come from your run on the real data,
  the same way we did for graph_diffusion and safety_gym.
