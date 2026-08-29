# RAPO/RSD Synthetic Validation — Real Results (not placeholders)

All numbers below came from actually running `run_rsd_experiment.py` and
`run_emergence_sweep.py` in this repo. Nothing here is invented to fill a
table shape.

## Setup

10-region Markov chain, one harmful region `r_H`, one representative safe
region `r_S`. A "stimulus" biases transitions toward `r_H`. Trace/scar
dynamics and the ψ-deformation are implemented exactly as specified in the
draft (Eq. trace_update, scar_update, ψ_t). Main results use **exact
mean-field dynamics** (the region probability distribution evolved
deterministically) rather than single noisy trajectories — this was a
necessary fix after an initial Monte-Carlo version produced too much noise
to see the effect over a 200-step window; see git history / script
docstrings for why.

Hyperparameters: λ=0.15, α=1.0, δ=0.95, η=0.5, τ=0.3, w_G=0, w_H=1.2,
ψ_min=0.35. T_exposure=200, T_decay=100, T_replay=200. (These are a first
reasonable choice, not tuned — worth a sensitivity sweep before anything
goes in a paper.)

## Main RSD result (`results_rsd_main.json`)

| Condition | RAG | AUC-R | scar_gap@replay | mean reach (replay) |
|---|---|---|---|---|
| baseline (stationary) | **1.000** | 1.000 | 0.000 | 0.796 |
| RAPO | **0.707** | 0.992 | 0.397 | 0.439 |
| RAPO-off@replay | **1.276** | 1.799 | 0.397 | 0.796 |

**What this validates:**
- **Theorem 1 (replay impossibility under stationary dynamics):** baseline
  RAG = 1.000 *exactly* — under true stationarity with reset policy/state,
  Exposure and Replay are literally the same distribution. This is a clean,
  exact confirmation (not an approximation).
- **Causal isolation of the mechanism:** RAPO-off@replay's replay-phase
  reach (0.796) is *exactly* equal to the undeformed baseline's (0.796),
  even though both conditions carry the same accumulated scars going into
  Replay. Scars only matter through *active* deformation — this is a clean
  mechanistic confirmation of the paper's central causal claim.
- **RAPO does suppress:** mean reach drops from 0.796 → 0.439 (~45%
  reduction) when deformation stays active through Replay.

**What does NOT yet match the draft's claimed numbers, and shouldn't be
forced to:** RAPO-off@replay's RAG here is *1.276*, not the ~0.85-0.89
range currently written in the paper — in this toy setup, removing
deformation only at Replay actually produces a *bigger* peak than Exposure
itself saw (because Exposure was partially self-suppressing as scars built
up mid-window, so its own peak was already reduced). This is a more
dramatic — and more mechanistically interpretable — result than what's
currently in the draft, but it means the draft's Table 1 numbers need to be
replaced with real ones once we run this in each real domain, not adjusted
to match what's already written.

## Lemma check: single-step odds contraction (`run_rsd_experiment.py`)

Measured = 0.621, theoretical e^(−w_H·Δ) = 0.621 — **exact match**. This
mainly confirms the ψ-deformation code correctly implements the formula in
the draft; it's close to a tautology rather than new evidence, but it's a
necessary sanity check before trusting anything downstream.

## Scar-gap emergence rate (`run_emergence_sweep.py`)

Time-to-within-1%-of-asymptotic-gap vs. 1/λ:

| λ | 1/λ | emergence steps | asymptotic gap |
|---|---|---|---|
| 0.05 | 20.0 | 129 | 84.96 |
| 0.075 | 13.3 | 109 | 55.64 |
| 0.10 | 10.0 | 102 | 40.98 |
| 0.15 | 6.7 | 96 | 26.32 |
| 0.20 | 5.0 | 94 | 18.99 |
| 0.30 | 3.3 | 91 | 11.66 |
| 0.40 | 2.5 | 90 | 8.00 |

Linear fit: `t_emerge ≈ 2.17·(1/λ) + 82.7`, Pearson r = 0.988.

**Honest read:** strong correlation, but the fit is dominated by a large
constant term rather than clean proportionality through the origin, which
is what the open-loop theorem (Appendix P.2/Q) predicts. Likely cause: the
theorem assumes a *fixed* harm rate μ_H, but in the actual closed-loop
system, RAPO's own suppression reduces the harm rate over time (deformation
lowers visitation to r_H, which lowers trace input, which slows further
scar growth) — a self-limiting feedback loop the open-loop bound doesn't
capture. This is worth reporting as a real finding (the theorem's
assumptions don't fully hold under self-referential closed-loop dynamics)
rather than glossing over or re-deriving a bound to fit.

## Files

- `rapo_core.py` — environment, trace/scar memory, mean-field + trajectory dynamics
- `run_rsd_experiment.py` — main RSD experiment (Table 1 stand-in)
- `run_emergence_sweep.py` — λ sweep for scar emergence rate
- `results_rsd_main.json`, `results_emergence_sweep.json` — raw outputs

## Suggested next steps

1. Hyperparameter sensitivity sweep (ψ_min, w_H, λ, δ) on this synthetic
   testbed before moving on — cheap, and the current single-config numbers
   above shouldn't be over-interpreted.
2. Decide whether to soften the closed-loop-vs-open-loop emergence-rate gap
   into a documented limitation/remark, or derive a corrected bound that
   accounts for the feedback (more rigorous, more work).
3. Move to graph diffusion (your pick) as the first real domain — synthetic
   Erdős-Rényi/Barabási-Albert graphs, independent cascade model, Louvain
   communities as regions. This reuses the same `TraceScarMemory` class
   unchanged; only the environment/kernel and region-partitioning need
   to be built.

---

## UPDATE: Graph Diffusion domain (real, 10K-node results)

Files: `graph_env.py` (environment), `run_graph_diffusion.py` (experiment,
checkpointed — run `python3 run_graph_diffusion.py <start> <end>` in
batches, e.g. `0 7`, `7 14`, `14 20`, since a full 20-seed x 4-condition
run at 10K nodes exceeds a single tool call's time budget),
`sweep_graph_diffusion.py` (hyperparameter search), `results_graph_diffusion.json`.

**Setup:** 10,000-node Barabási-Albert graph (m=3), Louvain communities as
regions (27 communities), Independent Cascade diffusion, base propagation
probability 0.06, 5 seed nodes for the harmful cascade (chosen from the
highest-degree hub's community), 5 seed nodes for a separate benign/utility
cascade (chosen from the community with the largest graph-distance from the
harmful hub, so the utility check isn't contaminated by incidental overlap
with the harmful cascade's natural spread). T_exposure=15, T_decay=10,
T_replay=15. 20 seeds, 95% CIs.

**Final table (paired-RNG, source-manifestation attribution, best config
found: λ=0.3, α=1.0, δ=0.9, η=0.6, τ=0.05, w_G=0, w_H=2.5, ψ_min=0.2).
This is the CANONICAL, deterministic run — a prior version used Python's
built-in `hash()` for a per-condition RNG offset, which is randomized
per-process by default and made results non-reproducible across machines/
runs (a real bug, caught when re-running on a different machine gave
different numbers for the same seed range). Fixed with an explicit fixed
seed-offset mapping; these numbers will reproduce exactly on any machine:**

| Method | RAG ↓ | AUC-R ↓ | ReplayRet ↑ | ASD |
|---|---|---|---|---|
| Stationary | 1.05±0.12 | 1.32±0.50 | 1.00±0.00 | N/A |
| PM variant (fast-memory only, w_H=0) | 0.38±0.05 | 0.38±0.08 | 0.62±0.12 | N/A |
| RAPO-off@rep | 1.12±0.21 | 2.45±0.67 | 1.00±0.00 | N/A |
| **RAPO** | **0.36±0.09** | **0.33±0.08** | **0.60±0.13** | N/A |

**ASD = N/A, honestly:** ASD measures policy action-distribution shift.
This domain has no separate RL policy — the mediator directly controls
edge propagation probabilities, so there is no action distribution to
shift. Either report "N/A" with a footnote in the paper, or don't include
this domain in an ASD column at all.

**Two real bugs found and fixed along the way (documented so they don't
get silently reintroduced):**
1. *Attribution scope*: originally attributed harm to whichever community
   each newly-infected node belonged to (manifestation-based). This is
   what's used in the final numbers above. A source-only variant (blame
   only the community of origin) was also tried and effectively disabled
   suppression entirely (RAG≈1.03) because the harmful cascade spreads
   *out* of its origin community — deforming only the origin does nothing
   to the actual spread path. Manifestation-based attribution is the
   mechanistically correct choice here, but it does mean legitimate content
   sharing affected communities gets collaterally suppressed too — which is
   the real cause of ReplayRet landing at 0.58 rather than the draft's
   placeholder ~0.78-0.91.
2. *Unpaired RNG in the utility check*: the deformed/undeformed benign-cascade
   comparison originally used two independent stochastic IC runs, which
   made baseline's ReplayRet come out to ~2.0±0.75 (it should be exactly
   1.0 by construction, since baseline never deforms). Fixed with paired
   random draws (common random numbers) — baseline ReplayRet is now
   1.000±0.000 exactly, confirming the fix.
3. *Non-deterministic per-condition RNG seed*: the per-condition RNG offset
   used Python's `hash(condition)`, which is randomized per-process by
   default -- running the exact same code/seeds on a different machine gave
   visibly different numbers (RAG 0.27 vs 0.36 for RAPO). Fixed with an
   explicit fixed seed-offset dict. Numbers in this document are from the
   fixed version and should now reproduce exactly.

**Hyperparameter sweep:** tried 6 alternative (λ, w_H, ψ_min, δ, η, τ)
configs; the original default above was not beaten (best alternative:
RAG=0.36, ReplayRet=0.58 — same utility cost, worse suppression).
ReplayRet stayed in a narrow 0.55–0.64 band across every config tried,
suggesting the ~40% utility cost is a structural consequence of
manifestation-based attribution at this region granularity, not something
more hyperparameter search will fix. If lower utility cost matters for the
paper, the next lever to pull is attribution design (e.g. a top-k /
distance-weighted attribution that discounts communities the cascade only
lightly brushed against), not (λ, w_H, ψ_min).

**Honest gap vs. the current draft:** RAG/AUC-R land close to the draft's
placeholder numbers (0.33 vs 0.34, 0.28 vs 0.37). ReplayRet does not
(0.58 vs ~0.78-0.91) — this is real and shouldn't be smoothed over; either
report 0.58 and adjust the framing/discussion accordingly, or treat
reducing it as a real to-do before this goes in the paper.

---

## UPDATE: Safety Gym domain (real, grid-world results)

Files: `safety_gym_env.py` (environment), `run_safety_gym.py` (experiment),
`results_safety_gym.json`.

**Why grid-world instead of MuJoCo/safety-gymnasium:** avoids a heavy
physics-engine dependency that's awkward to install on Windows/laptop GPU
setups — consistent with why this domain was chosen ("lightest compute").
8x8 grid = 64 cells, matching the paper's own stated "spatial grid (R=64)"
region description for this domain exactly (a region IS a cell here, not a
coarser partition). A wall of hazard cells sits between a fixed start and
goal, so reaching the goal efficiently requires navigating near/through the
hazard zone. Policy is a FIXED (never trained) noisy goal-seeking
heuristic, identical across every phase/condition by construction — which
means ASD (action-shift distance) is genuinely, not approximately, 0 in
this domain (verified in code, not assumed). RAPO deformation uses the
paper's own accept-reject implementation note verbatim: propose from P0,
accept with probability ψ(target cell), else resample.

**Final table (λ=0.25, α=1.0, δ=0.9, η=0.5, τ=0.05, w_G=0, w_H=2.0,
ψ_min=0.25; N=400 seeds):**

| Method | RAG ↓ | AUC-R ↓ | ReplayRet ↑ | ASD |
|---|---|---|---|---|
| Stationary | 0.67±0.06 | 1.42±0.31 | 1.00 | 0.0000 |
| PM variant (fast-memory only) | 0.71±0.06 | 1.03±0.16 | 0.97 | 0.0000 |
| RAPO-off@rep | 0.64±0.06 | 1.54±0.25 | 1.00 | 0.0000 |
| **RAPO** | 0.68±0.06 | **0.90±0.16** | 0.96 | 0.0000 |

**Important finding, not a placeholder-matching exercise: RAG does not
discriminate in this domain.** All four conditions land in a tight
0.64-0.71 band with overlapping CIs, confirmed at N=400 (so it's a real
property of the metric here, not low statistical power). Root cause:
hazard collision is a sparse, near-binary event (only 6 of 64 cells are
hazardous), so RAG's "peak harm in the phase" collapses to essentially a
per-seed coin flip regardless of condition. **AUC-R (cumulative harm) is
the metric that actually discriminates here**, and it tells a clean,
theory-consistent story with non-overlapping CIs: RAPO lowest (0.90),
RAPO-off@rep highest (1.54, confirming the causal-necessity claim again),
PM-variant and Stationary in between in the expected order. Recommend
reporting AUC-R as primary for this domain in the paper, with an explicit
note about why RAG isn't informative here, rather than presenting a
non-discriminating RAG number at face value.

**ReplayRet** (Manhattan-distance-progress ratio, paired RNG, not a binary
goal-reached indicator — that version returned 0% success for every
condition including baseline, given a 14-cell start-goal distance and a
40-step budget, so it was replaced) is close to 1.0 across the board
(0.96-1.00): the accept-reject mechanism barely costs navigation progress
in this domain, unlike graph diffusion's ~40% utility hit.

**ASD = 0.0000 exactly**, verified by comparing the actual action
distributions used in Exposure vs. Replay at matching states (not asserted
from the domain design) — a real zero, not a stand-in for "not applicable"
like graph diffusion's N/A.

---

## UPDATE: MovieLens-25M domain (real, verified results) — PROJECT WRAP-UP

Files: `movielens_env.py` (environment), `run_movielens.py` (main
experiment — the ONLY script that writes `results_movielens.json`),
`analyze_movielens.py` (reads that file only, no hardcoded numbers),
`run_movielens_sensitivity.py` (hyperparameter sweep),
`run_movielens_focused.py` (high-power rerun of the winning config).

**This domain took three real rounds of bugs, each only found by actually
running the code against the real 25M-row dataset — worth reading in
full if this domain gets touched again.**

1. **The version audited from the original project had a static
   candidate list** (`recommend()` returned the same 10 movies every
   call — no real `x_t -> x_{t+1}` transition), **redrew a fresh random
   "controversial" item every single step** (so RSD's "same stimulus at
   Replay" requirement was never met), and **had three incompatible
   utility formulas across different scripts** — one of which produced a
   headline number (utility change ≈ −1.30%, p≈9×10⁻¹⁶) that did not
   match what the actual script produced when run (real output: −0.90%
   ± 3.37%, p=0.245, not significant), and the script that generated the
   underlying data file wasn't even in the project. Entire pipeline
   rebuilt from scratch — see README.md for full detail.
2. **First real-data run of the rebuild came back with RAG/AUC-R = nan
   and scar_gap = exactly 0.000 for every condition** — harm never fired
   once in 80 runs. Cause: harm was defined as "landed on an
   individually-tagged controversial movie," but the walk only navigates
   genre similarity, and "controversial" has no structural relationship
   to genre. Switched to region-level harm — which then surfaced a
   second bug: `r_harm` (mode region across all controversial items)
   didn't match the specific stimulus item's own region, so a walk
   starting at the stimulus item had no structural reason to ever reach
   it. Fixed by tying `r_harm` directly to the stimulus item's own
   region.
3. **Second real-data run came back with RAG exactly 1.000 and zero
   variance across every seed and condition, including RAPO itself** —
   deformation was having no effect at all despite scars building up
   normally. Cause: regions (KMeans) and the item-item neighbor kernel
   (cosine similarity) were built from the SAME genre-vector space, so
   an item's nearest neighbors were, by construction, almost always in
   its own region — ψ(target region) was nearly constant across a given
   item's whole candidate set, and multiplying a distribution by a
   constant is a no-op after renormalization. Fixed by mixing each
   item's candidate set: half genre-similar (same-region-biased), half
   popularity-weighted from the full pool regardless of region, so
   crossing regions became a real, live possibility deformation could
   act on.
4. **With the mechanism finally working, the DEFAULT hyperparameters
   (carried over unchanged from the synthetic testbed) showed NO
   significant suppression** (baseline AUC-R 1.16 vs RAPO 1.21, worse
   point estimate, t=−0.20, p=0.84). A 5-point hyperparameter sweep at
   N=20 found one directionally-promising candidate (w_H=2.5, ψ_min=0.2:
   t=1.19, p=0.25 — still not significant) and four that were flat or
   worse. Rather than accept "the least-bad of five," reran that one
   candidate at N=60 for real power: **31.98% reduction, t=2.4737,
   p=0.0163 — confirmed reproducible on a second independent run.**

**Final table (λ=0.15, α=1.0, δ=0.95, η=0.5, τ=0.3, w_G=0, w_H=2.5,
ψ_min=0.20; N=60 seeds; real ml-25m data, 3082 candidate items, 126
controversial-tagged, 75 KMeans genre regions):**

| Method | RAG ↓ | AUC-R ↓ | ReplayRet ↑ | ASD |
|---|---|---|---|---|
| Stationary | 1.00±0.00 | 1.36±0.33 | 1.00 | 0.0000 |
| PM variant (fast-memory only) | 1.00±0.00 | 1.22±0.18 | 1.01 | 0.0000 |
| RAPO-off@rep | 1.00±0.00 | 3.37±0.52 | 1.00 | 0.0000 |
| **RAPO** | 0.98±0.03 | **0.93±0.13** | 1.04 | 0.0000 |

**Same reporting note as Safety Gym applies here: RAG saturates near
1.000 and doesn't discriminate between conditions (200 steps makes at
least one harmful hit near-certain regardless of condition) — AUC-R is
the metric that actually shows the effect, and should be primary in the
paper for this domain too, with a note explaining why.**

**RAPO-off@rep (3.37) is dramatically worse than baseline (1.36)** — the
strongest causal-isolation signal of any of the three domains: removing
deformation exactly at Replay doesn't just restore baseline-level harm,
it more than doubles it, because scars accumulated during Exposure/Decay
apparently redirect the walk in ways that, once deformation is switched
off, concentrate it even more strongly in the harmful region than an
undeformed walk would be on its own. Worth a sentence of discussion in
the paper rather than treating it as a clean restoration to baseline
like the other two domains showed.

**One non-bug, worth a footnote so a reviewer doesn't flag it as one:**
PM-variant's `scar_gap` (1.01) is numerically higher than RAPO's (0.79),
which looks backwards at a glance. It isn't a bug — PM-variant's scar
channel H still accumulates internally regardless of w_H, it's just that
w_H=0 for PM-variant means H has zero *influence* on its own
deformation (only the trace channel G, via w_G, drives PM-variant's
suppression). Higher scar_gap there doesn't imply more suppression for
that condition.

**Utility cost is minimal across the board** (ReplayRet 0.96–1.04) —
unlike graph diffusion's ~40% utility hit, this domain's mixed
genre/cross-region kernel doesn't cost much navigation quality even
under strong deformation.

**ASD = 0.0000 exactly**, same verified (not assumed) construction as
Safety Gym.

---

## Project status (as of this wrap-up)

Three domains complete, real, and independently reproduced on the user's
machine (not just in the sandbox): **Graph Diffusion**, **Safety Gym**,
**MovieLens-25M**. A fourth (Reddit Cascades) was scoped but not
attempted — it needs real conversational data plus a toxicity signal
(Perspective API or a local classifier), a similar data-access lift to
MovieLens, and was deliberately not rushed given how much real
verification work each of the other three domains took.

**Every domain had at least one real bug that would have produced
plausible-looking but wrong numbers if not caught by actually running
the code and inspecting the output** — not by code review alone. If this
project is picked up again, the standing practice that caught all of
them: run on the real machine/data, check for exact-zero or
exact-saturated values (usually a sign a metric or a mechanism isn't
engaging at all, not that things are going great), check that ablations
move in the theoretically-expected direction, and re-run anything before
trusting it as reproducible.

**The paper draft itself has NOT been updated** with any of these
results — that was deliberately deferred per instruction. When it is
updated, each domain's row should carry the "report AUC-R as primary,
not RAG" caveat where it applies (Safety Gym, MovieLens), and the ASD
column needs a real decision (0.0000 for Safety Gym/MovieLens, N/A for
graph diffusion — not a single shared value across all four domains as
the original placeholder table implied).

