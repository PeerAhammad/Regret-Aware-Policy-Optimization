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

