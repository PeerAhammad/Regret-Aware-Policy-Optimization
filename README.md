# RAPO: Replay-Aware Policy Optimization under Replay-Specific Deformation

Anonymous code repository accompanying the paper submitted to ICLR.

This repository contains the implementations, experimental pipelines, analysis
scripts, plotting code, and figures used to evaluate **Replay-Aware Policy
Optimization (RAPO)** under **Replay-Specific Deformation (RSD)**.

The code is organized around executable experiment pipelines: reported results
are generated from experiment scripts and analyzed from their output files,
rather than maintained as manually entered values.

---

## Repository Structure

```text
.
├── figures/
│   ├── fig_odds_ratio.pdf
│   ├── fig_scar_evolution.pdf
│   ├── fig_sensitivity.pdf
│   └── fig_utility_safety.pdf
│
├── plotting_scripts/
│   ├── plot_fig_sensitivity.py
│   └── plot_fig_utility_safety.py
│
├── data/
│   └── ml-25m/
│       └── README.txt
│
├── movielens_env.py
├── run_movielens.py
├── analyze_movielens.py
├── run_movielens_sensitivity.py
├── check_movielens.py
└── inspect_movielens_tags.py
````

The repository also contains the implementations and experiment scripts for
the other evaluation domains described in the paper.

---

## Method

RSD evaluates a setting in which a harmful stimulus observed during an
Exposure phase is reintroduced during Replay.

RAPO maintains region-level trace and scar state and uses this state to deform
the Replay transition kernel. The experiments measure whether this
Replay-specific deformation reduces cumulative harm while preserving utility.

The common protocol consists of:

```text
Exposure → Decay → Replay
```

with the base policy held fixed during the RSD evaluation.

---

## Experimental Domains

The paper evaluates RSD across four domains:

* **MovieLens-25M** — item-transition recommendation diffusion
* **Safety Gym** — spatial navigation
* **Reddit Cascades** — information-cascade diffusion
* **Graph Diffusion** — controlled graph-based diffusion

The repository provides the code and analysis artifacts used for the reported
experiments.

---

# MovieLens-25M

The MovieLens implementation uses a state-dependent item-transition process.

Unlike a static recommendation list, the environment constructs an item-item
transition kernel from movie genres. The next item therefore depends on the
current item:

$$
x_t \rightarrow x_{t+1}.
$$

### Environment construction

The MovieLens environment:

1. constructs a candidate pool from the most-rated movies;
2. represents movies using their genre vectors;
3. constructs a cosine-similarity item-transition kernel;
4. partitions candidate items into genre-based regions;
5. selects one controversial stimulus item from the MovieLens tags;
6. reuses the same stimulus during Exposure and Replay;
7. defines the harmful region from the selected stimulus item's own region.

The default region count is:

$$
R=75.
$$

The actual value used by a run is printed by the environment.

---

## Dataset

MovieLens-25M is not included in this repository.

Download the standard MovieLens-25M dataset separately and place:

```text
data/ml-25m/movies.csv
data/ml-25m/ratings.csv
data/ml-25m/tags.csv
```

The `tags.csv` file is required because the Replay stimulus is selected using
the `controversial` tag.

The ratings file is processed in chunks, so the complete 25M-row file does not
need to be loaded into memory simultaneously.

See:

```text
data/ml-25m/README.txt
```

for dataset-specific information.

---

## Installation

The MovieLens pipeline requires:

```bash
pip install pandas numpy scipy scikit-learn
```

---

## Running the MovieLens Experiment

Run the experiment:

```bash
python run_movielens.py
```

This writes:

```text
results_movielens.json
```

Analyze the generated results:

```bash
python analyze_movielens.py
```

`analyze_movielens.py` reads the generated JSON file and does not contain
manually entered headline results.

---

## Experimental Conditions

The MovieLens pipeline evaluates the same four conditions used in the RSD
evaluation:

1. **Stationary / baseline**
2. **Policy-side memory variant**
3. **RAPO-off@Replay**
4. **RAPO**

The `RAPO-off@Replay` condition disables Replay-time transition deformation
while retaining the associated trace/scar updates. This provides a
causal-isolation comparison for the effect of Replay-time deformation.

---

## Reproducibility

Random seeds are explicitly specified for each experimental condition.

The MovieLens implementation does not use Python's built-in `hash()` to
construct random seeds. Python string hashing is randomized between processes,
which can otherwise produce different seeds across machines.

The implementation instead uses an explicit deterministic mapping from
conditions to seeds.

Utility comparisons use paired random-number streams so that the compared
benign walks receive matched randomness.

---

## Utility

MovieLens utility is computed using one definition throughout the pipeline.

Utility is evaluated from raw popularity counts visited on a held-out benign
walk using paired random-number streams for the comparison.

Earlier versions of the experiment used multiple incompatible utility
definitions. The current implementation removes those separate calculation
paths so that the experiment and its analysis use the same definition.

---

## Sensitivity Analysis

Run the MovieLens sensitivity sweep with:

```bash
python run_movielens_sensitivity.py
```

The sweep directly reuses the main experiment implementation rather than
reimplementing the experiment separately.

The principal ranges are:

$$
w_H\in[1,6],
\qquad
\psi_{\min}\in[0.05,0.35],
\qquad
\lambda\in[0.10,0.20].
$$

The resulting sweep is used to assess sensitivity to the principal RAPO
hyperparameters.

---

## Sanity Checks

Before using a real-data MovieLens run for reported results, verify:

### Region construction

The reported region count should be approximately:

```text
75
```

unless the available candidate set is degenerate.

### Stimulus and harmful region

The harmful region should satisfy:

```python
env.r_harm == env.region_of[env.stimulus_item]
```

### Harm activation

For deforming conditions, the Replay scar gap should be positive.

If:

```text
scar_gap@replay == 0
```

for every seed, inspect whether the harmful region is actually reached during
Exposure.

### Causal isolation

`RAPO-off@Replay` should remain close to the baseline on the corresponding
harm measure. A large unexplained deviation should be investigated before
interpreting the run.

### Numerical validity

RAG, AUC-R, utility, and scar-gap statistics should be finite. NaNs should be
treated as an experimental failure rather than silently replaced or omitted.

---

## Experimental History and Corrections

The current MovieLens implementation replaces an earlier pipeline.

Several issues were identified during reconstruction:

* the previous recommendation policy produced a static top-10 list rather than
  a state-dependent transition;
* a new controversial item was sampled at each step, so Replay did not
  reintroduce the Exposure stimulus;
* utility had been computed using incompatible definitions;
* a previously reported utility statistic could not be reproduced from the
  available analysis code;
* item-level controversial-tag harm produced no harmful visits under the
  genre-based transition kernel;
* the harmful region was initially computed independently of the actual
  stimulus item's region.

The current implementation addresses these issues directly.

In particular, the same stimulus item is used throughout Exposure and Replay,
the harmful region is tied to that stimulus, and the result-producing pipeline
has a single source of experimental output.

---

## Current MovieLens Reproduction Status

The rebuilt MovieLens pipeline has been tested end-to-end on a synthetic
dataset matching the expected MovieLens file schema.

The full MovieLens-25M experiment must be run on the real dataset before newly
generated MovieLens values are used as reported results.

The following quantities should come directly from the real-data run:

* RAG
* AUC-R
* ReplayRet
* AUC-R statistics
* scar-gap statistics
* hyperparameter sensitivity

The repository does not preserve earlier MovieLens headline values as
authoritative results. The executable pipeline is the source of truth.

---

## Figures

The `figures/` directory contains the figures used in the paper:

| Figure                   | Description                           |
| ------------------------ | ------------------------------------- |
| `fig_utility_safety.pdf` | Utility--safety tradeoff              |
| `fig_odds_ratio.pdf`     | Replay transition-odds contraction    |
| `fig_scar_evolution.pdf` | Scar-gap evolution and emergence rate |
| `fig_sensitivity.pdf`    | Hyperparameter sensitivity            |

The available plotting scripts are provided under:

```text
plotting_scripts/
```

---

## Reproducibility Checklist

For a clean MovieLens reproduction:

```bash
python run_movielens.py
python analyze_movielens.py
```

For sensitivity analysis:

```bash
python run_movielens_sensitivity.py
```

Before reporting the results, verify the sanity checks above and confirm that
the generated values correspond to the real MovieLens-25M dataset.

---

## Scope and Limitations

The experiments evaluate RAPO under the RSD protocol and should not be
interpreted as evidence for unrestricted deployment in production
recommendation systems.

In particular:

* the base policy is frozen during the RSD evaluation;
* MovieLens uses a constructed item-transition environment rather than a
  production recommender;
* the harmful region depends on the chosen region partition;
* systematic subgroup-concentrated attribution errors are not fully tested;
* adversarial accumulation of false scars is not evaluated;
* recovery dynamics after harmful evidence disappears are not fully
  characterized.

These cases are outside the current experimental scope.
