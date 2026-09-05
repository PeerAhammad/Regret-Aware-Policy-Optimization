# RAPO under Replay-Specific Deformation (RSD)

This repository contains the code and experimental pipelines for evaluating
**Replay-Aware Policy Optimization (RAPO)** under **Replay-Specific
Deformation (RSD)**.

The repository is organized so that the reported experimental results are
generated from executable pipelines rather than manually maintained result
tables. In particular, each domain has a single result-producing entry point,
and analysis scripts read the resulting JSON artifacts without hardcoded
experimental values.

> **Anonymous repository for double-blind review.**
>
> Author identities, affiliations, and other identifying information are
> intentionally omitted from this repository.

---

## Overview

RSD evaluates whether an environment-side transition mechanism can reduce
delayed harm when previously observed harmful states are reintroduced during
Replay.

RAPO maintains region-level trace and scar fields and uses them to deform the
Replay transition kernel. The experiments evaluate whether this deformation
reduces cumulative Replay harm while preserving the underlying policy and
measuring utility separately.

The experimental suite covers four domains:

- **MovieLens-25M** — item-item recommendation diffusion
- **Safety Gym** — spatial navigation
- **Reddit Cascades** — delayed downstream harm in information cascades
- **Graph Diffusion** — controlled diffusion on graphs

The common RSD schedule is

\[
(T_{\mathrm{exp}},T_{\mathrm{decay}},T_{\mathrm{rep}})
=(200,100,200).
\]

All experiments used a frozen base policy to isolate the environment-side
effect of RAPO.

---

## Repository Structure

```text
.
├── movielens_env.py
├── run_movielens.py
├── analyze_movielens.py
├── run_movielens_sensitivity.py
├── check_movielens.py
├── inspect_movielens_tags.py
├── requirements.txt
├── README.md
└── data/
    └── ml-25m/
        ├── movies.csv
        ├── ratings.csv
        └── tags.csv
````

The MovieLens pipeline is intentionally kept small:

* `movielens_env.py` constructs the recommendation environment.
* `run_movielens.py` is the **single source of experimental results** and
  writes `results_movielens.json`.
* `analyze_movielens.py` reads `results_movielens.json` and produces the
  reported statistics.
* `run_movielens_sensitivity.py` runs the hyperparameter sweeps by directly
  reusing `run_movielens.run_one_seed`, avoiding a second implementation of
  the experiment.
* `check_movielens.py` and `inspect_movielens_tags.py` provide read-only
  dataset and environment checks.

No reported headline value is manually entered into the analysis script.

---

## Requirements

Python 3.9+ is recommended.

Install the required packages with:

```bash
pip install pandas numpy scipy scikit-learn
```

---

## MovieLens-25M Dataset

Download the standard **MovieLens-25M** dataset and place the following files
at:

```text
data/ml-25m/movies.csv
data/ml-25m/ratings.csv
data/ml-25m/tags.csv
```

All three files are required.

`ratings.csv` contains approximately 25 million ratings and is processed in
chunks; the complete file does not need to be loaded into memory at once.

The controversial Replay stimulus is selected from `tags.csv` using the
`controversial` tag.

---

## MovieLens Environment

The rebuilt MovieLens environment uses a genuine state-dependent transition
process.

### Candidate items

The environment constructs its candidate pool from the 3,000 most-rated
movies.

### Transition kernel

Transitions are generated from an item-item similarity graph based on
genre vectors and cosine similarity. The next item therefore depends on the
current item:

$$
x_t \rightarrow x_{t+1}.
$$

This replaces the previous static recommendation list and provides the same
state-transition structure used by the other RSD domains.

### Regions

Items are partitioned into

```text
R = 75
```

genre-based regions using KMeans, subject to the available candidate count.

The actual number of regions is printed by the environment at runtime.

### Replay stimulus

A single controversial stimulus item is selected when the environment is
constructed.

The **same stimulus item** is used during both Exposure and Replay.

This is required by RSD: Replay must reintroduce the stimulus observed during
Exposure rather than sample a new controversial item at every step.

### Harm region

Harm is defined at the region level. The harmful region is tied directly to
the region containing the selected stimulus:

```python
r_harm == region_of[stimulus_item]
```

This prevents the harmful region from being inferred independently from the
global mode of controversial items.

---

## Running the MovieLens Experiment

From the repository root:

```bash
python run_movielens.py
```

This produces:

```text
results_movielens.json
```

Then run:

```bash
python analyze_movielens.py
```

The analysis script reads the generated result file and computes the reported
metrics.

There is no separate generator for a second result format: the experiment and
analysis therefore share a single result artifact.

---

## Experimental Conditions

The MovieLens experiment uses the same four conditions used throughout the
RSD evaluation:

1. **Baseline / Stationary**
2. **Policy-side memory variant**
3. **RAPO-off@Replay**
4. **RAPO**

`RAPO-off@Replay` is the causal-isolation condition. It disables transition
deformation during Replay while retaining the other trace, scar, and
attribution updates.

This tests whether the observed suppression depends specifically on
Replay-time transition deformation rather than merely on changing the
internal memory state.

---

## Reproducibility

Per-condition seeds are explicitly specified in the experiment code.

The implementation does **not** use Python's built-in `hash()` to construct
random seeds. Python string hashing is randomized between processes and can
therefore silently break reproducibility across machines.

The MovieLens pipeline instead uses an explicit deterministic seed mapping.

The utility evaluation also uses paired random-number streams so that the
utility comparison is performed on matched benign walks.

---

## Utility Measurement

Utility is measured using a single definition throughout the MovieLens
pipeline.

A held-out benign walk is evaluated using the raw popularity counts of the
visited items. RAPO and the comparison condition use paired random-number
streams for the comparison.

This avoids mixing incompatible utility definitions such as:

* normalization by the maximum popularity,
* raw popularity sums,
* independently sampled walks.

The utility statistic reported by the analysis script is therefore generated
from the same procedure used to produce the experimental results.

---

## Hyperparameter Sensitivity

The sensitivity sweep is run with:

```bash
python run_movielens_sensitivity.py
```

The sweep directly reuses:

```python
run_movielens.run_one_seed
```

rather than maintaining an independent implementation.

The reported sensitivity ranges include:

$$
w_H \in [1,6],
\qquad
\psi_{\min}\in[0.05,0.35],
\qquad
\lambda\in[0.10,0.20].
$$

The sweep is used to assess whether the reported operating point lies in a
stable region of the tested hyperparameter space.

---

## Sanity Checks

Before using a real-data run for reported results, verify the following.

### 1. Region count

The runtime configuration should report approximately:

```text
R = 75
```

unless the candidate data are degenerate or contain too few usable items.

### 2. Harm is reached

For deforming conditions, the Replay scar gap should be positive:

```text
scar_gap@replay > 0
```

If the scar gap is exactly zero for every seed, inspect:

```python
env.r_harm
env.region_of[env.stimulus_item]
```

These should agree.

Also verify that harmful-region states are actually visited during
Exposure.

### 3. Replay causal isolation

`RAPO-off@Replay` should remain close to the baseline on the relevant harm
measure.

A substantial unexplained deviation indicates that deformation is being
disabled incompletely or that another Replay-time component differs between
the conditions.

### 4. No NaNs

The analysis output should contain finite values for RAG, AUC-R, and utility.
A NaN result should be treated as a failed experiment rather than silently
converted or omitted.

---

## Important Implementation History

The current MovieLens implementation replaces an earlier experimental
pipeline.

The earlier implementation had several problems that materially affected the
validity of the experiment:

1. The recommendation policy returned a static top-10 list rather than a
   state-dependent transition.
2. A new controversial item was sampled at every step, so Replay did not
   reintroduce the same stimulus.
3. Utility was computed using multiple incompatible definitions.
4. A previously reported utility statistic could not be reproduced from the
   available analysis script.
5. The original item-level controversial-tag harm definition produced no
   harmful visits under the genre-based transition kernel.
6. The harmful region was initially computed independently from the actual
   Replay stimulus region.

These issues were identified and corrected in the current pipeline.

The present implementation therefore treats the executable experiment as the
source of truth and does not preserve the earlier headline statistics.

---

## First Real-Data Run

The rebuilt MovieLens pipeline has been verified end-to-end on a synthetic
dataset matching the expected MovieLens file schema.

**The full MovieLens-25M experiment must still be run on the real dataset before
using its resulting numbers in the paper.**

In particular, the following quantities must be generated from the real
25M-row dataset:

* RAG
* AUC-R
* ReplayRet
* scar-gap statistics
* sensitivity results

Do not copy earlier MovieLens headline numbers into the paper or repository
unless they are reproduced by:

```bash
python run_movielens.py
python analyze_movielens.py
```

---

## Computational Notes

The MovieLens rating file is processed in chunks to avoid requiring the entire
25M-row file in memory.

The principal computational costs come from:

* loading and processing the ratings data,
* constructing the item similarity structure,
* region assignment,
* repeated RSD Exposure/Decay/Replay trials.

The exact runtime depends on the hardware and dataset storage.

---

## Expected Experimental Output

The primary result file is:

```text
results_movielens.json
```

Analysis should be performed only from this generated file.

For a clean reproduction:

```bash
python run_movielens.py
python analyze_movielens.py
```

For sensitivity analysis:

```bash
python run_movielens_sensitivity.py
```

---

## Scope and Limitations

This repository evaluates the RSD mechanism under controlled experimental
conditions. The experiments do not establish that RAPO is appropriate for all
recommendation systems or production ranking systems.

In particular:

* the policy is frozen in the reported RSD experiments;
* the MovieLens environment is a constructed item-transition model rather
  than a production recommender;
* the controversial stimulus is selected using the available MovieLens
  tagging information;
* region-level harm depends on the chosen partition;
* systematic or subgroup-concentrated attribution errors are not fully
  evaluated;
* adversarial accumulation of false scars is not evaluated;
* recovery from a falsely scarred region after harmful evidence disappears is
  not fully characterized.

These limitations are part of the experimental scope and should not be
interpreted as claims about settings that were not evaluated.

