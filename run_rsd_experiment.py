"""
Runs the Replay Suppression Diagnostic (RSD) on the synthetic RegionEnv
using EXACT mean-field dynamics (the region probability distribution is
evolved deterministically each step; no Monte Carlo noise). This is valid
here because harm and attribution are both concentrated on a single region
r_H, so the expected trace/scar update has closed form and matches the
mean of the stochastic single-trajectory process exactly.

Conditions:
  1. baseline          -> psi == 1 always                 (tests Theorem 1)
  2. rapo               -> deformation on throughout        (main result)
  3. rapo_off_replay    -> deformation on except in Replay  (causal ablation)

All numbers are exact simulation outputs given the stated hyperparameters
(MEM_KWARGS below) — none are placeholders.
"""

import json
import numpy as np
from rapo_core import RegionEnv, TraceScarMemory, run_meanfield

N_REGIONS = 10
R_HARM = 0
R_SAFE = 1

T_EXPOSURE = 200
T_DECAY = 100
T_REPLAY = 200

MEM_KWARGS = dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3,
                   w_G=0.0, w_H=1.2, psi_min=0.35)


def delta_dist(R, x):
    p = np.zeros(R)
    p[x] = 1.0
    return p


def run_condition(condition):
    env = RegionEnv(n_regions=N_REGIONS, harmful_region=R_HARM)
    memory = None if condition == "baseline" else TraceScarMemory(N_REGIONS, **MEM_KWARGS)

    x_star = R_SAFE
    p0 = delta_dist(N_REGIONS, x_star)

    exposure_deform = condition != "baseline"
    ps_exp, harm_exp = run_meanfield(env, memory, T_EXPOSURE, stimulus_on=True,
                                      p0_dist=p0, apply_deformation=exposure_deform)

    decay_deform = condition != "baseline"
    ps_decay, _ = run_meanfield(env, memory, T_DECAY, stimulus_on=False,
                                 p0_dist=ps_exp[-1], apply_deformation=decay_deform)

    scar_gap_at_replay = memory.scar_gap(R_HARM, R_SAFE) if memory is not None else 0.0

    replay_deform = {"baseline": False, "rapo": True, "rapo_off_replay": False}[condition]

    # single-step odds check (Lemma 1 is a single-decision statement at a
    # FIXED scar gap; averaging over the whole replay window confounds it
    # with the scars continuing to evolve during replay itself)
    p0_row_at_xstar = env.base_kernel_row(x_star, stimulus_on=True)
    p0_single_step = p0_row_at_xstar[R_HARM]
    if memory is not None and replay_deform:
        psi_at_replay_start = memory.psi()
        deformed_row = p0_row_at_xstar * psi_at_replay_start
        deformed_row = deformed_row / deformed_row.sum()
        p_single_step = deformed_row[R_HARM]
    else:
        p_single_step = p0_single_step
    ps_rep, harm_rep = run_meanfield(env, memory, T_REPLAY, stimulus_on=True,
                                      p0_dist=delta_dist(N_REGIONS, x_star),
                                      apply_deformation=replay_deform)

    reach_exp = ps_exp[1:, R_HARM]
    reach_rep = ps_rep[1:, R_HARM]

    RAG = reach_rep.max() / reach_exp.max()
    AUC_R = reach_rep.sum() / reach_exp.sum()
    mean_reach_rep = reach_rep.mean()

    return dict(condition=condition, RAG=float(RAG), AUC_R=float(AUC_R),
                scar_gap_at_replay=float(scar_gap_at_replay),
                mean_reach_replay=float(mean_reach_rep),
                p0_single_step=float(p0_single_step), p_single_step=float(p_single_step),
                reach_exp_curve=reach_exp.tolist(), reach_rep_curve=reach_rep.tolist())


def odds(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return p / (1 - p)


def main():
    results = {c: run_condition(c) for c in ["baseline", "rapo", "rapo_off_replay"]}

    p0_replay_mean = results["baseline"]["mean_reach_replay"]  # undeformed reference
    for c, r in results.items():
        r["odds_ratio_vs_undeformed"] = odds(r["mean_reach_replay"]) / odds(p0_replay_mean)

    for c in ["baseline", "rapo", "rapo_off_replay"]:
        r = results[c]
        print(f"[{c:16s}] RAG = {r['RAG']:.3f} | AUC-R = {r['AUC_R']:.3f} | "
              f"scar_gap@replay = {r['scar_gap_at_replay']:.3f} | "
              f"mean_reach_replay = {r['mean_reach_replay']:.3f} | "
              f"odds_ratio(vs undeformed) = {r['odds_ratio_vs_undeformed']:.3f}")

    w_H = MEM_KWARGS["w_H"]
    delta_measured = results["rapo"]["scar_gap_at_replay"]
    theory_contraction = float(np.exp(-w_H * delta_measured))
    r_rapo = results["rapo"]
    measured_singlestep = odds(r_rapo["p_single_step"]) / odds(r_rapo["p0_single_step"])
    print(f"\nLemma check (single-step odds contraction at Replay's first step):\n"
          f"  measured  = {measured_singlestep:.3f}\n"
          f"  theoretical e^(-w_H*Delta) = {theory_contraction:.3f}  "
          f"(Delta={delta_measured:.3f}, w_H={w_H})")
    results["lemma_check"] = dict(measured_singlestep=measured_singlestep,
                                   theoretical=theory_contraction, delta=delta_measured, w_H=w_H)

    results["_meta"] = dict(n_regions=N_REGIONS, T_exposure=T_EXPOSURE, T_decay=T_DECAY,
                             T_replay=T_REPLAY, mem_kwargs=MEM_KWARGS,
                             theoretical_odds_contraction=theory_contraction,
                             note="Exact mean-field dynamics, deterministic, no Monte Carlo noise.")

    with open("results_rsd_main.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
