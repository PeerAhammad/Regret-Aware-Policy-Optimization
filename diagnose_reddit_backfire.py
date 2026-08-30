"""
Diagnostic for the wrong-direction finding: RAPO showed HIGHER cumulative
harm than baseline (significant in some configs) on real Reddit data.
Rather than guess, trace actual per-step trajectories for a few seeds
under baseline vs. rapo (using cfg2 from the sweep, the most significant
wrong-direction result: w_H=6.0, psi_min=0.05, lam=0.15) and print exactly
where the walk goes, to see the mechanism directly instead of inferring it
from an aggregate statistic.
"""

import numpy as np
from reddit_env import RedditEnv
from rapo_core import TraceScarMemory
import run_reddit as R

CFG = dict(lam=0.15, alpha=1.0, delta=0.95, eta=0.5, tau=0.3, w_G=0.0, w_H=6.0, psi_min=0.05)
N_TRACE_STEPS = 30  # just the start of Exposure, where the divergence should be clearest


def trace_condition(seed, condition, env, cfg):
    mem_kwargs = None if condition == "baseline" else cfg
    R_ = env.n_regions
    memory = None if condition == "baseline" else TraceScarMemory(R_, **mem_kwargs)
    rng = np.random.default_rng(seed * 1000 + {"baseline": 0, "rapo": 3}[condition])
    deform = condition != "baseline"

    region = env.r_harm
    trace = [region]
    for t in range(N_TRACE_STEPS):
        psi = memory.psi() if (memory is not None and deform) else np.ones(R_)
        new_region, harm = env.step(region, psi, rng, deform)
        if memory is not None:
            w = np.zeros(R_)
            w[new_region] = 1.0
            memory.update(harm, w)
        region = new_region
        trace.append(region)
    return [env.subreddits[i] for i in trace]


def trace_full_rsd(seed, condition, env, cfg):
    """Trace the ACTUAL full Exposure->Decay->Replay sequence (matching
    run_reddit.run_one_seed exactly), reporting harm counts per phase --
    not just an isolated Exposure-like snippet, which the first version
    of this diagnostic used and which (misleadingly) showed rapo doing
    BETTER in the first 30 steps, the opposite of the full-run sweep
    finding. The real divergence must be later in Exposure or in Replay
    specifically, after scars have accumulated through Decay."""
    mem_kwargs = None if condition == "baseline" else cfg
    R_ = env.n_regions
    memory = None if condition == "baseline" else TraceScarMemory(R_, **mem_kwargs)
    rng = np.random.default_rng(seed * 1000 + {"baseline": 0, "rapo": 3}[condition])
    deform = condition != "baseline"

    def run_phase(n_steps, reseed, region):
        if reseed:
            region = env.r_harm
        trace = [region]
        harm_hits = 0
        for t in range(n_steps):
            psi = memory.psi() if (memory is not None and deform) else np.ones(R_)
            new_region, harm = env.step(region, psi, rng, deform)
            if memory is not None:
                w = np.zeros(R_)
                w[new_region] = 1.0
                memory.update(harm, w)
            region = new_region
            trace.append(region)
            harm_hits += int(harm > 0)
        return region, trace, harm_hits

    region, exp_trace, exp_harm = run_phase(200, True, env.r_harm)
    region, decay_trace, decay_harm = run_phase(100, False, region)
    scar_gap = memory.scar_gap(env.r_harm, env.r_safe) if memory is not None else 0.0
    psi_r_harm_at_replay_start = memory.psi()[env.r_harm] if memory is not None else 1.0
    region, rep_trace, rep_harm = run_phase(200, True, None)

    return dict(exp_harm=exp_harm, decay_harm=decay_harm, rep_harm=rep_harm,
                scar_gap=scar_gap, psi_r_harm_at_replay=psi_r_harm_at_replay_start,
                exp_trace_tail=exp_trace[-20:], rep_trace_head=rep_trace[:20])


def main():
    print("Building environment...", flush=True)
    env = RedditEnv()
    print(f"r_harm = {env.harm_subreddit_name!r}\n")

    p0_row = env.base_kernel(env.r_harm)
    self_prob = p0_row[env.r_harm]
    top5_idx = np.argsort(-p0_row)[:6]
    print(f"P0(.|r_harm) self-transition probability: {self_prob:.4f}")
    print("Top 6 destinations from r_harm under P0 (undeformed):")
    for i in top5_idx:
        marker = " <- SELF" if i == env.r_harm else ""
        print(f"  {env.subreddits[i]:25s} p={p0_row[i]:.4f}{marker}")

    print("\n--- Full Exposure -> Decay -> Replay trace (this is what AUC-R actually measures) ---")
    for seed in range(5):
        print(f"\nSeed {seed}:")
        b = trace_full_rsd(seed, "baseline", env, CFG)
        r = trace_full_rsd(seed, "rapo", env, CFG)
        print(f"  baseline: exp_harm={b['exp_harm']:3d}  decay_harm={b['decay_harm']:3d}  "
              f"rep_harm={b['rep_harm']:3d}  AUC-R={b['rep_harm']/max(b['exp_harm'],1e-9):.3f}")
        print(f"  rapo:     exp_harm={r['exp_harm']:3d}  decay_harm={r['decay_harm']:3d}  "
              f"rep_harm={r['rep_harm']:3d}  AUC-R={r['rep_harm']/max(r['exp_harm'],1e-9):.3f}  "
              f"scar_gap={r['scar_gap']:.3f}  psi(r_harm)@replay_start={r['psi_r_harm_at_replay']:.4f}")
        if r['rep_harm'] > b['rep_harm'] or r['rep_harm']/max(r['exp_harm'],1e-9) > b['rep_harm']/max(b['exp_harm'],1e-9):
            print(f"    ^ rapo AUC-R >= baseline's here. rapo exp_trace tail: {r['exp_trace_tail']}")
            print(f"      rapo replay trace head: {r['rep_trace_head']}")


if __name__ == "__main__":
    main()
