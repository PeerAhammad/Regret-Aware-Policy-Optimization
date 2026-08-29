"""
Small hyperparameter sweep for RAPO (only) on the graph diffusion domain,
using the CURRENT (fixed) attribution + paired-RNG code. Goal: find a
config that keeps RAG low while improving ReplayRet (currently 0.58 with
lam=0.3, w_H=2.5, psi_min=0.2) over the default. Not an exhaustive grid --
a handful of genuinely different points, checkpointed across batches.
"""
import json
import os
import numpy as np
from run_graph_diffusion import GraphDiffusionEnv, run_one_seed, N_NODES, GRAPH_TYPE, BA_M, BASE_PROP_PROB

GRID = [
    dict(lam=0.3, alpha=1.0, delta=0.9, eta=0.6, tau=0.05, w_G=0.0, w_H=1.5, psi_min=0.3),
    dict(lam=0.3, alpha=1.0, delta=0.9, eta=0.6, tau=0.05, w_G=0.0, w_H=1.5, psi_min=0.45),
    dict(lam=0.3, alpha=1.0, delta=0.9, eta=0.6, tau=0.05, w_G=0.0, w_H=2.5, psi_min=0.3),
    dict(lam=0.3, alpha=1.0, delta=0.9, eta=0.6, tau=0.05, w_G=0.0, w_H=2.5, psi_min=0.45),
    dict(lam=0.15, alpha=1.0, delta=0.9, eta=0.6, tau=0.05, w_G=0.0, w_H=2.5, psi_min=0.3),
    dict(lam=0.3, alpha=1.0, delta=0.95, eta=0.4, tau=0.1, w_G=0.0, w_H=2.5, psi_min=0.3),
]

N_SEEDS = 8
CHECKPOINT = "sweep2_checkpoint.json"


def run_config(cfg_idx, cfg, env_cache):
    from rapo_core import TraceScarMemory
    rows = []
    for seed in range(N_SEEDS):
        if seed not in env_cache:
            env_cache[seed] = GraphDiffusionEnv(n_nodes=N_NODES, graph_type=GRAPH_TYPE,
                                                  ba_m=BA_M, base_prop_prob=BASE_PROP_PROB, seed=seed)
        env = env_cache[seed]
        rng = np.random.default_rng(seed * 1000 + cfg_idx)
        row = run_one_seed_custom(seed, env, rng, cfg)
        rows.append(row)
    rag = np.mean([r["RAG"] for r in rows])
    ret = np.mean([r["ReplayRet"] for r in rows])
    return rag, ret, rows


def run_one_seed_custom(seed, env, rng, mem_kwargs):
    """Same logic as run_graph_diffusion.run_one_seed's 'rapo' condition,
    but with a custom mem_kwargs override for sweeping."""
    from rapo_core import TraceScarMemory
    import run_graph_diffusion as M
    R = env.n_regions
    memory = TraceScarMemory(R, **mem_kwargs)

    def psi_vec(deform_on):
        return memory.psi() if deform_on else np.ones(R)

    def run_phase(n_rounds, reseed, deform_on, infected_mask, active):
        if reseed:
            infected_mask = np.zeros(M.N_NODES, dtype=bool)
            infected_mask[env.seed_nodes] = True
            active = env.seed_nodes.copy()
        harm_curve = np.zeros(n_rounds)
        for t in range(n_rounds):
            psi_by_region = psi_vec(deform_on)
            newly, infected_mask = env.run_ic_round(active, infected_mask, psi_by_region, rng)
            if len(newly) > 0:
                region_counts = np.bincount(env.region_of[newly], minlength=R).astype(float)
                total = region_counts.sum()
                w = region_counts / total
                memory.update(float(total), w)
            else:
                memory.update(0.0, np.zeros(R))
            active = newly
            harm_curve[t] = len(newly)
            if len(active) == 0:
                harm_curve[t + 1:] = 0
                break
        return infected_mask, active, harm_curve

    infected_mask, active, harm_exp = run_phase(M.T_EXPOSURE, True, True, None, None)
    infected_mask, active, _ = run_phase(M.T_DECAY, False, True, infected_mask, active)

    psi_snapshot = memory.psi()

    def run_benign(deform_on, benign_seed):
        local_rng = np.random.default_rng(benign_seed)
        infected_mask = np.zeros(M.N_NODES, dtype=bool)
        infected_mask[env.benign_seed_nodes] = True
        active = env.benign_seed_nodes.copy()
        psi_by_region = psi_snapshot if deform_on else np.ones(R)
        for t in range(M.T_REPLAY):
            if len(active) == 0:
                break
            newly, infected_mask = env.run_ic_round(active, infected_mask, psi_by_region, local_rng)
            active = newly
        return infected_mask.sum()

    pair_seed = seed * 7919 + 17
    benign_deformed = run_benign(True, pair_seed)
    benign_undeformed = run_benign(False, pair_seed)
    replay_ret = benign_deformed / benign_undeformed if benign_undeformed > 0 else float('nan')

    infected_mask, active, harm_rep = run_phase(M.T_REPLAY, True, True, None, None)

    RAG = harm_rep.max() / harm_exp.max() if harm_exp.max() > 0 else float('nan')
    return dict(seed=seed, RAG=float(RAG), ReplayRet=float(replay_ret))


def main():
    env_cache = {}
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            summary = json.load(f)
    else:
        summary = []

    done = {tuple(sorted(r["cfg"].items())) for r in summary}
    for i, cfg in enumerate(GRID):
        key = tuple(sorted(cfg.items()))
        if key in done:
            continue
        rag, ret, rows = run_config(i, cfg, env_cache)
        summary.append(dict(cfg=cfg, RAG_mean=rag, ReplayRet_mean=ret))
        print(f"cfg {i}: {cfg} -> RAG={rag:.3f}, ReplayRet={ret:.3f}", flush=True)
        with open(CHECKPOINT, "w") as f:
            json.dump(summary, f, indent=2)

    print("\n--- sorted by ReplayRet (desc), among RAG < 0.45 ---")
    good = [r for r in summary if r["RAG_mean"] < 0.45]
    for r in sorted(good, key=lambda r: -r["ReplayRet_mean"]):
        print(r)


if __name__ == "__main__":
    main()
