import json
import numpy as np
from rapo_core import TraceScarMemory
from reddit_env import RedditEnv

T_EXPOSURE = 200
T_DECAY = 100
T_REPLAY = 200
N_SEEDS = 60

MEM_KWARGS = dict(lam=0.1, alpha=1.0, delta=0.95, eta=0.5, tau=0.3, w_G=0.0,
                   w_H=6.0, psi_min=0.05)
PM_VARIANT_KWARGS = dict(lam=0.1, alpha=1.0, delta=0.95, eta=0.5, tau=0.3,
                          w_G=6.0, w_H=0.0, psi_min=0.05)

CONDITION_KWARGS = {"baseline": None, "pm_variant": PM_VARIANT_KWARGS,
                     "rapo_off_replay": MEM_KWARGS, "rapo": MEM_KWARGS}
CONDITION_SEED_OFFSET = {"baseline": 0, "pm_variant": 1, "rapo_off_replay": 2, "rapo": 3}


def run_one_seed(seed, condition, env):
    mem_kwargs = CONDITION_KWARGS[condition]
    R = env.n_regions
    memory = None if condition == "baseline" else TraceScarMemory(R, **mem_kwargs)
    rng = np.random.default_rng(seed * 1000 + CONDITION_SEED_OFFSET[condition])

    deform_exposure = condition != "baseline"
    deform_decay = condition != "baseline"
    deform_replay = condition in ("rapo", "pm_variant")

    def psi_vec(deform_on):
        return memory.psi() if (memory is not None and deform_on) else np.ones(R)

    def run_phase(n_steps, reseed, deform_on, start_idx):
        region = env.r_harm if reseed else start_idx
        harm_curve = np.zeros(n_steps)
        action_log = []
        for t in range(n_steps):
            psi_by_region = psi_vec(deform_on)
            p_base = env.base_kernel(region)
            action_log.append((region, p_base.copy()))
            new_region, harm = env.step(region, psi_by_region, rng, deform_on)
            if memory is not None:
                w = np.zeros(R)
                w[new_region] = 1.0
                memory.update(harm, w)
            region = new_region
            harm_curve[t] = harm
        return region, harm_curve, action_log

    end_region, harm_exp, log_exp = run_phase(T_EXPOSURE, True, deform_exposure, None)
    end_region, _, _ = run_phase(T_DECAY, False, deform_decay, end_region)
    scar_gap_at_replay = memory.scar_gap(env.r_harm, env.r_safe) if memory is not None else 0.0
    _, harm_rep, log_rep = run_phase(T_REPLAY, True, deform_replay, None)

    asd_diffs = []
    for (r1, p1), (r2, p2) in zip(log_exp, log_rep):
        if r1 == r2:
            asd_diffs.append(0.5 * np.abs(p1 - p2).sum())
    ASD = float(np.mean(asd_diffs)) if asd_diffs else 0.0

    def benign_utility(deform_on, pair_seed):
        local_rng = np.random.default_rng(pair_seed)
        region = env.r_safe
        psi_by_region = memory.psi() if (memory is not None and deform_on) else np.ones(R)
        total = 0.0
        for t in range(T_REPLAY):
            region, _ = env.step(region, psi_by_region, local_rng, deform_on)
            total += env.popularity[region]
        return total

    pair_seed = seed * 7919 + 31
    util_deformed = benign_utility(deform_replay, pair_seed)
    util_undeformed = benign_utility(False, pair_seed)

    RAG = harm_rep.max() / harm_exp.max() if harm_exp.max() > 0 else float('nan')
    AUC_R = harm_rep.sum() / harm_exp.sum() if harm_exp.sum() > 0 else float('nan')

    return dict(condition=condition, seed=seed, RAG=float(RAG), AUC_R=float(AUC_R),
                scar_gap_at_replay=float(scar_gap_at_replay), ASD=ASD,
                util_deformed=float(util_deformed), util_undeformed=float(util_undeformed),
                exp_harm_total=float(harm_exp.sum()), rep_harm_total=float(harm_rep.sum()))


def summarize(rows, key):
    vals = np.array([float(r[key]) for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))])
    if len(vals) == 0:
        return float('nan'), float('nan')
    mean = float(vals.mean())
    ci = float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean, ci


def main():
    print("Building environment (shared across all conditions/seeds)...", flush=True)
    env = RedditEnv()
    print(flush=True)

    results = {}
    for condition in ["baseline", "pm_variant", "rapo_off_replay", "rapo"]:
        rows = [run_one_seed(seed, condition, env) for seed in range(N_SEEDS)]
        rag_m, rag_ci = summarize(rows, "RAG")
        aucr_m, aucr_ci = summarize(rows, "AUC_R")
        asd_m, asd_ci = summarize(rows, "ASD")
        gap_m, gap_ci = summarize(rows, "scar_gap_at_replay")
        total_util_def = sum(r["util_deformed"] for r in rows)
        total_util_undef = sum(r["util_undeformed"] for r in rows)
        replay_ret = total_util_def / total_util_undef if total_util_undef > 0 else float('nan')
        results[condition] = dict(RAG_mean=rag_m, RAG_ci95=rag_ci,
                                   AUC_R_mean=aucr_m, AUC_R_ci95=aucr_ci,
                                   ASD_mean=asd_m, ASD_ci95=asd_ci,
                                   scar_gap_mean=gap_m, scar_gap_ci95=gap_ci,
                                   ReplayRet=replay_ret, raw=rows)
        print(f"[{condition:16s}] RAG = {rag_m:.3f} +/- {rag_ci:.3f} | "
              f"AUC-R = {aucr_m:.3f} +/- {aucr_ci:.3f} | "
              f"ReplayRet = {replay_ret:.3f} | ASD = {asd_m:.4f} | "
              f"scar_gap = {gap_m:.3f} +/- {gap_ci:.3f}", flush=True)

    results["_meta"] = dict(n_regions=env.n_regions, n_seeds=N_SEEDS,
                             T_exposure=T_EXPOSURE, T_decay=T_DECAY, T_replay=T_REPLAY,
                             mem_kwargs=MEM_KWARGS, pm_variant_kwargs=PM_VARIANT_KWARGS,
                             r_harm_subreddit=env.harm_subreddit_name,
                             r_safe_subreddit=env.safe_subreddit_name)
    with open("results_reddit.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results_reddit.json")
    return results


if __name__ == "__main__":
    main()
