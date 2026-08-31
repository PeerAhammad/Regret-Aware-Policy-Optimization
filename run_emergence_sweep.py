import json
import numpy as np
from rapo_core import RegionEnv, TraceScarMemory

N_REGIONS = 10
R_HARM = 0
R_SAFE = 1
T_ASYMPTOTIC = 4000
EPS_FRAC = 0.01

LAMBDAS = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4]


def delta_dist(R, x):
    p = np.zeros(R)
    p[x] = 1.0
    return p


def simulate_gap_trace(lam, n_steps):
    env = RegionEnv(n_regions=N_REGIONS, harmful_region=R_HARM)
    memory = TraceScarMemory(N_REGIONS, lam=lam, alpha=1.0, delta=0.95, eta=0.5,
                              tau=0.3, w_G=0.0, w_H=1.2, psi_min=0.35)
    p = delta_dist(N_REGIONS, R_SAFE)
    P0 = np.stack([env.base_kernel_row(x, stimulus_on=True) for x in range(N_REGIONS)])
    gaps = np.empty(n_steps)
    for t in range(n_steps):
        psi = memory.psi()
        D = P0 * psi[None, :]
        D = D / D.sum(axis=1, keepdims=True)
        p = p @ D
        harm = env.c_max * p[R_HARM]
        w = np.zeros(N_REGIONS)
        w[R_HARM] = 1.0
        memory.update(harm, w)
        gaps[t] = memory.scar_gap(R_HARM, R_SAFE)
    return gaps


def emergence_time(lam):
    gaps = simulate_gap_trace(lam, T_ASYMPTOTIC)
    gap_inf = gaps[-1]
    target = (1 - EPS_FRAC) * gap_inf
    idx = np.argmax(gaps >= target)
    if gaps[idx] < target:
        return None, float(gap_inf)
    return int(idx + 1), float(gap_inf)


def main():
    rows = []
    traces_for_plot = {}
    for lam in LAMBDAS:
        gaps = simulate_gap_trace(lam, 1000)  # keep full trace for plotting (first 1000 steps)
        traces_for_plot[lam] = gaps.tolist()
        t_emerge, gap_inf = emergence_time(lam)
        rows.append(dict(lam=lam, t_emerge=t_emerge, one_over_lambda=1.0 / lam,
                          asymptotic_gap=gap_inf))
        print(f"lambda = {lam:.3f}  ->  emergence time = {t_emerge:4d}  "
              f"(1/lambda = {1.0/lam:6.1f})  asymptotic_gap = {gap_inf:.4f}")

    valid = [r for r in rows if r["t_emerge"] is not None]
    x = np.array([r["one_over_lambda"] for r in valid])
    y = np.array([r["t_emerge"] for r in valid])
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = coef
    corr = np.corrcoef(x, y)[0, 1]
    print(f"\nLinear fit: t_emerge ~= {slope:.3f} * (1/lambda) + {intercept:.3f}  "
          f"(Pearson r = {corr:.4f})")

    with open("results_emergence_sweep.json", "w") as f:
        json.dump(dict(rows=rows, fit_slope=float(slope), fit_intercept=float(intercept),
                        pearson_r=float(corr), traces=traces_for_plot), f, indent=2)


if __name__ == "__main__":
    main()
