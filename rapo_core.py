"""
RAPO / RSD synthetic testbed — core implementation (rebuilt, identical
logic to the original). Fully deterministic mean-field dynamics.
"""

import numpy as np


class RegionEnv:
    def __init__(self, n_regions=10, harmful_region=0, base_stickiness=0.55,
                 stimulus_boost=0.35, c_max=1.0, rng=None):
        self.R = n_regions
        self.r_H = harmful_region
        self.base_stickiness = base_stickiness
        self.stimulus_boost = stimulus_boost
        self.c_max = c_max
        self.rng = rng if rng is not None else np.random.default_rng()

    def base_kernel_row(self, x, stimulus_on):
        p = np.full(self.R, (1 - self.base_stickiness) / (self.R - 1))
        p[x] = self.base_stickiness
        if stimulus_on:
            boost = self.stimulus_boost
            if x != self.r_H:
                take = p.copy()
                take[self.r_H] = 0
                take[x] = 0
                total_take = take.sum()
                if total_take > 0:
                    scale = boost / total_take
                    moved = take * scale
                    moved = np.minimum(moved, take)
                    p -= moved
                    p[self.r_H] += moved.sum()
            else:
                p[self.r_H] = min(1.0, p[self.r_H] + boost)
                rest_idx = [i for i in range(self.R) if i != self.r_H]
                rest_sum = p[rest_idx].sum()
                if rest_sum > 0 and p.sum() > 1:
                    excess = p.sum() - 1
                    p[rest_idx] -= (p[rest_idx] / rest_sum) * excess
        p = np.clip(p, 0, None)
        p = p / p.sum()
        return p

    def harm_signal(self, x):
        return self.c_max if x == self.r_H else 0.0


class TraceScarMemory:
    def __init__(self, n_regions, lam=0.15, alpha=1.0, delta=0.95, eta=0.5,
                 tau=0.3, w_G=0.0, w_H=1.2, psi_min=0.35):
        self.R = n_regions
        self.lam = lam
        self.alpha = alpha
        self.delta = delta
        self.eta = eta
        self.tau = tau
        self.w_G = w_G
        self.w_H = w_H
        self.psi_min = psi_min
        self.G = np.zeros(n_regions)
        self.H = np.zeros(n_regions)

    def update(self, harm_value, region_weights):
        self.G = (1 - self.lam) * self.G + self.alpha * harm_value * region_weights
        self.H = self.delta * self.H + self.eta * np.maximum(0, self.G - self.tau)

    def psi(self):
        raw = np.exp(-self.w_G * self.G - self.w_H * self.H)
        return np.clip(raw, self.psi_min, 1.0)

    def deform(self, p0_row):
        psi = self.psi()
        unnorm = p0_row * psi
        z = unnorm.sum()
        if z <= 0:
            return p0_row.copy()
        return unnorm / z

    def scar_gap(self, r_harm, r_safe):
        return self.H[r_harm] - self.H[r_safe]


def run_meanfield(env, memory, n_steps, stimulus_on, p0_dist, apply_deformation):
    R = env.R
    P = np.stack([env.base_kernel_row(x, stimulus_on) for x in range(R)])
    ps = np.empty((n_steps + 1, R))
    harms = np.empty(n_steps)
    ps[0] = p0_dist
    p = p0_dist.copy()
    for t in range(n_steps):
        if memory is not None and apply_deformation:
            psi = memory.psi()
            D = P * psi[None, :]
            D = D / D.sum(axis=1, keepdims=True)
        else:
            D = P
        p_next = p @ D
        harm = env.c_max * p_next[env.r_H]
        if memory is not None:
            region_weights = np.zeros(R)
            region_weights[env.r_H] = 1.0
            memory.update(harm, region_weights)
        harms[t] = harm
        ps[t + 1] = p_next
        p = p_next
    return ps, harms
