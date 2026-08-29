"""
RAPO / RSD synthetic testbed — core implementation.

This is a MINIMAL, honest testbed used to validate the theoretical claims
of the RAPO draft (Theorem: replay impossibility under stationary dynamics;
Lemma: odds contraction; Proposition: scar gap emergence) before any of the
four real-world domains (MovieLens, Safety Gym, Reddit, graph diffusion)
are touched. All numbers produced by this code are real simulation outputs
— nothing here is a stand-in for a result that hasn't been run.

Design choices (documented so they can be challenged/changed):
  - State space = R discrete "regions" (no separate action space). RSD's
    theory concerns x_{t+1} | x_t under a base kernel P0; folding in a
    frozen policy over actions vs. defining P0 directly over regions is
    mathematically equivalent for the purpose of testing Theorem 1 and
    the RAPO mechanism, and it removes a nuisance dimension that has
    nothing to do with what's being tested (environment-side memory).
  - A single harmful region r_H=0 and (R-1) safe regions.
  - "Stimulus" z in {0,1} boosts P0's mass toward r_H when active.
  - Harm signal tilde_c_t = c_max * 1[x_t == r_H] (no synthetic delay
    baked in beyond what emerges naturally from trace/scar decay — delay
    D matters for attribution in the real domains, not for this minimal
    mechanism check).
"""

import numpy as np


class RegionEnv:
    """Markov chain over R regions with a stimulus that biases transitions
    toward the harmful region r_H=0 when active."""

    def __init__(self, n_regions=10, harmful_region=0, base_stickiness=0.55,
                 stimulus_boost=0.35, c_max=1.0, rng=None):
        self.R = n_regions
        self.r_H = harmful_region
        self.base_stickiness = base_stickiness
        self.stimulus_boost = stimulus_boost
        self.c_max = c_max
        self.rng = rng if rng is not None else np.random.default_rng()

    def base_kernel_row(self, x, stimulus_on):
        """Return P0(. | x) as a length-R probability vector."""
        p = np.full(self.R, (1 - self.base_stickiness) / (self.R - 1))
        p[x] = self.base_stickiness
        if stimulus_on:
            # Redirect probability mass toward r_H, taken proportionally
            # from all other regions (never touching x's own self-mass
            # in a way that breaks normalization).
            boost = self.stimulus_boost
            if x != self.r_H:
                take = p.copy()
                take[self.r_H] = 0
                take[x] = 0
                total_take = take.sum()
                if total_take > 0:
                    scale = boost / total_take
                    moved = take * scale
                    moved = np.minimum(moved, take)  # can't take more than exists
                    p -= moved
                    p[self.r_H] += moved.sum()
            else:
                p[self.r_H] = min(1.0, p[self.r_H] + boost)
                # renormalize by shrinking the rest proportionally
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
    """Implements Eq. (trace_update), (scar_update) and the deformation
    weight psi_t(x') from the RAPO draft, region-indexed."""

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
        """region_weights: length-R attribution weights w_t(r), sum to 1
        (here we use a one-hot / uniform-on-visited-region attribution,
        the 'Uniform' strategy from the paper)."""
        self.G = (1 - self.lam) * self.G + self.alpha * harm_value * region_weights
        self.H = self.delta * self.H + self.eta * np.maximum(0, self.G - self.tau)

    def psi(self):
        raw = np.exp(-self.w_G * self.G - self.w_H * self.H)
        return np.clip(raw, self.psi_min, 1.0)

    def deform(self, p0_row):
        """Given base kernel row P0(.|x), return deformed distribution."""
        psi = self.psi()
        unnorm = p0_row * psi
        z = unnorm.sum()
        if z <= 0:
            return p0_row.copy()
        return unnorm / z

    def scar_gap(self, r_harm, r_safe):
        return self.H[r_harm] - self.H[r_safe]


def run_meanfield(env, memory, n_steps, stimulus_on, p0_dist, apply_deformation):
    """Exact (noise-free) evolution of the region *distribution* p_t, rather
    than a single sampled trajectory. Because only r_H ever produces a
    nonzero harm signal in this environment, and attribution is one-hot on
    whichever region harm occurred in, the expected trace/scar update
    collapses to updating a single scalar channel (r_H); all other regions'
    G,H remain exactly 0 by construction. This removes Monte Carlo noise
    from the headline numbers, which otherwise swamps the effect we're
    trying to measure over a 200-step window.

    Returns: array of shape (n_steps+1, R) of p_t, and array of length
    n_steps of expected harm E[c_t] = c_max * p_t[r_H].
    """
    R = env.R
    P = np.stack([env.base_kernel_row(x, stimulus_on) for x in range(R)])  # R x R
    ps = np.empty((n_steps + 1, R))
    harms = np.empty(n_steps)
    ps[0] = p0_dist
    p = p0_dist.copy()
    for t in range(n_steps):
        if memory is not None and apply_deformation:
            psi = memory.psi()  # length-R vector, only r_H entry != 1 here
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


def run_chain(env, memory, n_steps, stimulus_on, x0, apply_deformation, rng):
    """Simulate one trajectory of the region chain for n_steps.
    Returns array of visited states (length n_steps+1) and per-step harm.
    memory is updated online (traces/scars evolve during this run) unless
    apply_deformation is False AND memory is None (pure stationary baseline).
    """
    xs = np.empty(n_steps + 1, dtype=int)
    harms = np.empty(n_steps)
    xs[0] = x0
    x = x0
    for t in range(n_steps):
        p0 = env.base_kernel_row(x, stimulus_on)
        if memory is not None and apply_deformation:
            p = memory.deform(p0)
        else:
            p = p0
        x_next = rng.choice(env.R, p=p)
        h = env.harm_signal(x_next)
        harms[t] = h
        if memory is not None:
            region_weights = np.zeros(env.R)
            region_weights[x_next] = 1.0
            memory.update(h, region_weights)
        xs[t + 1] = x_next
        x = x_next
    return xs, harms
