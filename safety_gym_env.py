"""
Safety Gym domain for RAPO/RSD -- lightweight grid-world stand-in for full
continuous-control Safety Gym (avoids a MuJoCo/safety-gymnasium dependency,
which is heavy to install and awkward on Windows/laptop GPUs -- consistent
with why this domain was picked as "lightest compute").

Design, explicit so it can be scrutinized:
  - N x N grid (default 8x8 = 64 cells), matching the paper's own
    "Regions: spatial grid (R=64)" description for this domain exactly --
    here a region IS a cell, one-to-one, not a coarser partition of finer
    physics.
  - A cluster of hazard cells sits between a fixed start cell and a fixed
    goal cell, so reaching the goal efficiently requires passing near or
    through the hazard zone (this is the analog of Safety Gym's "goal
    beyond hazard field" scenario).
  - Policy: a FIXED (never trained/updated) noisy goal-seeking heuristic --
    P0(x'|x) puts more mass on neighbor cells that reduce Manhattan
    distance to the goal, less on neutral/backward moves, "stay" always
    has residual mass. Because this policy is a fixed function used
    identically in every phase/condition, ASD (action-shift distance) is
    exactly 0 by construction here -- unlike the graph diffusion domain,
    this domain genuinely has a policy concept, so ASD is a real
    (trivially-verified) number rather than N/A.
  - RAPO deformation: accept-reject exactly as described in the paper's
    Implementation Details ("propose x'~P0, accept with probability
    psi_t(x'), else resample") -- this is the ONE domain where the
    accept-reject implementation note in the draft is used verbatim,
    rather than adapted.
  - Harm: c_max if the agent's proposed/accepted next cell is a hazard
    cell.
  - Utility metric (ReplayRet): fraction of episodes that reach the goal
    cell within the phase's step budget, RAPO vs. undeformed baseline --
    a real trade-off, since accept-reject rejection near hazard cells can
    detour the agent and cost it the step budget.
"""

import numpy as np


class GridSafetyEnv:
    def __init__(self, grid_size=8, hazard_cells=None, start=(0, 0), goal=(7, 7),
                 goal_bias=0.55, c_max=1.0):
        self.N = grid_size
        self.n_regions = grid_size * grid_size
        self.start = start
        self.goal = goal
        self.goal_bias = goal_bias
        self.c_max = c_max

        if hazard_cells is None:
            # a wall of hazard cells roughly between start and goal
            mid = grid_size // 2
            hazard_cells = [(mid, c) for c in range(1, grid_size - 1)]
        self.hazard_cells = set(hazard_cells)

        # designate a harmful region = a hazard cell near the middle of the
        # wall (most-traversed), safe region = a cell far from both hazard
        # and the direct start-goal path (a corner not on the diagonal)
        mid_idx = len(hazard_cells) // 2
        self.r_harm = self._cell_to_idx(hazard_cells[mid_idx])
        far_corner = (0, grid_size - 1)
        self.r_safe = self._cell_to_idx(far_corner)

    def _cell_to_idx(self, cell):
        r, c = cell
        return r * self.N + c

    def _idx_to_cell(self, idx):
        return divmod(idx, self.N)

    def neighbors(self, cell):
        r, c = cell
        cands = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [(rr, cc) for rr, cc in cands if 0 <= rr < self.N and 0 <= cc < self.N]

    def base_kernel(self, cell, goal_active):
        """P0(.|cell) over its neighbor list (including staying put)."""
        nbrs = self.neighbors(cell)
        if not goal_active:
            # neutral random walk, uniform over neighbors -- used during
            # Decay, when there's no active "reach the goal" stimulus
            p = np.ones(len(nbrs)) / len(nbrs)
            return nbrs, p
        gr, gc = self.goal
        dists = np.array([abs(rr - gr) + abs(cc - gc) for rr, cc in nbrs])
        cur_dist = abs(cell[0] - gr) + abs(cell[1] - gc)
        improves = dists < cur_dist
        p = np.where(improves, self.goal_bias, (1 - self.goal_bias))
        p = p / p.sum()
        return nbrs, p

    def step(self, cell, goal_active, psi_by_region, rng, deform_on, max_resamples=5):
        """One accept-reject step, following the paper's Implementation
        Details verbatim: propose x'~P0, accept with probability psi_t(x'),
        else resample (not just 'stay' -- redraw a fresh proposal, capped
        to avoid infinite loops in pathological cases; falls back to
        staying put only if every resample attempt is rejected)."""
        nbrs, p = self.base_kernel(cell, goal_active)
        new_cell = cell
        for _ in range(max_resamples):
            proposal = nbrs[rng.choice(len(nbrs), p=p)]
            if not deform_on:
                new_cell = proposal
                break
            r_idx = self._cell_to_idx(proposal)
            psi = psi_by_region[r_idx]
            if rng.random() < psi:
                new_cell = proposal
                break
            # else: rejected, loop and resample again
        harm = self.c_max if new_cell in self.hazard_cells else 0.0
        return new_cell, harm
