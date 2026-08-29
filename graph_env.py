"""
Graph diffusion domain for RAPO/RSD — first real (non-toy) domain.

Environment: synthetic scale-free (Barabasi-Albert) or Erdos-Renyi graph.
Regions: Louvain communities.
Dynamics: Independent Cascade (IC) model. Each directed edge (u,v) has a
base activation probability p0_uv; if u is active at round t and v is not
yet infected, v becomes active at round t+1 with probability p0_uv (IC is
memoryless per-edge, standard formulation).

RAPO deformation for this domain: since IC edge activations are
independent Bernoulli trials (NOT a normalized distribution over
neighbors, unlike the region-Markov-chain synthetic testbed), the
deform-and-clip rule is applied WITHOUT the Z-normalization used in the
theory section's P(x'|x,a) = P0(x'|x,a)*psi(x')/Z(x,a). Instead:

    p_uv(G,H) = clip(p0_uv * psi_t(rho(v)), 0, 1)
    psi_t(r)  = clip(exp(-w_G*G_t^r - w_H*H_t^r), psi_min, 1)

This is a direct, honest adaptation of the same mechanism to a domain
where the base kernel isn't a per-source normalized distribution. It is
documented here explicitly so it can be scrutinized/challenged rather than
silently assumed equivalent to the Markov-chain case.
"""

import numpy as np
import networkx as nx


class GraphDiffusionEnv:
    def __init__(self, n_nodes=2000, graph_type="ba", ba_m=3, er_p=0.003,
                 base_prop_prob=0.06, seed=0):
        self.n_nodes = n_nodes
        rng_seed = seed
        if graph_type == "ba":
            self.G = nx.barabasi_albert_graph(n_nodes, ba_m, seed=rng_seed)
        elif graph_type == "er":
            self.G = nx.erdos_renyi_graph(n_nodes, er_p, seed=rng_seed)
        else:
            raise ValueError(graph_type)

        # Louvain communities -> region map
        communities = nx.algorithms.community.louvain_communities(self.G, seed=rng_seed)
        self.communities = communities
        self.n_regions = len(communities)
        self.region_of = np.empty(n_nodes, dtype=int)
        for r, nodes in enumerate(communities):
            for u in nodes:
                self.region_of[u] = r

        # base per-edge propagation probability, uniform for simplicity
        # (could be weighted by degree; kept simple and documented)
        self.base_prop_prob = base_prop_prob

        # precompute adjacency as list of neighbor arrays for speed
        self.neighbors = [np.array(list(self.G.neighbors(u)), dtype=int)
                           for u in range(n_nodes)]

        # designate harmful region = community containing the largest-degree
        # node (a plausible "influential misinformation hub")
        degrees = dict(self.G.degree())
        hub = max(degrees, key=degrees.get)
        self.r_harm = int(self.region_of[hub])

        # safe region for the utility/benign-content check: pick a
        # reasonably-sized community that is FAR (in graph hops) from the
        # harmful hub, so the benign seed lives in a genuinely different
        # part of the network rather than one the harmful cascade is
        # likely to sweep through anyway. (An earlier version picked
        # purely by community size and ended up choosing a community the
        # cascade spreads straight into, which conflates "collateral
        # suppression of an unrelated community" with "this community was
        # never really unrelated" -- not a fair utility-cost measurement.)
        dist_from_hub = nx.single_source_shortest_path_length(self.G, hub)
        min_size = 5
        candidates = []
        for i, c in enumerate(communities):
            if i == self.r_harm or len(c) < min_size:
                continue
            mean_dist = np.mean([dist_from_hub.get(u, 0) for u in c])
            candidates.append((mean_dist, i))
        self.r_safe = max(candidates)[1]  # farthest mean distance from hub

        self.seed_nodes = self._pick_seed_nodes(hub, k=5)
        self.benign_seed_nodes = self._pick_benign_seed_nodes(k=5)

    def _pick_benign_seed_nodes(self, k=5):
        """Seed set for a legitimate/benign cascade, drawn from r_safe
        community, used only to measure utility preservation (ReplayRet)."""
        nodes_in_safe = list(self.communities[self.r_safe])
        degrees = dict(self.G.degree())
        nodes_in_safe.sort(key=lambda x: -degrees[x])
        return np.array(nodes_in_safe[:k], dtype=int)

    def _pick_seed_nodes(self, hub, k=5):
        """Seed set = hub node + a few of its highest-degree neighbors,
        all within the harmful community where possible."""
        cand = [hub] + list(self.neighbors[hub])
        cand = [c for c in cand if self.region_of[c] == self.r_harm]
        degrees = dict(self.G.degree())
        cand = sorted(set(cand), key=lambda x: -degrees[x])
        if len(cand) < k:
            # fall back: fill with any nodes from the harmful community
            extra = [u for u in self.communities[self.r_harm] if u not in cand]
            cand = cand + extra
        return np.array(cand[:k], dtype=int)

    def run_ic_round(self, active_now, infected_mask, psi_by_region, rng):
        """One IC round. active_now: array of node ids active THIS round
        (will attempt to infect neighbors). infected_mask: boolean array,
        updated in place is avoided (functional style) — returns
        (newly_infected_ids, updated_infected_mask)."""
        newly = []
        for u in active_now:
            nbrs = self.neighbors[u]
            if len(nbrs) == 0:
                continue
            not_infected = nbrs[~infected_mask[nbrs]]
            if len(not_infected) == 0:
                continue
            psi_vals = psi_by_region[self.region_of[not_infected]]
            probs = np.clip(self.base_prop_prob * psi_vals, 0.0, 1.0)
            draws = rng.random(len(not_infected)) < probs
            newly.extend(not_infected[draws].tolist())
        newly = np.unique(np.array(newly, dtype=int)) if newly else np.array([], dtype=int)
        new_mask = infected_mask.copy()
        new_mask[newly] = True
        return newly, new_mask
