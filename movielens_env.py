"""
MovieLens-25M environment for RAPO/RSD -- rebuilt from scratch to fix
problems found reviewing the previous version:

  1. The old `MovieLensPolicy.recommend()` returned the SAME top-10
     popular movies on every single call, regardless of history -- there
     was no real x_t -> x_{t+1} transition, just a static list with one
     random "controversial" swap-in per step. Fixed here with a genuine
     item-to-item similarity kernel (genre-cosine, top-K neighbors), so
     the walk actually depends on the current item, like the graph
     diffusion and Safety Gym domains.
  2. The old code drew a FRESH random controversial movie every single
     step (both in Exposure and Replay), so RSD's "reintroduce the SAME
     stimulus" requirement was never actually satisfied. Fixed: one
     controversial seed item is chosen once per run and reused identically
     for both Exposure and Replay resets.
  3. Utility was computed with three different, incompatible formulas
     across different old scripts (normalized-by-max, raw popularity sum,
     and an untraceable third version behind a missing script). Fixed:
     ONE utility definition (raw popularity count of visited items),
     used everywhere, computed via paired random draws for ReplayRet.

Candidate pool is capped to the top TOP_N most-rated movies (default
3000) -- full 162K-item pairwise similarity is unnecessary compute here;
matches the "top-k approximation" scalability note in the paper draft.
Regions are Louvain-style KMeans clusters on genre one-hot vectors
(default 75, matching the paper's stated R=75 for this domain).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
          "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "IMAX"]


def genre_vector(genres_str):
    genres = set(genres_str.split("|")) if isinstance(genres_str, str) else set()
    v = np.array([1.0 if g in genres else 0.0 for g in GENRES])
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class MovieLensEnv:
    def __init__(self, data_dir="data/ml-25m", top_n=3000, n_regions=75,
                 top_k_neighbors=20, ratings_chunksize=1_000_000, seed=0, verbose=True):
        self.data_dir = Path(data_dir)
        self.top_k_neighbors = top_k_neighbors
        self.c_max = 1.0

        movies_df = pd.read_csv(self.data_dir / "movies.csv")
        movies_df = movies_df[movies_df["genres"].notna() &
                               (movies_df["genres"] != "(no genres listed)")]

        if verbose:
            print("  aggregating ratings (chunked)...", flush=True)
        counts = {}
        for chunk in pd.read_csv(self.data_dir / "ratings.csv", usecols=["movieId"],
                                  chunksize=ratings_chunksize):
            vc = chunk["movieId"].value_counts()
            for mid, c in vc.items():
                counts[mid] = counts.get(mid, 0) + int(c)
        pop = pd.Series(counts, name="count")

        merged = movies_df.set_index("movieId").join(pop, how="inner")
        merged = merged.sort_values("count", ascending=False).head(top_n)

        # controversial movies from tags.csv -- guarantee they're in the
        # candidate pool even if not in the top_n by popularity, otherwise
        # the stimulus set could end up empty
        tags_df = pd.read_csv(self.data_dir / "tags.csv")
        controversial_ids = set(
            tags_df.loc[tags_df["tag"].astype(str).str.lower() == "controversial", "movieId"]
        )
        missing_controversial = controversial_ids - set(merged.index)
        if missing_controversial:
            extra = movies_df.set_index("movieId").loc[
                movies_df.set_index("movieId").index.intersection(missing_controversial)]
            extra = extra.join(pop.reindex(extra.index).fillna(1).rename("count"))
            merged = pd.concat([merged, extra])

        merged = merged.reset_index().rename(columns={"index": "movieId"})
        if "movieId" not in merged.columns:
            merged = merged.rename(columns={merged.columns[0]: "movieId"})

        self.movies = merged.reset_index(drop=True)
        self.n_items = len(self.movies)
        self.movie_ids = self.movies["movieId"].to_numpy()
        self.id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.popularity = self.movies["count"].to_numpy(dtype=float)

        self.controversial_idx = set(
            self.id_to_idx[m] for m in controversial_ids if m in self.id_to_idx)
        if not self.controversial_idx:
            raise ValueError("No controversial-tagged movies found in candidate pool -- "
                              "check tags.csv has 'controversial' tags.")

        if verbose:
            print(f"  {self.n_items} candidate items, "
                  f"{len(self.controversial_idx)} controversial, clustering regions...", flush=True)

        vecs = np.stack([genre_vector(g) for g in self.movies["genres"]])
        self.item_vecs = vecs
        k = max(2, min(n_regions, self.n_items - 1))
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        self.region_of = km.fit_predict(vecs)
        self.n_regions = k
        self.region_centers = km.cluster_centers_

        # designate r_harm = the region the ACTUAL chosen stimulus item
        # lives in (not the mode across all controversial items -- tried
        # first, and it's a real bug: with many controversial items spread
        # across regions, the mode region can differ from the specific
        # stimulus item's own region, so a walk starting AT the stimulus
        # item explores a neighborhood that has no structural reason to
        # ever reach the independently-computed mode region. Confirmed via
        # direct reproduction: stimulus item's own region was 3, but the
        # mode-based r_harm was 2, and harm consequently never fired in
        # 200 steps despite the walk genuinely wandering through the
        # stimulus item's real neighborhood the whole time).
        # r_safe = region whose center is most dissimilar (cosine) from r_harm's.
        rng = np.random.default_rng(seed)
        self.stimulus_item = int(rng.choice(list(self.controversial_idx)))
        self.r_harm = int(self.region_of[self.stimulus_item])
        sims_to_harm = self.region_centers @ self.region_centers[self.r_harm]
        sims_to_harm[self.r_harm] = np.inf
        self.r_safe = int(np.argmin(sims_to_harm))

        if verbose:
            print("  building item-item neighbor kernel...", flush=True)
        self._build_neighbor_kernel()

        # benign/utility seed item (chosen after r_harm/r_safe are settled)
        safe_pool = [i for i in range(self.n_items)
                     if self.region_of[i] == self.r_safe and i not in self.controversial_idx]
        if not safe_pool:
            safe_pool = [i for i in range(self.n_items) if i not in self.controversial_idx]
        self.benign_item = int(safe_pool[np.argmax(self.popularity[safe_pool])])

    def _build_neighbor_kernel(self):
        """For each item, precompute a candidate neighbor set and a base
        distribution over it.

        NOT just top-K genre-cosine neighbors (tried first): regions are
        KMeans clusters on the SAME genre-vector space used for
        similarity, so an item's nearest genre-neighbors are, by
        construction, overwhelmingly in its OWN region -- that's what
        KMeans in that space optimizes for. Confirmed directly: with a
        pure genre-similarity neighbor set, the walk never left r_harm in
        200+200 steps, for any of 20 seeds, under any condition including
        RAPO (RAG/AUC-R exactly 1.000, zero variance across seeds).
        RAPO's deformation multiplies by psi(target region); if every
        candidate neighbor is already in the same region, psi is the same
        constant across the whole candidate set and cancels out under
        renormalization -- mathematically inert, however large the scars
        get.

        Fixed by mixing each item's candidate set: HALF genre-similar
        neighbors (same-region-biased, "people who liked X also like
        similar-genre Y") and HALF popularity-weighted candidates drawn
        from the FULL pool regardless of region (a "trending/popular
        elsewhere" channel) -- so crossing into/out of a region is a real,
        live possibility that deformation can actually suppress or allow.
        """
        sims = self.item_vecs @ self.item_vecs.T
        np.fill_diagonal(sims, -np.inf)
        K_total = min(self.top_k_neighbors, self.n_items - 1)
        K_genre = max(1, K_total // 2)
        K_cross = max(1, K_total - K_genre)

        genre_nbr_idx = np.argpartition(-sims, K_genre, axis=1)[:, :K_genre]
        genre_nbr_sims = np.take_along_axis(sims, genre_nbr_idx, axis=1)
        genre_nbr_sims = np.clip(genre_nbr_sims, 0, None) + 0.01

        pop_weights = self.popularity / self.popularity.sum()

        rng = np.random.default_rng(12345)  # fixed, not tied to experiment seeds --
        # this is a one-time structural property of the environment, built
        # once and shared across every condition/seed, same as the genre
        # kernel itself
        self.neighbor_idx = np.zeros((self.n_items, K_genre + K_cross), dtype=int)
        self.neighbor_probs = np.zeros((self.n_items, K_genre + K_cross))
        for i in range(self.n_items):
            cross_candidates = rng.choice(self.n_items, size=K_cross, replace=False, p=pop_weights)
            idx = np.concatenate([genre_nbr_idx[i], cross_candidates])
            genre_w = genre_nbr_sims[i]
            cross_w = pop_weights[cross_candidates] * pop_weights.sum() / pop_weights[cross_candidates].sum()
            cross_w = np.full(K_cross, genre_w.mean())  # comparable base weight to genre channel
            w = np.concatenate([genre_w, cross_w])
            self.neighbor_idx[i] = idx
            self.neighbor_probs[i] = w / w.sum()

    def base_kernel(self, item_idx):
        return self.neighbor_idx[item_idx], self.neighbor_probs[item_idx]

    def is_harmful(self, item_idx):
        """Harm is defined at the REGION level (landed in r_harm, the
        genre-cluster the stimulus item belongs to), not at the level of
        the exact controversial-tag identity.

        This was NOT the first version tried. An item-level definition
        ("is this specific movie tagged controversial") was tried first
        and produced EXACTLY ZERO harm events across every seed and every
        condition at real 25M-row scale (scar_gap stayed at precisely
        0.000, RAG/AUC-R came out as nan from a 0/0 guard). Root cause:
        "controversial" is a user tag, unrelated to genre; the transition
        kernel only navigates genre-cosine similarity, so once the walk
        leaves the stimulus item, its genre-neighbors are essentially
        never going to be one of the ~4% of candidates that happen to
        ALSO carry the controversial tag -- the two concepts aren't
        structurally linked. Region-level harm (matching how the graph
        diffusion and Safety Gym domains define harm as landing in a
        hazard community/cell, not a specific hazard identity) is the
        concept the kernel actually navigates through, so it registers.
        """
        return self.region_of[item_idx] == self.r_harm

    def step(self, item_idx, psi_by_region, rng, deform_on):
        nbrs, p = self.base_kernel(item_idx)
        if deform_on:
            psi = psi_by_region[self.region_of[nbrs]]
            weighted = p * psi
            s = weighted.sum()
            p_use = weighted / s if s > 0 else p
        else:
            p_use = p
        new_idx = nbrs[rng.choice(len(nbrs), p=p_use)]
        harm = self.c_max if self.is_harmful(new_idx) else 0.0
        return int(new_idx), harm