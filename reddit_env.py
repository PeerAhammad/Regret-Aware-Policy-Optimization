import json
import zipfile
import numpy as np
from collections import defaultdict, Counter

LOCAL_ZIP = "reddit-corpus-small.corpus.zip"
UTT_PATH = "reddit-corpus-small/utterances.json"
MIN_VOLUME = 2000


class RedditEnv:
    def __init__(self, zip_path=LOCAL_ZIP, min_volume=MIN_VOLUME, smoothing=1.0, verbose=True):
        if verbose:
            print("  loading real utterances...", flush=True)
        with zipfile.ZipFile(zip_path) as z:
            with z.open(UTT_PATH) as f:
                records = json.load(f)

        sr_counts = Counter()
        sr_very_neg = Counter()
        user_events = defaultdict(list)
        for r in records:
            meta = r.get("meta", {})
            sr = meta.get("subreddit")
            score = meta.get("score")
            user = r.get("user")
            ts = r.get("timestamp")
            if sr is None:
                continue
            sr_counts[sr] += 1
            if score is not None and score <= -5:
                sr_very_neg[sr] += 1
            if user and ts is not None and user not in ("[deleted]", "AutoModerator"):
                user_events[user].append((ts, sr))

        self.subreddits = sorted(sr_counts.keys())
        self.n_regions = len(self.subreddits)
        self.sr_to_idx = {sr: i for i, sr in enumerate(self.subreddits)}
        self.popularity = np.array([sr_counts[sr] for sr in self.subreddits], dtype=float)

        if verbose:
            print("  building empirical transition matrix from real author behavior...", flush=True)
        trans_counts = np.full((self.n_regions, self.n_regions), smoothing)
        for user, events in user_events.items():
            events.sort(key=lambda e: e[0])
            for i in range(len(events) - 1):
                a = self.sr_to_idx.get(events[i][1])
                b = self.sr_to_idx.get(events[i + 1][1])
                if a is not None and b is not None:
                    trans_counts[a, b] += 1
        self.P0 = trans_counts / trans_counts.sum(axis=1, keepdims=True)

        eligible = [sr for sr in self.subreddits if sr_counts[sr] >= min_volume]
        ranked = sorted(eligible, key=lambda sr: sr_very_neg[sr] / sr_counts[sr], reverse=True)
        self.r_harm = self.sr_to_idx[ranked[0]]
        self.r_safe = self.sr_to_idx[ranked[-1]]
        self.harm_subreddit_name = ranked[0]
        self.safe_subreddit_name = ranked[-1]
        self.c_max = 1.0

        if verbose:
            print(f"  {self.n_regions} regions (subreddits). r_harm={ranked[0]!r} "
                  f"(score<=-5 rate={100*sr_very_neg[ranked[0]]/sr_counts[ranked[0]]:.2f}%), "
                  f"r_safe={ranked[-1]!r} "
                  f"(score<=-5 rate={100*sr_very_neg[ranked[-1]]/sr_counts[ranked[-1]]:.2f}%)", flush=True)

    def base_kernel(self, region_idx):
        return self.P0[region_idx]

    def is_harmful(self, region_idx):
        return region_idx == self.r_harm

    def step(self, region_idx, psi_by_region, rng, deform_on):
        p0 = self.base_kernel(region_idx)
        if deform_on:
            weighted = p0 * psi_by_region
            s = weighted.sum()
            p_use = weighted / s if s > 0 else p0
        else:
            p_use = p0
        new_idx = int(rng.choice(self.n_regions, p=p_use))
        harm = self.c_max if self.is_harmful(new_idx) else 0.0
        return new_idx, harm
