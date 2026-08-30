"""
Feasibility check for the proposed Reddit domain design (empirical
author-path transition kernel over subreddits) -- run BEFORE building the
full environment, since the whole design hinges on users actually posting
across multiple different subreddits. If most users only ever post in one
subreddit, a cross-subreddit transition kernel would be too sparse/
data-starved to be meaningful, and the design needs to change.

Also computes real candidate r_harm subreddits from the actual score data
(highest negative-score-rate among subreddits with enough volume to be
statistically reliable), and basic transition-matrix sparsity stats.

Assumes reddit-corpus-small.corpus.zip is already downloaded (from
inspect_reddit_corpus.py).
"""

import json
import zipfile
from collections import defaultdict, Counter

LOCAL_ZIP = "reddit-corpus-small.corpus.zip"
MIN_VOLUME = 2000  # minimum utterances for a subreddit to be an eligible
                     # r_harm/r_safe candidate -- avoids picking a small,
                     # noisy subreddit by chance (same lesson as
                     # MovieLens's region-mismatch bug: guard against
                     # small-sample flukes)


def load_utterances():
    with zipfile.ZipFile(LOCAL_ZIP) as z:
        with z.open("reddit-corpus-small/utterances.json") as f:
            return json.load(f)


def main():
    print("Loading real utterances...")
    records = load_utterances()
    print(f"  {len(records)} total\n")

    # --- Part 1: per-subreddit negative-score rate (real r_harm candidates) ---
    sr_counts = Counter()
    sr_neg = Counter()
    sr_very_neg = Counter()
    for r in records:
        meta = r.get("meta", {})
        sr = meta.get("subreddit")
        score = meta.get("score")
        if sr is None or score is None:
            continue
        sr_counts[sr] += 1
        if score < 0:
            sr_neg[sr] += 1
        if score <= -5:
            sr_very_neg[sr] += 1

    eligible = [sr for sr, c in sr_counts.items() if c >= MIN_VOLUME]
    print(f"--- Part 1: r_harm candidates (subreddits with >= {MIN_VOLUME} utterances) ---")
    print(f"{len(eligible)} eligible subreddits\n")
    ranked = sorted(eligible, key=lambda sr: sr_very_neg[sr] / sr_counts[sr], reverse=True)
    print("Top 15 by score<=-5 rate (real r_harm candidates):")
    for sr in ranked[:15]:
        rate = sr_very_neg[sr] / sr_counts[sr]
        print(f"  {sr:25s} n={sr_counts[sr]:5d}  score<=-5 rate={rate*100:.2f}%  "
              f"negative rate={100*sr_neg[sr]/sr_counts[sr]:.2f}%")
    print("\nBottom 10 by score<=-5 rate (real r_safe candidates):")
    for sr in ranked[-10:]:
        rate = sr_very_neg[sr] / sr_counts[sr]
        print(f"  {sr:25s} n={sr_counts[sr]:5d}  score<=-5 rate={rate*100:.2f}%  "
              f"negative rate={100*sr_neg[sr]/sr_counts[sr]:.2f}%")

    # --- Part 2: do users cross subreddits? (feasibility of the whole design) ---
    print("\n--- Part 2: author cross-subreddit behavior (feasibility check) ---")
    user_events = defaultdict(list)  # user -> list of (timestamp, subreddit)
    for r in records:
        meta = r.get("meta", {})
        sr = meta.get("subreddit")
        user = r.get("user")
        ts = r.get("timestamp")
        if sr and user and ts is not None and user != "[deleted]" and user != "AutoModerator":
            user_events[user].append((ts, sr))

    n_distinct_subs_per_user = []
    n_events_per_user = []
    for user, events in user_events.items():
        subs = set(sr for _, sr in events)
        n_distinct_subs_per_user.append(len(subs))
        n_events_per_user.append(len(events))

    import statistics
    print(f"Users (excluding deleted/AutoModerator): {len(user_events)}")
    print(f"Median events per user: {statistics.median(n_events_per_user)}")
    print(f"Mean distinct subreddits per user: {statistics.mean(n_distinct_subs_per_user):.2f}")
    print(f"Median distinct subreddits per user: {statistics.median(n_distinct_subs_per_user)}")
    multi_sub_users = sum(1 for n in n_distinct_subs_per_user if n >= 2)
    print(f"Users active in >=2 distinct subreddits: {multi_sub_users} "
          f"({100*multi_sub_users/len(user_events):.1f}%)")
    prolific_multi = sum(1 for n in n_distinct_subs_per_user if n >= 5)
    print(f"Users active in >=5 distinct subreddits: {prolific_multi} "
          f"({100*prolific_multi/len(user_events):.1f}%)")

    # --- Part 3: build the actual transition matrix and check sparsity ---
    print("\n--- Part 3: empirical subreddit-to-subreddit transition matrix ---")
    transitions = Counter()  # (sr_from, sr_to) -> count
    for user, events in user_events.items():
        events.sort(key=lambda e: e[0])
        for i in range(len(events) - 1):
            sr_from = events[i][1]
            sr_to = events[i + 1][1]
            transitions[(sr_from, sr_to)] += 1

    all_subs = sorted(sr_counts.keys())
    n_subs = len(all_subs)
    possible_pairs = n_subs * n_subs
    observed_pairs = len(transitions)
    cross_sub_transitions = sum(c for (a, b), c in transitions.items() if a != b)
    same_sub_transitions = sum(c for (a, b), c in transitions.items() if a == b)
    print(f"Total observed (from, to) transition pairs: {observed_pairs} "
          f"out of {possible_pairs} possible ({100*observed_pairs/possible_pairs:.1f}% dense)")
    print(f"Cross-subreddit transitions: {cross_sub_transitions}")
    print(f"Same-subreddit transitions: {same_sub_transitions}")
    print(f"Cross-subreddit fraction: {100*cross_sub_transitions/(cross_sub_transitions+same_sub_transitions):.1f}%")


if __name__ == "__main__":
    main()
