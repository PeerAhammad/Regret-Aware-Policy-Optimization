"""
Downloads and inspects the REAL reddit-corpus-small dataset directly.

v2: fixed after the first version wrongly assumed line-delimited JSON.
Confirmed from actual output on the real file: utterances.json is ONE
JSON array (not JSONL), and the real field names are:
  id, user, root, reply_to, timestamp, text,
  meta: {score, subreddit, top_level_comment, gilded, stickied,
         permalink, author_flair_text, retrieved_on, gildings}
(docs said "speaker"/"conversation_id" -- actual files say "user"/"root".
Going with what the real file says.)
"""

import json
import zipfile
import urllib.request
from collections import Counter
import statistics

URL = "https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/reddit-corpus-small.corpus.zip"
LOCAL_ZIP = "reddit-corpus-small.corpus.zip"


def download():
    import os
    if os.path.exists(LOCAL_ZIP):
        print(f"{LOCAL_ZIP} already exists, skipping download.")
        return
    print(f"Downloading {URL} (~38MB)...")
    urllib.request.urlretrieve(URL, LOCAL_ZIP)
    print("Done.")


def inspect():
    with zipfile.ZipFile(LOCAL_ZIP) as z:
        names = z.namelist()
        print(f"Files in zip ({len(names)}):")
        for n in names:
            print(" ", n)

        utt_file = "reddit-corpus-small/utterances.json"
        print(f"\nLoading {utt_file} (whole file, single JSON array)...")
        with z.open(utt_file) as f:
            records = json.load(f)

        print(f"Type: {type(records)}, count: {len(records)}")
        print("\nFirst record:")
        print(json.dumps(records[0], indent=2))

        print("\n--- Full-corpus stats ---")
        subreddits = Counter()
        has_reply_to = 0
        has_score = 0
        score_values = []
        missing_meta = 0

        for obj in records:
            meta = obj.get("meta", {})
            if not meta:
                missing_meta += 1
            sr = meta.get("subreddit")
            if sr:
                subreddits[sr] += 1
            if obj.get("reply_to") is not None:
                has_reply_to += 1
            if "score" in meta and meta["score"] is not None:
                has_score += 1
                try:
                    score_values.append(float(meta["score"]))
                except (TypeError, ValueError):
                    pass

        total = len(records)
        print(f"Total utterances: {total}")
        print(f"  with reply_to set (i.e. a reply, not a top-level post): "
              f"{has_reply_to} ({100*has_reply_to/total:.1f}%)")
        print(f"  with score field: {has_score} ({100*has_score/total:.1f}%)")
        print(f"  missing meta entirely: {missing_meta}")
        if score_values:
            print(f"  score stats: min={min(score_values):.0f}, max={max(score_values):.0f}, "
                  f"mean={statistics.mean(score_values):.2f}, "
                  f"median={statistics.median(score_values):.0f}")
            neg = sum(1 for s in score_values if s < 0)
            very_neg = sum(1 for s in score_values if s <= -5)
            print(f"  negative-score fraction: {neg}/{len(score_values)} = {100*neg/len(score_values):.1f}%")
            print(f"  score <= -5 fraction: {very_neg}/{len(score_values)} = {100*very_neg/len(score_values):.1f}%")

        print(f"\n  {len(subreddits)} distinct subreddits present.")
        print(f"  top 15 by utterance count:")
        for sr, c in subreddits.most_common(15):
            print(f"    {sr}: {c}")
        print(f"  bottom 5 by utterance count (of those present):")
        for sr, c in subreddits.most_common()[-5:]:
            print(f"    {sr}: {c}")

        # quick check: does reply_to point to ids that exist? (needed to
        # build real reply-tree structure later)
        ids = set(o["id"] for o in records)
        sample_replies = [o for o in records if o.get("reply_to")][:1000]
        resolvable = sum(1 for o in sample_replies if o["reply_to"] in ids)
        print(f"\n  reply_to resolvability check (first 1000 replies found): "
              f"{resolvable}/{len(sample_replies)} point to an id present in this file")

        # also peek at conversations.json and users.json briefly
        for other_file in ["reddit-corpus-small/conversations.json", "reddit-corpus-small/users.json"]:
            print(f"\n--- peeking at {other_file} ---")
            with z.open(other_file) as f:
                data = json.load(f)
            if isinstance(data, dict):
                keys = list(data.keys())
                print(f"  dict with {len(keys)} keys, first key: {keys[0]!r}")
                print(f"  value for first key: {json.dumps(data[keys[0]], indent=2)[:500]}")
            elif isinstance(data, list):
                print(f"  list with {len(data)} items, first item:")
                print(f"  {json.dumps(data[0], indent=2)[:500]}")


if __name__ == "__main__":
    download()
    inspect()
