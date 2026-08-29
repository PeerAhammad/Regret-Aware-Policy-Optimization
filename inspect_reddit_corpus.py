"""
Downloads and inspects the REAL reddit-corpus-small dataset directly
(38MB zip, no convokit package needed -- convokit's pip install pulls in
torch/transformers/spacy for no reason relevant to just reading JSON
data, and filled up disk during testing).

Run this FIRST, before any environment/pipeline code gets built, so we
design against the actual schema rather than guessing -- that's the
lesson from MovieLens needing three rounds of real-data bugs.
"""

import json
import zipfile
import io
import urllib.request
from collections import Counter

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
        for n in names[:20]:
            print(" ", n)
        if len(names) > 20:
            print(f"  ... and {len(names) - 20} more")

        # ConvoKit corpus format is typically a folder of jsonl/json files.
        # Find and peek at the utterances file specifically.
        utt_file = next((n for n in names if "utterance" in n.lower()), None)
        if utt_file is None:
            print("\nCould not find an utterances file by name -- listing all names above,"
                  " check manually.")
            return

        print(f"\nReading first few records from: {utt_file}")
        with z.open(utt_file) as f:
            lines = []
            for i, line in enumerate(io.TextIOWrapper(f, encoding="utf-8")):
                if i >= 5:
                    break
                lines.append(line)

        for line in lines:
            try:
                obj = json.loads(line)
                print(json.dumps(obj, indent=2)[:800])
                print("---")
            except json.JSONDecodeError:
                print("(not line-delimited JSON, or this file is a single JSON blob"
                      " -- will need a different parse strategy)")
                break

        # If it parsed as JSONL, do a fuller pass for schema/stats
        with z.open(utt_file) as f:
            subreddits = Counter()
            has_reply_to = 0
            has_score = 0
            score_values = []
            total = 0
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                meta = obj.get("meta", {})
                sr = meta.get("subreddit")
                if sr:
                    subreddits[sr] += 1
                if obj.get("reply_to") is not None:
                    has_reply_to += 1
                if "score" in meta:
                    has_score += 1
                    try:
                        score_values.append(float(meta["score"]))
                    except (TypeError, ValueError):
                        pass
                if total >= 50000:  # cap the scan, this is just a sanity check
                    break

        print(f"\nScanned {total} utterances:")
        print(f"  with reply_to set: {has_reply_to} ({100*has_reply_to/max(total,1):.1f}%)")
        print(f"  with score field: {has_score} ({100*has_score/max(total,1):.1f}%)")
        if score_values:
            import statistics
            print(f"  score stats: min={min(score_values):.0f}, max={max(score_values):.0f}, "
                  f"mean={statistics.mean(score_values):.2f}, "
                  f"median={statistics.median(score_values):.0f}")
            neg = sum(1 for s in score_values if s < 0)
            print(f"  negative-score fraction: {neg}/{len(score_values)} = {100*neg/len(score_values):.1f}%")
        print(f"\n  top 15 subreddits by utterance count in this scan:")
        for sr, c in subreddits.most_common(15):
            print(f"    {sr}: {c}")


if __name__ == "__main__":
    download()
    inspect()
