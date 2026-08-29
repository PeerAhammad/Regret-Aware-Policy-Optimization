import pandas as pd

path = "data/ml-25m/tags.csv"

tags = pd.read_csv(path)

keywords = [
    "controversial", "politics", "political",
    "religion", "violence", "racism",
    "sex", "war", "terrorism"
]

for k in keywords:
    matches = tags[tags["tag"].str.contains(k, case=False, na=False)]
    print(f"\n{k}: {len(matches)} matches")
    if len(matches):
        print(matches[["movieId", "tag"]].head(10).to_string(index=False))