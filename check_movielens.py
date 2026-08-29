from pathlib import Path
import pandas as pd

DATA = Path("data/ml-25m")

movies = pd.read_csv(DATA / "movies.csv")
print("Movies:", len(movies))
print("\nMovie columns:", list(movies.columns))

print("\nFirst 5 movies:")
print(movies.head())

print("\nNumber of genres:", movies["genres"].str.split("|").explode().nunique())

ratings = pd.read_csv(
    DATA / "ratings.csv",
    usecols=["userId", "movieId", "rating", "timestamp"]
)

print("\nRatings:", len(ratings))
print("Users:", ratings["userId"].nunique())
print("Rated movies:", ratings["movieId"].nunique())
print("Average rating:", round(ratings["rating"].mean(), 3))