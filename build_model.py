"""
Build the recommendation artifacts (movies.pkl + similarity.pkl).

Usage:
    python build_model.py

Expected input (place these CSVs from the TMDB 5000 dataset in ./raw/):
    raw/tmdb_5000_movies.csv
    raw/tmdb_5000_credits.csv

Dataset:
    https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
"""
import ast
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_names(text: str):
    try:
        return [d["name"] for d in ast.literal_eval(text)]
    except Exception:
        return []


def parse_cast(text: str, top: int = 3):
    return parse_names(text)[:top]


def parse_director(text: str):
    try:
        for d in ast.literal_eval(text):
            if d.get("job") == "Director":
                return [d["name"]]
    except Exception:
        pass
    return []


def collapse(items):
    return [str(i).replace(" ", "").lower() for i in items]


def main():
    movies = pd.read_csv("raw/tmdb_5000_movies.csv")
    credits = pd.read_csv("raw/tmdb_5000_credits.csv")

    df = movies.merge(credits, on="title")
    df = df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    df = df.dropna()

    df["genres"] = df["genres"].apply(parse_names)
    df["keywords"] = df["keywords"].apply(parse_names)
    df["cast"] = df["cast"].apply(parse_cast)
    df["crew"] = df["crew"].apply(parse_director)
    df["overview"] = df["overview"].apply(lambda x: x.split())

    for col in ["genres", "keywords", "cast", "crew"]:
        df[col] = df[col].apply(collapse)

    df["tags"] = (
        df["overview"] + df["genres"] + df["keywords"] + df["cast"] + df["crew"]
    )
    final = df[["movie_id", "title", "tags"]].copy()
    final["tags"] = final["tags"].apply(lambda x: " ".join(x).lower())

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(final["tags"]).toarray()
    similarity = cosine_similarity(vectors)

    os.makedirs("data", exist_ok=True)
    with open("data/movies.pkl", "wb") as f:
        pickle.dump(final[["movie_id", "title"]].reset_index(drop=True), f)
    with open("data/similarity.pkl", "wb") as f:
        pickle.dump(similarity, f)

    print(f"Saved data/movies.pkl ({len(final)} movies) and data/similarity.pkl")


if __name__ == "__main__":
    main()
