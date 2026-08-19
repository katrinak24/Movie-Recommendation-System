"""
evaluation.py

The original project had NO evaluation of any kind - no metrics were
computed anywhere. This module adds real, executable evaluation for
the content-based recommender that is actually in the app.

Why not Precision@K / Recall@K / NDCG@K in the classic sense?
Those metrics need ground-truth relevance labels (e.g. "did this user
actually watch/like this movie?"). This project has no user
interaction data wired to the content-based engine (the MovieLens
ratings in ml-100k are only used by the separate, disconnected
collaborative_filtering.py module - see that file's docstring).
Reporting Precision@K here against no ground truth would mean
fabricating the "relevant" set, so this module does not do that.

What this module does compute, for real, on the actual model:
  1. Genre-overlap@K: of the top-K recommendations for a movie, what
     fraction share at least one genre with the query movie? This is
     a standard sanity metric for content-based systems (it does not
     require user ground truth) - it answers "are the recommendations
     even topically related?", not "would a specific user like them".
  2. Self-recommendation check: a movie must never recommend itself.
  3. Duplicate check: a single recommendation list must not contain
     duplicate movies.
  4. Coverage: fraction of a query sample that resolves to a
     recommendation (vs. "not found").

Run directly with:
    python -m src.evaluation
"""
from pathlib import Path
from typing import List

import pandas as pd

from src.content_based_recommender import ContentBasedRecommender


def _genres_set(genre_string: str) -> set:
    if not genre_string:
        return set()
    return {g.strip().lower() for g in genre_string.split(",") if g.strip()}


def genre_overlap_at_k(
    recommender: ContentBasedRecommender, query_ids: List[int], k: int = 10
) -> float:
    """Fraction of recommended movies (across all queries) that share
    at least one genre with their query movie. Queries are resolved by
    unique movie id (see docstring at the top of this file / the note
    in ContentBasedRecommender.__init__ about duplicate titles)."""
    total, overlapping = 0, 0
    for movie_id in query_ids:
        query_genres = _genres_set(
            recommender.movies.loc[recommender.movies["id"] == movie_id, "genres"].iloc[0]
        )
        recs = recommender.recommend(title="", n=k, movie_id=movie_id)
        if "Error" in recs.columns:
            continue
        for rec_genres in recs["genres"]:
            total += 1
            if query_genres & _genres_set(rec_genres):
                overlapping += 1
    return overlapping / total if total else 0.0


def self_recommendation_check(
    recommender: ContentBasedRecommender, query_ids: List[int], k: int = 10
) -> int:
    """Returns the number of queries where the query movie's own id
    appeared in its own recommendation list (should always be 0)."""
    violations = 0
    for movie_id in query_ids:
        recs = recommender.recommend(title="", n=k, movie_id=movie_id)
        if "Error" in recs.columns:
            continue
        if movie_id in recs["id"].values:
            violations += 1
    return violations


def duplicate_check(
    recommender: ContentBasedRecommender, query_ids: List[int], k: int = 10
) -> int:
    """Returns the number of queries whose recommendation list contains
    a duplicate movie id (should always be 0)."""
    violations = 0
    for movie_id in query_ids:
        recs = recommender.recommend(title="", n=k, movie_id=movie_id)
        if "Error" in recs.columns:
            continue
        if recs["id"].duplicated().any():
            violations += 1
    return violations


def coverage(
    recommender: ContentBasedRecommender, query_ids: List[int], k: int = 10
) -> float:
    """Fraction of queries that produced recommendations (vs 'not found')."""
    found = 0
    for movie_id in query_ids:
        recs = recommender.recommend(title="", n=k, movie_id=movie_id)
        if "Error" not in recs.columns:
            found += 1
    return found / len(query_ids) if query_ids else 0.0


def run_evaluation(tmdb_csv_path: Path, sample_size: int = 3000, k: int = 10, n_queries: int = 50):
    recommender = ContentBasedRecommender(tmdb_csv_path, sample_size=sample_size)
    id_sample = recommender.movies["id"].sample(
        n=min(n_queries, len(recommender.movies)), random_state=42
    ).tolist()

    results = {
        "model": "Content-based (TF-IDF + cosine similarity)",
        "catalog_size": len(recommender.movies),
        "k": k,
        "n_query_movies": len(id_sample),
        "genre_overlap_at_k": round(genre_overlap_at_k(recommender, id_sample, k), 4),
        "self_recommendation_violations": self_recommendation_check(recommender, id_sample, k),
        "duplicate_violations": duplicate_check(recommender, id_sample, k),
        "coverage": round(coverage(recommender, id_sample, k), 4),
    }
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import TMDB_CSV_PATH, CONTENT_SAMPLE_SIZE

    res = run_evaluation(TMDB_CSV_PATH, sample_size=CONTENT_SAMPLE_SIZE)
    for key, value in res.items():
        print(f"{key}: {value}")
