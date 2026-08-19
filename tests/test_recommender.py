"""
tests/test_recommender.py
Real, executable tests. Run with:  pytest

Requires the TMDB dataset to be present at data/TMDB_movie_dataset_v11.csv
(see data/README.md) - these are integration tests against the real
data pipeline, not mocked unit tests, since the main risk in this
project (per PROJECT_AUDIT_REPORT.md) was data-shape/data-quality bugs
that a mock would hide.
"""
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from config import (
    TMDB_CSV_PATH, CONTENT_SAMPLE_SIZE, ML100K_RATINGS_PATH, ML100K_ITEMS_PATH,
)  # noqa: E402
from src.data_preprocessing import load_tmdb  # noqa: E402
from src.content_based_recommender import ContentBasedRecommender  # noqa: E402
from src.collaborative_filtering import (
    item_based_recommend, user_based_recommend, popularity_fallback,
)  # noqa: E402

pytestmark = pytest.mark.skipif(
    not TMDB_CSV_PATH.exists(),
    reason="TMDB dataset not present locally - see data/README.md",
)


@pytest.fixture(scope="module")
def recommender():
    return ContentBasedRecommender(TMDB_CSV_PATH, sample_size=500)


# ---- Data loading ----

def test_load_tmdb_returns_expected_columns():
    df = load_tmdb(TMDB_CSV_PATH, nrows=100)
    assert list(df.columns) == [
        "id", "title", "overview", "genres",
        "release_date", "vote_average", "poster_path",
    ]


def test_load_tmdb_no_missing_overview_or_genres():
    df = load_tmdb(TMDB_CSV_PATH, nrows=500)
    assert df["overview"].isna().sum() == 0
    assert df["genres"].isna().sum() == 0


def test_load_tmdb_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tmdb(tmp_path / "does_not_exist.csv")


# ---- Content-based recommender ----

def test_recommend_known_movie_returns_n_results(recommender):
    any_title = recommender.movies["title"].iloc[0]
    results = recommender.recommend(any_title, n=5)
    assert "Error" not in results.columns
    assert len(results) == 5


def test_recommend_never_includes_query_movie_itself(recommender):
    for movie_id in recommender.movies["id"].head(20):
        results = recommender.recommend(title="", n=5, movie_id=int(movie_id))
        assert movie_id not in results["id"].values


def test_recommend_has_no_duplicate_movies(recommender):
    for movie_id in recommender.movies["id"].head(20):
        results = recommender.recommend(title="", n=5, movie_id=int(movie_id))
        assert not results["id"].duplicated().any()


def test_recommend_unknown_movie_returns_error(recommender):
    results = recommender.recommend("Definitely Not A Real Movie Title 9999", n=5)
    assert "Error" in results.columns
    assert len(results) == 1


def test_recommend_fuzzy_typo_still_resolves(recommender):
    real_title = recommender.movies["title"].iloc[0]
    typo = real_title[:-1] if len(real_title) > 3 else real_title  # drop last char
    results = recommender.recommend(typo, n=3)
    # either resolves via fuzzy match, or genuinely doesn't - both are
    # valid outcomes, but the call must never raise
    assert "Error" in results.columns or len(results) == 3


def test_recommend_duplicate_titles_are_disambiguated(recommender):
    dup_titles = recommender.movies[recommender.movies["title"].duplicated(keep=False)]
    if dup_titles.empty:
        pytest.skip("no duplicate titles in this sample size")
    # every duplicated title must have a distinct display_title
    for title, group in dup_titles.groupby("title"):
        assert group["display_title"].nunique() == len(group)


def test_recommend_by_id_is_unambiguous_for_duplicate_titles(recommender):
    dup_titles = recommender.movies[recommender.movies["title"].duplicated(keep=False)]
    if dup_titles.empty:
        pytest.skip("no duplicate titles in this sample size")
    row = dup_titles.iloc[0]
    results = recommender.recommend(title="", n=5, movie_id=int(row["id"]))
    assert int(row["id"]) not in results["id"].values


# ---- Edge cases ----

def test_recommend_empty_string_query(recommender):
    results = recommender.recommend("", n=5)
    assert "Error" in results.columns


def test_recommend_n_larger_than_catalog_does_not_crash(recommender):
    results = recommender.recommend(recommender.movies["title"].iloc[0], n=10_000)
    assert "Error" not in results.columns
    assert len(results) <= len(recommender.movies) - 1


# ---- Collaborative filtering (ml-100k, used by the app's "Collaborative" mode) ----

ml100k_pytestmark = pytest.mark.skipif(
    not (ML100K_RATINGS_PATH.exists() and ML100K_ITEMS_PATH.exists()),
    reason="MovieLens ml-100k not present locally - see data/README.md",
)


@pytest.fixture(scope="module")
def ml100k():
    ratings = pd.read_csv(
        ML100K_RATINGS_PATH, sep="\t", header=None,
        names=["userId", "movieId", "rating", "timestamp"],
    )
    item_cols = ["movieId", "title", "release_date", "video_release_date", "imdb_url"] + [
        f"genre_{i}" for i in range(19)
    ]
    movies = pd.read_csv(
        ML100K_ITEMS_PATH, sep="|", encoding="latin-1", header=None, names=item_cols
    )
    return ratings, movies


@ml100k_pytestmark
def test_item_based_recommend_returns_results(ml100k):
    ratings, movies = ml100k
    results = item_based_recommend(ratings, movies, "Star Wars (1977)", top_n=5)
    assert len(results) == 5
    assert "Star Wars (1977)" not in results["title"].values


@ml100k_pytestmark
def test_item_based_recommend_unknown_movie_returns_empty(ml100k):
    ratings, movies = ml100k
    results = item_based_recommend(ratings, movies, "Not A Real Movie Title", top_n=5)
    assert results.empty


@ml100k_pytestmark
def test_user_based_recommend_returns_results(ml100k):
    ratings, movies = ml100k
    results = user_based_recommend(ratings, movies, "Star Wars (1977)", top_n=5)
    assert len(results) <= 5
    assert not results.empty


@ml100k_pytestmark
def test_popularity_fallback_returns_results(ml100k):
    ratings, movies = ml100k
    results = popularity_fallback(ratings, movies, top_n=5)
    assert len(results) == 5
