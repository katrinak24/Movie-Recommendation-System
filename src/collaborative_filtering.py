"""
collaborative_filtering.py

STATUS: implemented, functional, and now wired into the Streamlit app
(app.py) as an explicit "Collaborative (MovieLens demo)" sidebar mode.
In the originally uploaded project, app.py only ever imported
content_based.py and this module was never called - see
PROJECT_AUDIT_REPORT.md for that history.

Renamed from collaborative.py; algorithm logic unchanged. Also
runnable standalone against the MovieLens ml-100k dataset:

    python -m src.collaborative_filtering

What it actually implements:
  - User-based CF: builds a dense user-item ratings matrix, computes
    cosine similarity between users, and recommends movies liked by
    the 5 most-similar users to a fixed demo user (the first user in
    the matrix - there is no login/session concept in this project).
  - Item-based CF: computes cosine similarity between movie rating
    vectors (movies-as-columns) and recommends the movies most
    similar to a given movie title.
  - Popularity fallback: most-rated movies, for cold-start users/items.

What it does NOT implement:
  - Matrix factorization / SVD (there is no factorization anywhere in
    this codebase - only pairwise cosine similarity on the raw,
    dense user-item matrix).
  - A real cold-start strategy or per-user session: user_based_recommend()
    always benchmarks against "the first user in the matrix", not
    whichever user id is selected in the app's dropdown, since there
    is no concept of a logged-in user. The app's sidebar states this
    limitation directly next to the user id picker rather than
    implying it's personalized.

Known limitation: building a dense (943 users x 1682 movies) matrix is
fine for ml-100k, but this approach does not scale to large,
sparse catalogs without switching to a sparse-matrix representation.
"""
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def _title_to_movie_id(title: str, movies: pd.DataFrame):
    movie = movies[movies["title"] == title]
    if movie.empty:
        return None
    return movie.iloc[0]["movieId"]


def user_based_recommend(
    ratings: pd.DataFrame, movies: pd.DataFrame, movie_title: str, top_n: int = 10
) -> pd.DataFrame:
    """Recommend movies liked by users similar to a fixed demo user.

    NOTE: `movie_title` is accepted for interface symmetry with
    item_based_recommend() but is only used to validate that the
    movie exists; the recommendation itself is user-based and does
    not depend on which movie was passed in. This mirrors the
    original implementation's behavior.
    """
    movie_id = _title_to_movie_id(movie_title, movies)
    if movie_id is None:
        return pd.DataFrame()

    user_item_matrix = ratings.pivot(
        index="userId", columns="movieId", values="rating"
    ).fillna(0)

    similarity = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(
        similarity, index=user_item_matrix.index, columns=user_item_matrix.index
    )

    demo_user = user_item_matrix.index[0]
    similar_users = similarity_df[demo_user].sort_values(ascending=False).iloc[1:6].index

    similar_users_ratings = ratings[ratings["userId"].isin(similar_users)]
    top_movies = (
        similar_users_ratings.groupby("movieId")["rating"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    return movies[movies["movieId"].isin(top_movies.index)][["movieId", "title"]]


def item_based_recommend(
    ratings: pd.DataFrame, movies: pd.DataFrame, movie_title: str, top_n: int = 10
) -> pd.DataFrame:
    """Recommend movies whose rating patterns are similar to movie_title's."""
    movie_id = _title_to_movie_id(movie_title, movies)
    if movie_id is None:
        return pd.DataFrame()

    user_item_matrix = ratings.pivot(
        index="userId", columns="movieId", values="rating"
    ).fillna(0)

    similarity = cosine_similarity(user_item_matrix.T)
    similarity_df = pd.DataFrame(
        similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns
    )

    if movie_id not in similarity_df:
        return pd.DataFrame()

    similar_items = (
        similarity_df[movie_id].sort_values(ascending=False).iloc[1 : top_n + 1].index
    )
    return movies[movies["movieId"].isin(similar_items)][["movieId", "title"]]


def popularity_fallback(
    ratings: pd.DataFrame, movies: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """Most-rated movies - used as a cold-start fallback."""
    most_rated = ratings.groupby("movieId").size().sort_values(ascending=False).head(top_n)
    return movies[movies["movieId"].isin(most_rated.index)][["movieId", "title"]]


def _demo():
    """Small runnable demo against ml-100k, for `python -m src.collaborative_filtering`."""
    from config import ML100K_RATINGS_PATH, ML100K_ITEMS_PATH

    if not ML100K_RATINGS_PATH.exists():
        print(f"ml-100k data not found at {ML100K_RATINGS_PATH}. See data/README.md.")
        return

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

    print("Item-based CF, similar to 'Toy Story (1995)':")
    print(item_based_recommend(ratings, movies, "Toy Story (1995)", top_n=5))
    print("\nUser-based CF, for the demo user:")
    print(user_based_recommend(ratings, movies, "Toy Story (1995)", top_n=5))
    print("\nPopularity fallback:")
    print(popularity_fallback(ratings, movies, top_n=5))


if __name__ == "__main__":
    _demo()
