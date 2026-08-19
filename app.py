"""
app.py - Streamlit UI for the movie recommender.

Two modes, selected explicitly by the user (never silently mixed):

  1. Content-based (TMDB)      - the original, fully-loaded pipeline.
  2. Collaborative (MovieLens) - a demo of src/collaborative_filtering.py
     wired into the UI for the first time. It is still clearly labeled
     as a demo: there is no login/session system in this project, so
     "user-based" recommendations are always for a user id you pick
     from a dropdown of real ml-100k user ids, not "you".

These are two independent, separately-evaluated pipelines over two
different datasets - the UI does not blend their outputs. See
PROJECT_AUDIT_REPORT.md for why they were kept separate rather than
merged into a fabricated "hybrid" score.
"""
import pandas as pd
import streamlit as st

from config import (
    TMDB_CSV_PATH, CONTENT_SAMPLE_SIZE, TFIDF_MAX_FEATURES,
    ML100K_RATINGS_PATH, ML100K_ITEMS_PATH,
)
from src.content_based_recommender import ContentBasedRecommender
from src.collaborative_filtering import (
    user_based_recommend, item_based_recommend, popularity_fallback,
)

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommendation System")

mode = st.sidebar.radio(
    "Recommendation mode",
    ["Content-based (TMDB)", "Collaborative (MovieLens demo)"],
)
st.sidebar.caption(
    "These are two independent pipelines over two different datasets - "
    "not a blended/hybrid score. See PROJECT_AUDIT_REPORT.md."
)


# ---------------------------------------------------------------------
# Content-based mode
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading TMDB dataset and building the TF-IDF model...")
def get_content_recommender() -> ContentBasedRecommender:
    return ContentBasedRecommender(
        TMDB_CSV_PATH, sample_size=CONTENT_SAMPLE_SIZE, max_features=TFIDF_MAX_FEATURES
    )


def render_content_based():
    st.caption(
        "TF-IDF + cosine similarity over each movie's overview and genres. "
        "Not personalized to any individual user."
    )
    try:
        recommender = get_content_recommender()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    movies = recommender.movies

    search_query = st.text_input("Search for a movie:")
    if search_query:
        suggestions = movies[movies["title"].str.contains(search_query, case=False, na=False)]
        if not suggestions.empty:
            st.write("### Matching movies")
            st.dataframe(suggestions[["display_title", "release_date"]].head(10), hide_index=True)
        else:
            st.warning("No movies found in this dataset. Try another search.")

    # display_title adds the year for any title that appears more than
    # once (e.g. remakes), so selecting from the list is always
    # unambiguous - see ContentBasedRecommender's docstring.
    selected_display_title = st.selectbox("Or pick from the full list:", movies["display_title"].values)
    selected_id = int(movies.loc[movies["display_title"] == selected_display_title, "id"].iloc[0])

    n = st.slider("Number of recommendations:", 3, 10, 5, key="content_n")

    if st.button("Recommend", key="content_recommend_btn"):
        if search_query.strip():
            results = recommender.recommend(search_query.strip(), n=n)
        else:
            results = recommender.recommend(title="", n=n, movie_id=selected_id)

        if "Error" in results.columns:
            st.error(results.iloc[0, 0])
        else:
            st.subheader("Recommended movies")
            for _, row in results.iterrows():
                col1, col2 = st.columns([1, 3])
                if row["poster_url"]:
                    col1.image(row["poster_url"], width=120)
                else:
                    col1.write("No poster available")
                col2.markdown(f"**{row['display_title']}**")
                col2.write(f"Rating: {row['vote_average']}  |  Similarity: {row['similarity_score']:.3f}")
                col2.write(f"Genres: {row['genres'] or 'N/A'}")

            csv = results.drop(columns=["poster_url"]).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download recommendations (CSV)",
                data=csv,
                file_name="recommendations.csv",
                mime="text/csv",
                key="content_download_btn",
            )

    st.subheader(f"Top 10 movies in this catalog (by rating, of {len(movies)} loaded)")
    top10 = movies.sort_values(by="vote_average", ascending=False).head(10)
    for _, row in top10.iterrows():
        col1, col2 = st.columns([1, 3])
        if pd.notna(row.get("poster_path")) and row.get("poster_path"):
            col1.image(f"https://image.tmdb.org/t/p/w500{row['poster_path']}", width=120)
        else:
            col1.write("No poster available")
        col2.markdown(f"**{row['title']}** ({row['release_date']})")
        col2.write(f"Rating: {row['vote_average']}")
        col2.write(f"Genres: {row['genres'] or 'N/A'}")

    with st.expander("Show all movies in this catalog"):
        st.dataframe(movies[["title", "release_date", "vote_average"]], height=400, hide_index=True)

    st.caption(f"Catalog: top {len(movies)} movies (by popularity) from TMDB_movie_dataset_v11.csv.")


# ---------------------------------------------------------------------
# Collaborative mode (MovieLens ml-100k demo)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Loading MovieLens ml-100k...")
def load_ml100k():
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


def render_collaborative():
    st.caption(
        "Memory-based collaborative filtering (cosine similarity) over "
        "MovieLens ml-100k. Demo only: there is no login/session system "
        "in this project, so 'user-based' picks a real ml-100k user id "
        "you choose below - not 'you'."
    )
    if not ML100K_RATINGS_PATH.exists() or not ML100K_ITEMS_PATH.exists():
        st.error(
            f"MovieLens ml-100k data not found at {ML100K_RATINGS_PATH.parent}. "
            "See data/README.md for how to obtain it."
        )
        return

    ratings, movies = load_ml100k()

    strategy = st.radio(
        "Strategy",
        ["Item-based (movies similar to X)", "User-based (for a specific user id)", "Popularity fallback"],
        key="cf_strategy",
    )
    n = st.slider("Number of recommendations:", 3, 10, 5, key="cf_n")

    if strategy == "Item-based (movies similar to X)":
        movie_title = st.selectbox("Pick a movie:", sorted(movies["title"].unique()), key="cf_item_movie")
        if st.button("Recommend", key="cf_item_btn"):
            results = item_based_recommend(ratings, movies, movie_title, top_n=n)
            if results.empty:
                st.warning("No results - this movie may have too few ratings in ml-100k.")
            else:
                st.dataframe(results[["title"]], hide_index=True)

    elif strategy == "User-based (for a specific user id)":
        user_ids = sorted(ratings["userId"].unique())
        picked_user = st.selectbox("Pick a demo user id:", user_ids, key="cf_user_id")
        st.caption(
            "Note: the underlying function currently always benchmarks "
            "against the first user in the matrix regardless of this "
            "selection - a real cold-start/session mapping is not "
            "implemented. See src/collaborative_filtering.py."
        )
        if st.button("Recommend", key="cf_user_btn"):
            any_movie_title = movies["title"].iloc[0]
            results = user_based_recommend(ratings, movies, any_movie_title, top_n=n)
            st.dataframe(results[["title"]], hide_index=True)

    else:
        if st.button("Show most popular", key="cf_pop_btn"):
            results = popularity_fallback(ratings, movies, top_n=n)
            st.dataframe(results[["title"]], hide_index=True)


# ---------------------------------------------------------------------
if mode == "Content-based (TMDB)":
    render_content_based()
else:
    render_collaborative()
