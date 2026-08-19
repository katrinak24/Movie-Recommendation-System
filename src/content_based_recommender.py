"""
content_based_recommender.py
The recommendation engine actually used by the Streamlit app.

Verified implementation (see PROJECT_AUDIT_REPORT.md):
  - Content-based filtering only (no collaborative filtering, no
    matrix factorization, no hybrid scoring).
  - Features: each movie's overview + genres, concatenated into one
    text field.
  - Vectorization: scikit-learn TfidfVectorizer, English stop words
    removed, max_features=5000.
  - Similarity: cosine similarity (via linear_kernel, which is
    equivalent to cosine similarity for L2-normalized TF-IDF vectors).
  - Ranking: top-N most similar movies, excluding the query movie itself.
  - Unknown titles: difflib.get_close_matches finds the closest known
    title (cutoff=0.6) before giving up.

This is a straightforward rewrite of the original content_based.py:
same algorithm, same parameters, only the plumbing (paths, imports,
class-based caching so `n_recommendations` can vary) changed.
"""
from difflib import get_close_matches
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.data_preprocessing import load_tmdb


class ContentBasedRecommender:
    """TF-IDF + cosine-similarity content-based movie recommender."""

    def __init__(
        self,
        tmdb_csv_path: Path,
        sample_size: int = 3000,
        max_features: int = 5000,
    ):
        self.movies = load_tmdb(tmdb_csv_path, nrows=sample_size)
        self.movies["content"] = (
            self.movies["overview"] + " " + self.movies["genres"]
        )

        # KNOWN DATA ISSUE (see PROJECT_AUDIT_REPORT.md "Recommendation
        # Quality"): this dataset legitimately contains multiple movies
        # with the identical title (remakes - e.g. two "Aladdin"
        # entries, from 1992 and 2019). Title alone is NOT a unique
        # identifier. The original project indexed purely by title,
        # which silently collapsed remakes onto whichever row happened
        # to be seen last, and could "recommend" the other same-titled
        # movie back for a query - looking like a self-recommendation
        # bug even though it's really an identity-resolution bug.
        # Fix: index by the dataset's own unique "id" column, and
        # build a human-readable, disambiguated label (adding the
        # release year whenever a title is duplicated) for the UI and
        # for title-based lookups.
        year = self.movies["release_date"].str.slice(0, 4)
        title_counts = self.movies["title"].value_counts()
        self.movies["display_title"] = self.movies.apply(
            lambda r: f"{r['title']} ({year.loc[r.name]})"
            if title_counts.get(r["title"], 0) > 1 and year.loc[r.name]
            else r["title"],
            axis=1,
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=max_features
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.movies["content"].values.astype("U")
        )
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

        self._id_to_index = {
            movie_id: idx for idx, movie_id in enumerate(self.movies["id"])
        }
        self._display_title_to_index = {
            t: idx for idx, t in enumerate(self.movies["display_title"])
        }
        # First-occurrence title lookup, kept only for plain-title
        # free-text search where the caller has no id/display_title
        # to disambiguate with (matches the original app's behavior,
        # documented limitation: an ambiguous plain title resolves to
        # its first occurrence in the catalog).
        self._title_to_index = {}
        for idx, title in enumerate(self.movies["title"]):
            self._title_to_index.setdefault(title, idx)

    def _resolve_index(self, title: str) -> Optional[int]:
        """Resolve a query string (display title, plain title, or a
        close/fuzzy match) to a row index."""
        if title in self._display_title_to_index:
            return self._display_title_to_index[title]
        if title in self._title_to_index:
            return self._title_to_index[title]
        matches = get_close_matches(
            title, self.movies["display_title"].values, n=1, cutoff=0.6
        )
        if matches:
            return self._display_title_to_index[matches[0]]
        matches = get_close_matches(
            title, self.movies["title"].values, n=1, cutoff=0.6
        )
        return self._title_to_index[matches[0]] if matches else None

    def recommend(self, title: str, n: int = 5, movie_id: Optional[int] = None) -> pd.DataFrame:
        """Return the top-n movies most similar to `title`.

        Pass `movie_id` (the TMDB id) when it's available (e.g. from a
        UI selection) to resolve the query unambiguously, even when
        multiple movies share the same title. Falls back to
        title/display_title/fuzzy-match resolution otherwise.

        Returns a DataFrame with an "Error" column (and no other rows)
        if nothing resolves - this mirrors the original app's contract
        so app.py's error-handling didn't need to change.
        """
        if movie_id is not None and movie_id in self._id_to_index:
            idx = self._id_to_index[movie_id]
        else:
            idx = self._resolve_index(title)
            if idx is None:
                return pd.DataFrame({"Error": [f"'{title}' not found in dataset"]})

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1 : n + 1]  # exclude the movie itself

        movie_indices = [i for i, _ in sim_scores]
        scores = [s for _, s in sim_scores]

        results = self.movies.iloc[movie_indices][
            ["id", "title", "display_title", "release_date", "vote_average", "poster_path", "genres"]
        ].copy()
        results["similarity_score"] = scores
        results["poster_url"] = results["poster_path"].apply(
            lambda p: f"https://image.tmdb.org/t/p/w500{p}" if pd.notna(p) and p else None
        )
        return results.reset_index(drop=True)
