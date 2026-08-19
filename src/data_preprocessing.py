"""
data_preprocessing.py
Loading and cleaning for the TMDB movie dataset used by the
content-based recommender.

Renamed from the original utils/data_prep.py. Behavior is unchanged
except:
  - uses pathlib instead of a hardcoded relative string
  - accepts an optional `nrows` so callers don't have to read the
    entire ~1.28M row / ~550MB CSV into memory just to keep the first
    few thousand rows (this was the main performance bug in the
    original project - see PROJECT_AUDIT_REPORT.md, section "Performance").
"""
from pathlib import Path
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = [
    "id", "title", "overview", "genres",
    "release_date", "vote_average", "poster_path",
]


def load_tmdb(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load and lightly clean the TMDB movie metadata CSV.

    Parameters
    ----------
    path : Path
        Location of TMDB_movie_dataset_v11.csv.
    nrows : int, optional
        If given, only the first `nrows` rows of the CSV are read.
        The dataset is (approximately) ordered from most to least
        popular, so this is used to cheaply take a "top N popular
        movies" slice without reading the full file.

    Raises
    ------
    FileNotFoundError
        With a clear, actionable message if the dataset is missing
        (see data/README.md for how to obtain it).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"TMDB dataset not found at {path}.\n"
            "See data/README.md for download instructions."
        )

    movies = pd.read_csv(path, nrows=nrows)

    missing = [c for c in REQUIRED_COLUMNS if c not in movies.columns]
    if missing:
        raise ValueError(
            f"TMDB dataset at {path} is missing expected column(s): {missing}. "
            "This project was built against TMDB_movie_dataset_v11.csv - "
            "a different TMDB export may use different column names."
        )

    movies = movies[REQUIRED_COLUMNS].copy()

    # Fill missing text fields so downstream TF-IDF never chokes on NaN.
    movies["overview"] = movies["overview"].fillna("")
    movies["genres"] = movies["genres"].fillna("")

    # Drop rows with no title at all - can't recommend or display these.
    movies = movies.dropna(subset=["title"]).reset_index(drop=True)

    return movies
