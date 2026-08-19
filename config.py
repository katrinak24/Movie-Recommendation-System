"""
config.py
Central configuration and portable path definitions.

No machine-specific paths anywhere in this project - everything is
relative to PROJECT_ROOT so the app runs on any machine / any OS.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

# --- Content-based recommender (TMDB) ---
TMDB_CSV_PATH = DATA_DIR / "TMDB_movie_dataset_v11.csv"
# Number of most-popular rows to load from the TMDB CSV for the
# in-memory TF-IDF model. The CSV ships roughly ordered from most to
# least popular, so taking the first N rows approximates "top N
# popular movies". This keeps the TF-IDF matrix small enough to build
# in a few seconds on a laptop. Increase if you have more RAM/CPU time.
CONTENT_SAMPLE_SIZE = 3000
TFIDF_MAX_FEATURES = 5000

# --- Collaborative filtering demo (MovieLens ml-100k) ---
ML100K_DIR = DATA_DIR / "ml-100k"
ML100K_RATINGS_PATH = ML100K_DIR / "u.data"
ML100K_ITEMS_PATH = ML100K_DIR / "u.item"
