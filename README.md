# Movie Recommendation System

A content-based movie recommender built with Python, scikit-learn, and
Streamlit. Given a movie, it recommends similar titles using TF-IDF
over each movie's overview and genres, ranked by cosine similarity.

This README describes only what is actually implemented and verified.

## Features (verified)

- **Two explicit modes in the sidebar** - Content-based (TMDB) and
  Collaborative (MovieLens demo). They are two independent pipelines
  over two different datasets; the UI never blends their outputs into
  a fabricated "hybrid" score.
- Content-based mode: search and dropdown movie selection
  (disambiguated by release year when multiple movies share a title,
  e.g. remakes), recommendations with poster/rating/genres/similarity
  score, top-10-by-rating section, downloadable CSV.
- Collaborative mode: item-based CF ("movies similar to X"), user-based
  CF (against a demo user id you pick), and a popularity fallback -
  all running live against MovieLens ml-100k.

## Recommendation approach

**Content-based filtering:**
- Features: `overview + " " + genres` per movie.
- Vectorization: `TfidfVectorizer(stop_words="english", max_features=5000)`.
- Similarity: cosine similarity (`sklearn.metrics.pairwise.linear_kernel`
  on the TF-IDF matrix).
- Catalog: the first `CONTENT_SAMPLE_SIZE` (default 3000) rows of the
  TMDB dataset, which is roughly popularity-ordered.

**Collaborative filtering (implemented, and now wired into the app as
a separate mode):** `src/collaborative_filtering.py` implements
user-based and item-based memory-based CF (cosine similarity over a
MovieLens ml-100k user-item matrix) plus a popularity fallback. Select
"Collaborative (MovieLens demo)" in the app's sidebar to use it. It is
still clearly labeled as a demo in the UI: there is no login/session
system in this project, so "user-based" recommendations are always
computed against a fixed demo user, not "you" - this limitation is
shown directly in the app, not hidden. You can also still run it
standalone: `python -m src.collaborative_filtering`. See that file's
docstring for other documented limitations (no real cold-start
handling, dense matrix only - doesn't scale past ml-100k's size
without switching to a sparse representation).

**Matrix factorization / SVD:** discussed in the project's original
LinkedIn description, but **not implemented anywhere** in this
codebase. There is no factorization of any kind - only pairwise cosine
similarity. This claim has been removed from the description below and
should be removed from LinkedIn/GitHub topics too.

**Hybrid recommendation:** not implemented.

## Dataset

- **TMDB_movie_dataset_v11.csv** (required) - ~1.28M rows of movie
  metadata. Only the first 3000 rows (by the file's approximate
  popularity ordering) are loaded into memory. See `data/README.md`
  for how to obtain it - it is not bundled here (550MB, third-party
  redistribution terms).
- **MovieLens ml-100k** (optional) - 943 users, 1682 movies, 100,000
  ratings. Only used by the standalone collaborative-filtering demo,
  not by the app. See `data/README.md`.

## System architecture

```
TMDB_movie_dataset_v11.csv
        |
        v
data_preprocessing.load_tmdb()      (select columns, fill NaN overview/genres)
        |
        v
ContentBasedRecommender             (TF-IDF vectorize -> cosine similarity matrix)
        |
        v
        recommend(title | movie_id) -> ranked DataFrame
        |
        v
    Streamlit UI (app.py) -- "Content-based (TMDB)" mode

MovieLens ml-100k (u.data, u.item)
        |
        v
collaborative_filtering.{user,item}_based_recommend() / popularity_fallback()
        |
        v
    Streamlit UI (app.py) -- "Collaborative (MovieLens demo)" mode
```

The two modes are independent pipelines selected explicitly in the
sidebar - the app never blends their outputs.

## Technology stack

- Python 3.10+ (tested on 3.12)
- pandas, scikit-learn (TF-IDF, cosine similarity)
- Streamlit
- pytest

## Project structure

```
Movie-Recommendation-System/
├── app.py                          # Streamlit app (mode toggle: content-based / collaborative demo)
├── config.py                       # portable paths + tunable constants
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── data_preprocessing.py       # TMDB CSV loading/cleaning
│   ├── content_based_recommender.py# the recommender actually used by the app
│   ├── collaborative_filtering.py  # user/item-based CF, wired into the app as a separate mode
│   └── evaluation.py               # real, executable evaluation metrics
├── data/
│   └── README.md                   # how to obtain the datasets
├── tests/
│   └── test_recommender.py
├── PROJECT_AUDIT_REPORT.md
├── FILE_RENAMING_REPORT.md
└── TEST_REPORT.md
```

## Installation

```bash
git clone <your-repo-url>
cd Movie-Recommendation-System
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset setup

See `data/README.md`.

- Content-based mode needs `data/TMDB_movie_dataset_v11.csv`.
- Collaborative demo mode needs `data/ml-100k/` (ships with
  `FINAL_WORKING_VERSION` since it's small; not included in the
  GitHub-ready version - see `data/README.md`).

Either mode works if only its own dataset is present; the app shows a
clear error (not a crash) if you pick a mode whose data is missing.

## Running the application

```bash
streamlit run app.py
```

Open the printed local URL (default `http://localhost:8501`).

## Evaluation

Run:

```bash
python -m src.evaluation
```

This computes, on the actual model against the real dataset (last run
on a 3000-movie catalog, k=10, 50 sampled query movies):

| Metric | Value | What it means |
|---|---|---|
| Genre overlap@10 | 0.776 | 77.6% of recommended movies share at least one genre with the query movie |
| Self-recommendation violations | 0 | a movie is never recommended for itself |
| Duplicate violations | 0 | no recommendation list contains the same movie twice |
| Coverage | 1.0 | 100% of sampled queries produced recommendations |

**Why not Precision@K / Recall@K / NDCG@K?** Those require ground-truth
user relevance labels. This project has no user-interaction data wired
to the content-based engine (the only relevance data present -
MovieLens ratings - is only used by the disconnected collaborative
module). Reporting those metrics here would mean fabricating a
"relevant" set, so this project reports genre-overlap and integrity
checks instead, and says so plainly. See `src/evaluation.py`.

## Tests

```bash
pytest
```

See `TEST_REPORT.md` for the last run's results.

## Screenshots

Not included - see `assets/screenshots/` in the GitHub-ready version
for placeholders; capture your own after installing.

## Limitations

- Only the first `CONTENT_SAMPLE_SIZE` (3000) movies are loaded - the
  full 1.28M-row catalog is never used by the running app.
- Recommendations are topical/content similarity only - not
  personalized to any individual user (there is no user account or
  interaction history in this app).
- Duplicate titles in the source dataset (remakes) are disambiguated
  by year in the UI and internally resolved by unique id - see
  `src/content_based_recommender.py` docstring.
- Collaborative mode's "user-based" strategy always evaluates against
  a fixed demo user, not the id you pick in the dropdown - the app
  says this explicitly rather than pretending it's personalized.
- No hybrid model, no matrix factorization.
- Genre-overlap@K is a topical sanity metric, not a measure of whether
  any real user would like the recommendations.

## Future improvements

- Make user-based CF actually respect the selected demo user id
  (currently a known, UI-disclosed limitation - see above).
- Add a proper hybrid score once user interaction data is available
  for the same catalog on both sides.
- Move to a sparse/approximate-neighbor similarity search to scale
  past the current 3000-movie catalog.

## License

Add a license of your choosing (e.g. MIT) before publishing. The TMDB
and MovieLens datasets are **not** covered by that license - they carry
their own separate terms (see `data/README.md`).
