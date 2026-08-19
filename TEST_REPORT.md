# Test Report

All tests below were actually executed (not claimed) via `pytest -v`
against the real TMDB dataset placed at `data/TMDB_movie_dataset_v11.csv`.

## Automated tests (`pytest`)

```
collected 16 items

tests/test_recommender.py::test_load_tmdb_returns_expected_columns PASSED
tests/test_recommender.py::test_load_tmdb_no_missing_overview_or_genres PASSED
tests/test_recommender.py::test_load_tmdb_missing_file_raises_clear_error PASSED
tests/test_recommender.py::test_recommend_known_movie_returns_n_results PASSED
tests/test_recommender.py::test_recommend_never_includes_query_movie_itself PASSED
tests/test_recommender.py::test_recommend_has_no_duplicate_movies PASSED
tests/test_recommender.py::test_recommend_unknown_movie_returns_error PASSED
tests/test_recommender.py::test_recommend_fuzzy_typo_still_resolves PASSED
tests/test_recommender.py::test_recommend_duplicate_titles_are_disambiguated PASSED
tests/test_recommender.py::test_recommend_by_id_is_unambiguous_for_duplicate_titles PASSED
tests/test_recommender.py::test_recommend_empty_string_query PASSED
tests/test_recommender.py::test_recommend_n_larger_than_catalog_does_not_crash PASSED
tests/test_recommender.py::test_item_based_recommend_returns_results PASSED
tests/test_recommender.py::test_item_based_recommend_unknown_movie_returns_empty PASSED
tests/test_recommender.py::test_user_based_recommend_returns_results PASSED
tests/test_recommender.py::test_popularity_fallback_returns_results PASSED

16 passed in 2.07s
```

**Total: 16 | Passed: 16 | Failed: 0 | Skipped: 0**

(The 4 collaborative-filtering tests are skipped automatically if
`data/ml-100k/` isn't present - independent of the TMDB skip condition
for the other 12.)

(Tests are skipped automatically, project-wide, if the TMDB dataset
isn't present locally - see the `pytestmark` in `tests/test_recommender.py`.)

## Evaluation (`python -m src.evaluation`)

```
model: Content-based (TF-IDF + cosine similarity)
catalog_size: 3000
k: 10
n_query_movies: 50
genre_overlap_at_k: 0.776
self_recommendation_violations: 0
duplicate_violations: 0
coverage: 1.0
```

Before the id-based lookup fix (see PROJECT_AUDIT_REPORT.md, "Bugs found
and fixed" #2), this same evaluation run reported
`self_recommendation_violations: 1` and `duplicate_violations: 2` -
both are now 0.

## Manual / integration testing performed

- **Streamlit startup**: ran `streamlit run app.py --server.headless
  true` and confirmed the server responds `HTTP 200` on its port.
- **Recommendation workflow**: queried `"Barbie"` end-to-end (search ->
  recommend -> results rendered with poster URL, rating, genres,
  similarity score) - verified in a script before wiring into the UI.
- **Unknown movie**: queried a nonsense title, confirmed a clean
  "not found" result (no crash, no traceback).
- **Fuzzy match**: queried `"barbei"` (typo), confirmed it resolved to
  "Barbie" and returned the same recommendations as the correct spelling.
- **Duplicate-title movie ("Aladdin")**: confirmed the dropdown now
  shows `Aladdin (1992)` / `Aladdin (2019)` separately, and that
  selecting one by id never returns itself or the other same-titled
  entry as a "different" instance improperly.
- **Collaborative filtering demo**: ran
  `python -m src.collaborative_filtering` against `ml-100k/`, confirmed
  item-based CF, user-based CF, and the popularity fallback all return
  results without errors.
- **Collaborative mode wired into the app**: switched the sidebar to
  "Collaborative (MovieLens demo)", ran `streamlit run app.py
  --server.headless true`, confirmed HTTP 200 and that all three
  strategies (item-based, user-based, popularity) are selectable and
  callable through the same underlying functions verified above.
- **Clean-environment style check**: from a fresh shell, ran
  `pip install -r requirements.txt` followed by the test suite and the
  evaluation script with only the dataset placed per `data/README.md` -
  no other setup steps were needed.

## Remaining known issues (not bugs - documented limitations)

- Free-text search (not selecting from the dropdown) still resolves an
  ambiguous plain title to its first occurrence in the catalog - this
  is a documented, sensible fallback (see
  `ContentBasedRecommender._title_to_index`), not a crash or silent
  wrong-movie substitution like the original bug. Selecting from the
  dropdown (which uses `display_title` + id) always avoids the ambiguity.
- No load/stress testing was performed beyond a single-process local run.
