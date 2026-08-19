# Data Included in the Local Package

This local final package already contains the datasets needed by the application.

## 1. TMDB movie metadata

File:

`data/TMDB_movie_dataset_v11.csv`

Required columns used by the application:

`id, title, overview, genres, release_date, vote_average, poster_path`

The application loads the first 3000 rows by default (`CONTENT_SAMPLE_SIZE` in `config.py`) so the TF-IDF model stays practical on a normal laptop.

## 2. MovieLens ml-100k

Folder:

`data/ml-100k/`

Required files:

- `u.data`
- `u.item`

These power the separate collaborative-filtering demo mode.

## Important

Do not rename or move the files. The application uses portable paths relative to the project folder, so it does not depend on the original user's Windows path.
