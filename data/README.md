# Dataset Setup

This project uses external datasets that are not bundled with the repository.

## 1. TMDB Movie Dataset

Download `TMDB_movie_dataset_v11.csv` separately and place it here:

```text
data/TMDB_movie_dataset_v11.csv
```

The application uses this dataset for the content-based recommendation pipeline.

## 2. MovieLens 100K

The collaborative filtering demo uses the MovieLens 100K dataset.

Place the required files here:

```text
data/ml-100k/u.data
data/ml-100k/u.item
```

After placing the datasets, run the Streamlit application from the project root.

> Dataset files are intentionally excluded from the GitHub repository because of their size and distribution considerations.
