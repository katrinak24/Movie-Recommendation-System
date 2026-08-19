# 🎬 Movie Recommendation System

### Content-Based & Collaborative Movie Recommendation System

A Streamlit-based movie recommendation application that provides two independent recommendation approaches:

- 🎯 **Content-Based Recommendation** using TF-IDF and cosine similarity
- 🤝 **Collaborative Filtering** using MovieLens 100K data

The application provides an interactive interface for discovering movies, viewing recommendations, exploring ratings and genres, and downloading recommendation results.

---

## ✨ Features

### 🎯 Content-Based Recommendations

Recommends movies based on their content and metadata.

- Movie search and selection
- Movie title disambiguation using release year
- TF-IDF vectorization
- Cosine similarity
- Genre and overview-based similarity
- Recommendation similarity scores
- Movie posters and ratings
- Top-rated movie catalog
- Download recommendations as CSV

### 🤝 Collaborative Filtering

Uses the MovieLens 100K dataset to demonstrate multiple collaborative filtering approaches:

- Item-based collaborative filtering
- User-based collaborative filtering
- Popularity-based fallback
- MovieLens rating data
- User/movie interaction analysis

### 📊 Interactive Dashboard

Built with Streamlit and provides:

- Recommendation mode selection
- Number of recommendations control
- Movie search and selection
- Recommendation cards
- Movie ratings
- Genre information
- Similarity scores
- Downloadable recommendation results
- Top-rated movie exploration

---

## 🧠 Recommendation Approaches

### 1. Content-Based Filtering

The content-based pipeline represents movie information using TF-IDF features and calculates similarity using cosine similarity.

```text
Movie Metadata
      ↓
Text Feature Preparation
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Most Similar Movies
      ↓
Recommendations
````

The system uses movie overview and genre information to identify movies with similar content.

---

### 2. Collaborative Filtering

The collaborative filtering pipeline uses MovieLens 100K rating data.

```text
MovieLens Ratings
      ↓
User-Movie Interactions
      ↓
Collaborative Filtering
      ↓
Item/User Similarity
      ↓
Popularity Fallback
      ↓
Recommendations
```

The application keeps the collaborative filtering approach separate from the content-based pipeline.

> This project does not implement a hybrid recommendation model or matrix factorization.

---

## 🏗️ System Architecture

```text
                    Movie Recommendation System
                              │
                ┌─────────────┴─────────────┐
                │                           │
        Content-Based Mode         Collaborative Mode
                │                           │
        TMDB Movie Dataset           MovieLens 100K
                │                           │
        Text Feature Processing      Rating Processing
                │                           │
          TF-IDF Features          User/Movie Interactions
                │                           │
       Cosine Similarity          Collaborative Filtering
                │                           │
                └─────────────┬─────────────┘
                              │
                    Streamlit Application
                              │
                ┌─────────────┴─────────────┐
                │                           │
          Recommendations            Movie Information
                │                           │
        Similarity / Ratings         Genres / Posters
                │                           │
                └─────────────┬─────────────┘
                              │
                     CSV Download
```

---

## 📁 Project Structure

```text
Movie-Recommendation-System/
│
├── README.md
├── LICENSE
├── .gitignore
├── app.py
├── config.py
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── content_based_recommender.py
│   ├── collaborative_filtering.py
│   └── evaluation.py
│
├── data/
│   └── README.md
│
└── tests/
    └── test_recommender.py
```

---

## 📂 Dataset Setup

The datasets are intentionally **not included in this repository** because of their size and distribution considerations.

### TMDB Movie Dataset

Download:

```text
TMDB_movie_dataset_v11.csv
```

Place it at:

```text
data/TMDB_movie_dataset_v11.csv
```

This dataset is used by the content-based recommendation pipeline.

### MovieLens 100K

Download the MovieLens 100K dataset and place the required files at:

```text
data/ml-100k/u.data
data/ml-100k/u.item
```

More detailed dataset setup instructions are available in:

```text
data/README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/katrinak24/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the datasets

Follow the instructions in:

```text
data/README.md
```

---

## ▶️ Run the Application

From the project root:

```bash
streamlit run app.py
```

The Streamlit interface will open in your browser.

---

## 🧪 Testing

The repository includes automated tests for the recommendation components.

Run:

```bash
pytest
```

or:

```bash
python -m pytest
```

---

## 📊 Evaluation

The project includes an evaluation module for assessing recommendation performance.

The evaluation workflow is maintained separately from the Streamlit application so that recommendation logic and evaluation can be tested independently.

---

## 🛠️ Tech Stack

| Technology     | Purpose                         |
| -------------- | ------------------------------- |
| Python         | Core programming language       |
| Streamlit      | Interactive web application     |
| Pandas         | Data processing                 |
| NumPy          | Numerical operations            |
| Scikit-learn   | TF-IDF and cosine similarity    |
| MovieLens 100K | Collaborative filtering dataset |
| TMDB Dataset   | Content-based movie information |
| Pytest         | Testing                         |

---

## 🔍 How It Works

### Content-Based Mode

1. User searches for or selects a movie.
2. Movie metadata is processed.
3. Text features are transformed using TF-IDF.
4. Cosine similarity is calculated between movies.
5. The most similar movies are selected.
6. Recommendations are displayed with movie information and similarity scores.

### Collaborative Mode

1. MovieLens rating data is loaded.
2. User/movie interactions are processed.
3. Collaborative relationships are calculated.
4. Item-based or user-based recommendations are generated.
5. A popularity-based fallback can be used when sufficient interaction information is unavailable.

---

## 🎨 Application Interface

The Streamlit application is designed to provide an interactive movie discovery experience rather than only returning a list of movie titles.

Users can:

* Search for movies
* Select a specific movie
* Choose the recommendation approach
* Control the number of recommendations
* View movie ratings
* Explore genres
* View similarity scores
* Browse highly rated movies
* Download recommendation results

---

## 📌 Important Notes

* The content-based and collaborative recommendation modes are implemented as separate pipelines.
* The project does not claim to implement a hybrid recommender.
* Matrix factorization is not part of the current implementation.
* Large datasets are excluded from the GitHub repository.
* Dataset files must be placed locally according to `data/README.md`.
* Recommendations are intended for demonstration and educational purposes.

---

## 🚀 Future Improvements

Potential future extensions include:

* Hybrid recommendation strategies
* Matrix factorization
* Neural recommendation models
* Personalized recommendation profiles
* Improved ranking algorithms
* Additional movie metadata
* More comprehensive recommendation evaluation
* Deployment with a hosted Streamlit environment

---

## 👩‍💻 Author

### Katrina Kaur

**BCA (Cloud Computing) Student | AI/ML Developer | Python | Computer Vision | GIS & AgriTech**

GitHub:
[https://github.com/katrinak24](https://github.com/katrinak24)

LinkedIn:
[https://www.linkedin.com/in/katrinak24/](https://www.linkedin.com/in/katrinak24/)

---

⭐ If you find this project useful, feel free to explore the repository and its recommendation pipelines.

````

After that, send me the screenshot of the updated README. We'll do the final repository check before touching the local copy.
