# 🎬 Movie Recommendation System

A content-based movie recommender built with **Python** and **Streamlit**.
It suggests 5 movies similar to the one you pick, and fetches posters from the **TMDB API**.

## Features
- Content-based filtering using cosine similarity on movie metadata
- Clean interactive UI with Streamlit
- Live poster fetching from TMDB
- Cached artifacts (`.pkl`) for instant recommendations

## Tech Stack
| Layer | Tool |
|-------|------|
| Language | Python 3.9+ |
| UI | Streamlit |
| Data | Pandas |
| ML | Scikit-learn (`CountVectorizer`, `cosine_similarity`) |
| Persistence | Pickle |
| Posters | TMDB API |

## Project Structure
```
movie_recommender/
├── app.py              # Streamlit app
├── build_model.py      # Builds movies.pkl + similarity.pkl from raw CSVs
├── requirements.txt
├── README.md
├── raw/                # Place the TMDB 5000 CSVs here
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
└── data/               # Generated artifacts
    ├── movies.pkl
    └── similarity.pkl
```

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the dataset**
   Get the TMDB 5000 Movie Dataset from Kaggle:
   https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

   Put both CSV files inside a `raw/` folder.

3. **Build the model artifacts**
   ```bash
   python build_model.py
   ```
   This creates `data/movies.pkl` and `data/similarity.pkl`.

4. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Open http://localhost:8501 in your browser.

## TMDB API Key
The app ships with a public demo key. For production use, get your own
free key from https://www.themoviedb.org/settings/api and replace
`TMDB_API_KEY` in `app.py`.

## How It Works
1. Combine `genres + keywords + cast + director + overview` into a single tag string per movie.
2. Vectorize all tags with `CountVectorizer` (top 5000 features, English stop-words removed).
3. Compute a 4800×4800 cosine similarity matrix.
4. For a chosen movie, return the 5 highest-similarity titles (excluding itself).

## License
MIT — free to use and modify.
