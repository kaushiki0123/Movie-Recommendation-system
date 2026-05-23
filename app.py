"""
Movie Recommendation System
A content-based movie recommender built with Streamlit.
"""

import pickle
import streamlit as st

# ---------- Page Config ----------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------- Load Data ----------
@st.cache_resource(show_spinner=True)
def load_artifacts():
    """Load movies and similarity matrix."""

    with open("data/movies.pkl", "rb") as f:
        movies = pickle.load(f)

    with open("data/similarity.pkl", "rb") as f:
        similarity = pickle.load(f)

    return movies, similarity


# ---------- Recommendation Function ----------
def recommend(movie_title, movies, similarity, top_n=5):
    """Recommend similar movies."""

    # Get selected movie index
    idx = movies[movies["title"] == movie_title].index[0]

    # Get similarity scores
    distances = list(enumerate(similarity[idx]))

    # Sort movies by similarity
    distances = sorted(
        distances,
        key=lambda x: x[1],
        reverse=True
    )[1: top_n + 1]

    recommended_movies = []

    # Fetch recommended movie titles
    for movie in distances:

        recommended_movies.append(
            movies.iloc[movie[0]].title
        )

    return recommended_movies


# ---------- Main UI ----------
st.title("🎬 Movie Recommendation System")

st.caption(
    "Content-based recommender using cosine similarity on movie metadata."
)

# Load artifacts
movies, similarity = load_artifacts()

# Movie dropdown
selected_movie = st.selectbox(
    "Pick a movie you like and we'll suggest similar ones:",
    movies["title"].values
)

# Recommendation button
if st.button("Recommend", type="primary"):

    with st.spinner("Finding similar movies..."):

        recommendations = recommend(
            selected_movie,
            movies,
            similarity
        )

    st.subheader("Recommended Movies")

    cols = st.columns(5)

    for col, movie_name in zip(cols, recommendations):

        with col:

            st.markdown("## 🎬")

            st.markdown(
                f"**{movie_name}**"
            )


# ---------- About Project ----------
with st.expander("About this project"):

    st.markdown(
        """
### How It Works

1. Movie metadata is combined into tags
2. Text data is vectorized using CountVectorizer
3. Cosine similarity calculates movie similarity
4. Top similar movies are recommended

### Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- NLP
- Cosine Similarity

### Recommendation Technique

This project uses a **content-based filtering approach**
to recommend movies similar to the selected movie.
        """
    )
