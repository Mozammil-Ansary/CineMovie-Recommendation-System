import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CineMovie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_movies.csv")


@st.cache_resource
def build_model(tags):
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )
    vectors = tfidf.fit_transform(tags)
    similarity = cosine_similarity(vectors)
    return similarity


movies = load_data()
similarity = build_model(movies["tags"])


st.title("🎬 CineMovie Recommendation System")
st.write(
    "Select a movie to discover similar movies using a content-based "
    "recommendation system."
)

movie_title = st.selectbox(
    "Select a movie",
    movies["title"].tolist()
)

top_n = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=10,
    value=5
)


if st.button("Recommend Movies", type="primary"):
    movie_index = movies[movies["title"] == movie_title].index[0]
    scores = similarity[movie_index]

    sorted_indices = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in sorted_indices[1:top_n + 1]:
        recommendations.append({
            "title": movies.iloc[index]["title"],
            "similarity_score": round(float(score), 4)
        })

    recommendations_df = pd.DataFrame(recommendations)

    st.subheader(f"Recommendations for {movie_title}")
    st.dataframe(
        recommendations_df,
        use_container_width=True,
        hide_index=True
    )
