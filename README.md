# CineMovie — Movie Recommendation System

A content-based movie recommendation system built with Python, scikit-learn, and Streamlit using the TMDB 5000 Movie Dataset.

## Project Overview

CineMovie recommends movies that are similar to a selected movie by analyzing movie metadata and measuring content similarity.

The recommendation pipeline is:

```text
Movie metadata
      ↓
Data cleaning & feature engineering
      ↓
Combined movie tags
      ↓
TF-IDF vectorization
      ↓
Cosine similarity
      ↓
Top-N movie recommendations
      ↓
Streamlit application
```

## Features

- Data exploration and preprocessing
- Movie metadata feature engineering
- Combined `overview`, `genres`, `keywords`, `cast`, and `crew` tags
- TF-IDF vectorization with a 5,000-feature vocabulary
- Cosine-similarity based recommendations
- Configurable Top-N recommendations
- Invalid and empty-input handling
- Evaluation of relevance, coverage, diversity, latency, and failure cases
- Interactive Streamlit interface

## Recommendation Model

The baseline model uses `TfidfVectorizer` with:

- `max_features=5000`
- English stop-word removal

Cosine similarity is then calculated between movie vectors. The system returns the highest-scoring movies for the selected title while excluding the selected movie itself.

## Evaluation

Because the dataset does not contain user interaction or explicit relevance labels, standard ranking metrics such as Precision@K and Recall@K are not directly applicable without introducing additional assumptions.

The baseline was therefore evaluated using:

- Representative qualitative relevance tests
- Catalogue coverage across selected evaluation queries
- Recommendation diversity
- Recommendation latency
- Invalid-title and empty-input handling

The model produced particularly strong content matches for titles such as *The Dark Knight Rises* and *Toy Story*, while some titles showed weaker semantic matching, demonstrating the limitations of purely lexical TF-IDF similarity.

## Limitations

- TF-IDF captures lexical overlap rather than deeper semantic meaning.
- Recommendation quality depends on the available movie metadata.
- The system does not use user ratings, watch history, clicks, likes, or dislikes.
- Recommendations are not personalized to individual users.
- Recommendation quality varies across movies.

## Future Improvements

Potential improvements include:

- Comparing TF-IDF with a Bag-of-Words representation
- Stemming or lemmatization
- Semantic or sentence embeddings
- Field-aware weighting of movie metadata
- Hybrid recommendation using collaborative filtering
- Personalized recommendations using user behaviour
- Diversity-aware re-ranking
- More comprehensive evaluation with user relevance data
- Persisted model artifacts for production deployment

## Project Structure

```text
CineMovie-Recommendation-System/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   └── cleaned_movies.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_recommendation_model.ipynb
├── src/
├── tests/
└── artifacts/
```

## Running the Application

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Streamlit

From the project root:

```bash
streamlit run app.py
```

The application opens in the browser and allows the user to select a movie and choose the number of recommendations.

## Dataset

The project uses the TMDB 5000 Movie Dataset. The processed `cleaned_movies.csv` file is included in the repository for reproducibility of the recommendation application.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Jupyter Notebook
