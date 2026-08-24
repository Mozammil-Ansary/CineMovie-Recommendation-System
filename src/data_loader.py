"""Dataset loading utilities for CineMovie."""

from pathlib import Path

import pandas as pd


def load_movies(path: str | Path) -> pd.DataFrame:
    """Load the TMDB movie metadata CSV."""
    return pd.read_csv(path)


def load_credits(path: str | Path) -> pd.DataFrame:
    """Load the TMDB credits CSV."""
    return pd.read_csv(path)


def load_dataset(movies_path: str | Path, credits_path: str | Path) -> pd.DataFrame:
    """Load and merge movie metadata with cast/crew information.

    The merge uses the stable numeric movie identifier rather than title,
    avoiding title collisions and formatting differences.
    """
    movies = load_movies(movies_path)
    credits = load_credits(credits_path)

    required_movie_columns = {"id", "title"}
    required_credit_columns = {"movie_id", "title"}
    missing_movie = required_movie_columns - set(movies.columns)
    missing_credit = required_credit_columns - set(credits.columns)
    if missing_movie or missing_credit:
        raise ValueError(
            f"Missing required columns. movies={sorted(missing_movie)}, "
            f"credits={sorted(missing_credit)}"
        )

    return movies.merge(
        credits,
        left_on="id",
        right_on="movie_id",
        how="inner",
        suffixes=("", "_credits"),
        validate="one_to_one",
    )
