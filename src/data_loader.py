"""Dataset loading utilities."""

from pathlib import Path
import pandas as pd


def load_movies(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_credits(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
