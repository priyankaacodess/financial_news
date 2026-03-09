from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    normalized = html.unescape(str(text))
    normalized = re.sub(r"http[s]?://\S+|www\.\S+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _build_text_column(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    if text_col in df.columns:
        df[text_col] = df[text_col].apply(clean_text)
        return df

    candidate_cols = [col for col in ["headline", "title", "content", "body"] if col in df.columns]
    if not candidate_cols:
        raise ValueError(
            f"Missing '{text_col}' column and no fallback columns found. "
            "Expected one of: text/headline/title/content/body."
        )

    df[text_col] = (
        df[candidate_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df[text_col] = df[text_col].apply(clean_text)
    return df


def load_news_csv(
    path: str | Path,
    text_col: str = "text",
    date_col: str = "published_at",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    df = _build_text_column(df, text_col=text_col)
    df = df[df[text_col].str.len() > 0].reset_index(drop=True)

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if ticker_col in df.columns:
        df[ticker_col] = (
            df[ticker_col]
            .astype(str)
            .str.upper()
            .str.replace("$", "", regex=False)
            .str.strip()
        )
    return df


def train_val_test_split(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' column is required for supervised training.")

    split_kwargs = {
        "test_size": test_size,
        "random_state": random_state,
        "stratify": df[label_col] if df[label_col].nunique() > 1 else None,
    }
    train_val_df, test_df = train_test_split(df, **split_kwargs)

    val_ratio = val_size / (1 - test_size)
    split_kwargs = {
        "test_size": val_ratio,
        "random_state": random_state,
        "stratify": train_val_df[label_col] if train_val_df[label_col].nunique() > 1 else None,
    }
    train_df, val_df = train_test_split(train_val_df, **split_kwargs)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
