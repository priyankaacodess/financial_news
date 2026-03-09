from pathlib import Path

import pandas as pd

from financial_news_sentiment.data import clean_text, load_news_csv, train_val_test_split


def test_clean_text() -> None:
    dirty = "Great results!  Visit https://example.com  now."
    assert clean_text(dirty) == "Great results! Visit now."


def test_load_news_csv_builds_text_column(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "headline": ["AAPL beats earnings"],
            "content": ["Revenue grew faster than expected"],
            "ticker": ["aapl"],
        }
    )
    path = tmp_path / "news.csv"
    source.to_csv(path, index=False)
    loaded = load_news_csv(path, text_col="text")
    assert "text" in loaded.columns
    assert loaded.loc[0, "ticker"] == "AAPL"


def test_train_val_test_split_sizes() -> None:
    rows = 100
    df = pd.DataFrame(
        {
            "text": [f"text {i}" for i in range(rows)],
            "label": ["positive" if i % 2 == 0 else "negative" for i in range(rows)],
        }
    )
    train_df, val_df, test_df = train_val_test_split(df, label_col="label", test_size=0.1, val_size=0.1)
    assert len(train_df) + len(val_df) + len(test_df) == rows
