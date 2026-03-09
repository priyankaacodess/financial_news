import pandas as pd

from financial_news_sentiment.features import add_engineered_features, extract_ticker_mentions, keyword_signal_counts


def test_extract_ticker_mentions() -> None:
    text = "AAPL rallies while $MSFT and NVDA remain volatile."
    assert extract_ticker_mentions(text) == ["AAPL", "MSFT", "NVDA"]


def test_keyword_signal_counts() -> None:
    text = "Analysts upgrade outlook after strong growth but warn of lawsuit risk."
    scores = keyword_signal_counts(text)
    assert scores["positive_keyword_hits"] >= 2
    assert scores["negative_keyword_hits"] >= 1


def test_add_engineered_features() -> None:
    df = pd.DataFrame({"text": ["Tesla reports record growth and bullish guidance."]})
    out = add_engineered_features(df, text_col="text")
    assert {"entity_mentions", "ticker_mentions", "keyword_signal_score"}.issubset(set(out.columns))
