from __future__ import annotations

import re
from collections import Counter

import pandas as pd

POSITIVE_KEYWORDS = {
    "beat",
    "bullish",
    "buyback",
    "growth",
    "outperform",
    "record",
    "rebound",
    "surge",
    "upgrade",
    "upside",
}
NEGATIVE_KEYWORDS = {
    "bearish",
    "cut",
    "decline",
    "downgrade",
    "fraud",
    "lawsuit",
    "loss",
    "miss",
    "selloff",
    "warning",
}
TICKER_PATTERN = re.compile(r"\$?[A-Z]{1,5}\b")
ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")


def extract_ticker_mentions(text: str) -> list[str]:
    if not text:
        return []
    mentions = TICKER_PATTERN.findall(text)
    cleaned = [token.replace("$", "") for token in mentions if token.replace("$", "").isalpha()]
    return cleaned


def count_entity_mentions(text: str) -> int:
    if not text:
        return 0
    candidates = ENTITY_PATTERN.findall(text)
    # Remove first word if it likely represents sentence start and not a named entity.
    filtered = [name for name in candidates if len(name) > 2]
    return len(filtered)


def keyword_signal_counts(text: str) -> dict[str, int]:
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    token_counts = Counter(tokens)
    pos_hits = sum(token_counts.get(word, 0) for word in POSITIVE_KEYWORDS)
    neg_hits = sum(token_counts.get(word, 0) for word in NEGATIVE_KEYWORDS)
    return {
        "positive_keyword_hits": int(pos_hits),
        "negative_keyword_hits": int(neg_hits),
        "keyword_signal_score": int(pos_hits - neg_hits),
    }


def _normalize_finbert_output(item: list[dict] | dict) -> dict[str, float]:
    if isinstance(item, dict):
        records = [item]
    else:
        records = item
    out = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for record in records:
        label = str(record["label"]).strip().lower()
        if label.startswith("label_"):
            continue
        if label in out:
            out[label] = float(record["score"])
    return out


def finbert_sentiment_scores(
    texts: list[str],
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 16,
) -> pd.DataFrame:
    if not texts:
        return pd.DataFrame(columns=["finbert_positive", "finbert_negative", "finbert_neutral", "finbert_label"])

    from transformers import pipeline

    clf = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        return_all_scores=True,
        truncation=True,
    )
    outputs = clf(texts, batch_size=batch_size)
    rows = []
    for item in outputs:
        score_map = _normalize_finbert_output(item)
        label = max(score_map, key=score_map.get)
        rows.append(
            {
                "finbert_positive": score_map["positive"],
                "finbert_negative": score_map["negative"],
                "finbert_neutral": score_map["neutral"],
                "finbert_label": label,
                "finbert_confidence": score_map[label],
            }
        )
    return pd.DataFrame(rows)


def add_engineered_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in dataframe.")
    feature_df = df.copy()

    keyword_features = feature_df[text_col].astype(str).apply(keyword_signal_counts).apply(pd.Series)
    feature_df = pd.concat([feature_df, keyword_features], axis=1)

    feature_df["entity_mentions"] = feature_df[text_col].astype(str).apply(count_entity_mentions)
    feature_df["ticker_mentions"] = feature_df[text_col].astype(str).apply(lambda x: len(extract_ticker_mentions(x)))
    feature_df["char_length"] = feature_df[text_col].astype(str).str.len()
    feature_df["word_count"] = feature_df[text_col].astype(str).str.split().str.len()
    return feature_df
