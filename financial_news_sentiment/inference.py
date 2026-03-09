from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

SENTIMENT_TO_SCORE = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


def _normalize_label(label: str) -> str:
    normalized = str(label).strip().lower()
    if normalized in SENTIMENT_TO_SCORE:
        return normalized
    if normalized.endswith("_0") or normalized == "label_0":
        return "negative"
    if normalized.endswith("_1") or normalized == "label_1":
        return "neutral"
    if normalized.endswith("_2") or normalized == "label_2":
        return "positive"
    return normalized


def _normalize_scores(raw_scores: list[dict] | dict) -> dict[str, float]:
    rows = raw_scores if isinstance(raw_scores, list) else [raw_scores]
    scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    for row in rows:
        key = _normalize_label(row["label"])
        if key in scores:
            scores[key] = float(row["score"])
    return scores


@dataclass
class SentimentPredictor:
    model_name_or_path: str = "ProsusAI/finbert"
    device: int = -1

    def __post_init__(self) -> None:
        from transformers import pipeline

        self._pipeline = pipeline(
            "text-classification",
            model=self.model_name_or_path,
            tokenizer=self.model_name_or_path,
            return_all_scores=True,
            device=self.device,
            truncation=True,
        )

    def predict(self, texts: Iterable[str], batch_size: int = 16) -> pd.DataFrame:
        cleaned_texts = [str(text or "").strip() for text in texts]
        if not cleaned_texts:
            return pd.DataFrame(
                columns=[
                    "predicted_label",
                    "confidence",
                    "sentiment_score",
                    "score_negative",
                    "score_neutral",
                    "score_positive",
                ]
            )
        raw = self._pipeline(cleaned_texts, batch_size=batch_size)
        rows = []
        for output in raw:
            score_map = _normalize_scores(output)
            label = max(score_map, key=score_map.get)
            rows.append(
                {
                    "predicted_label": label,
                    "confidence": score_map[label],
                    "sentiment_score": SENTIMENT_TO_SCORE.get(label, 0.0),
                    "score_negative": score_map["negative"],
                    "score_neutral": score_map["neutral"],
                    "score_positive": score_map["positive"],
                }
            )
        return pd.DataFrame(rows)
