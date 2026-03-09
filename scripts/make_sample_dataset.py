from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "JPM", "AMD", "GOOGL", "NFLX"]

POSITIVE_TEMPLATES = [
    "{ticker} beats quarterly estimates as cloud revenue accelerates.",
    "{ticker} announces aggressive buyback plan after record margin expansion.",
    "Analysts upgrade {ticker} on stronger than expected demand outlook.",
    "{ticker} secures strategic partnership expected to boost long-term growth.",
]
NEGATIVE_TEMPLATES = [
    "{ticker} misses earnings guidance and warns about weaker demand.",
    "Regulatory lawsuit raises legal risk for {ticker} and pressures shares.",
    "Analysts downgrade {ticker} after margin contraction and soft pipeline.",
    "{ticker} reports decline in subscriptions and increases restructuring costs.",
]
NEUTRAL_TEMPLATES = [
    "{ticker} holds annual investor day with no major forecast revisions.",
    "Management at {ticker} reiterates full-year guidance during conference call.",
    "{ticker} launches incremental product refresh in line with expectations.",
    "Sector analysts maintain mixed views on {ticker} ahead of earnings.",
]


def _sample_article(ticker: str, label: str) -> tuple[str, str]:
    if label == "positive":
        template = random.choice(POSITIVE_TEMPLATES)
    elif label == "negative":
        template = random.choice(NEGATIVE_TEMPLATES)
    else:
        template = random.choice(NEUTRAL_TEMPLATES)
    headline = template.format(ticker=ticker)
    details = (
        f"Traders monitored {ticker} closely as commentary pointed to {label} sentiment. "
        f"Institutions discussed position sizing and short-term volatility in post-market flows."
    )
    return headline, details


def create_dataset(rows: int) -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)
    labels = np.random.choice(["positive", "neutral", "negative"], size=rows, p=[0.38, 0.24, 0.38])
    start = datetime.utcnow() - timedelta(days=180)

    records = []
    for idx, label in enumerate(labels):
        ticker = random.choice(TICKERS)
        headline, body = _sample_article(ticker, label)
        published_at = start + timedelta(minutes=30 * idx)
        sentiment_score = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}[label]
        next_day_return = np.random.normal(loc=0.012 * sentiment_score, scale=0.02)
        records.append(
            {
                "id": idx + 1,
                "ticker": ticker,
                "published_at": published_at.strftime("%Y-%m-%d %H:%M:%S"),
                "headline": headline,
                "content": body,
                "text": f"{headline} {body}",
                "label": label,
                "simulated_next_day_return": next_day_return,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic financial news dataset.")
    parser.add_argument("--rows", type=int, default=5000, help="Number of rows to generate.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample_data/financial_news_sample.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = create_dataset(rows=args.rows)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
