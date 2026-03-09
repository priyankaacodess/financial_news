from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from financial_news_sentiment.correlation import run_correlation_analysis
from financial_news_sentiment.data import load_news_csv
from financial_news_sentiment.inference import SentimentPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlate sentiment signal with short-term stock moves.")
    parser.add_argument("--data-path", type=str, required=True, help="News CSV path.")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker to analyze.")
    parser.add_argument("--model-path", type=str, default="ProsusAI/finbert", help="Model path or HF id.")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--date-col", type=str, default="published_at")
    parser.add_argument("--ticker-col", type=str, default="ticker")
    parser.add_argument("--lag-days", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_news_csv(args.data_path, text_col=args.text_col, date_col=args.date_col, ticker_col=args.ticker_col)

    if "sentiment_score" not in df.columns:
        predictor = SentimentPredictor(model_name_or_path=args.model_path)
        predictions = predictor.predict(df[args.text_col].tolist())
        df = df.join(predictions["sentiment_score"])

    result, merged = run_correlation_analysis(
        sentiment_df=df,
        ticker=args.ticker,
        date_col=args.date_col,
        ticker_col=args.ticker_col,
        sentiment_col="sentiment_score",
        lag_days=args.lag_days,
    )
    print(
        f"{result.ticker} | obs={result.observations} | lag={result.lag_days} "
        f"| pearson={result.pearson:.4f} | spearman={result.spearman:.4f}"
    )
    print(merged.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
