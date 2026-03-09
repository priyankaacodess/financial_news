from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from financial_news_sentiment.data import load_news_csv
from financial_news_sentiment.features import add_engineered_features
from financial_news_sentiment.inference import SentimentPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch sentiment prediction on a CSV.")
    parser.add_argument("--model-path", type=str, default="ProsusAI/finbert", help="Model path or HF model id.")
    parser.add_argument("--data-path", type=str, required=True, help="Input CSV path.")
    parser.add_argument("--text-col", type=str, default="text", help="Text column name.")
    parser.add_argument("--output", type=str, default="outputs/batch_predictions.csv", help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_news_csv(args.data_path, text_col=args.text_col)
    df = add_engineered_features(df, text_col=args.text_col)

    predictor = SentimentPredictor(model_name_or_path=args.model_path)
    preds = predictor.predict(df[args.text_col].tolist())
    out_df = pd.concat([df.reset_index(drop=True), preds], axis=1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
