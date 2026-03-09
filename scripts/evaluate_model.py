from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from financial_news_sentiment.data import load_news_csv
from financial_news_sentiment.features import add_engineered_features
from financial_news_sentiment.inference import SentimentPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained sentiment model.")
    parser.add_argument("--model-path", type=str, required=True, help="Local model folder path.")
    parser.add_argument("--data-path", type=str, required=True, help="Evaluation CSV path.")
    parser.add_argument("--text-col", type=str, default="text", help="Text column name.")
    parser.add_argument("--label-col", type=str, default="label", help="Ground truth label column name.")
    parser.add_argument("--output", type=str, default="outputs/eval_metrics.json", help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_news_csv(args.data_path, text_col=args.text_col)
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    df = add_engineered_features(df, text_col=args.text_col)
    predictor = SentimentPredictor(model_name_or_path=args.model_path)
    pred_df = predictor.predict(df[args.text_col].tolist())
    eval_df = df.reset_index(drop=True).join(pred_df)

    y_true = eval_df[args.label_col].astype(str).str.lower()
    y_pred = eval_df["predicted_label"]
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "classification_report": report,
                "confusion_matrix": matrix.tolist(),
                "labels": sorted(set(y_true)),
            },
            handle,
            indent=2,
        )
    print(f"Saved evaluation metrics to {output_path}")
    print(f"Accuracy: {report.get('accuracy', 'n/a')}")


if __name__ == "__main__":
    main()
