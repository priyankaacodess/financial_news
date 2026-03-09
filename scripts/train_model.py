from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from financial_news_sentiment.config import load_project_config
from financial_news_sentiment.data import load_news_csv, train_val_test_split
from financial_news_sentiment.features import add_engineered_features
from financial_news_sentiment.inference import SentimentPredictor
from financial_news_sentiment.modeling import train_transformer_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transformer model for financial news sentiment.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--data-path", type=str, default=None, help="Override data path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override model output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_project_config(args.config)

    data_path = args.data_path or cfg.data.data_path
    output_dir = args.output_dir or cfg.model.output_dir

    df = load_news_csv(
        data_path,
        text_col=cfg.data.text_col,
        date_col=cfg.data.date_col,
        ticker_col=cfg.data.ticker_col,
    )
    if cfg.data.label_col not in df.columns:
        raise ValueError(f"'{cfg.data.label_col}' is required in the dataset for training.")

    df = add_engineered_features(df, text_col=cfg.data.text_col)
    train_df, val_df, test_df = train_val_test_split(
        df,
        label_col=cfg.data.label_col,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        random_state=cfg.data.random_state,
    )

    train_output = train_transformer_classifier(
        train_df=train_df,
        val_df=val_df,
        text_col=cfg.data.text_col,
        label_col=cfg.data.label_col,
        output_dir=output_dir,
        model_name=cfg.model.model_name,
        max_length=cfg.model.max_length,
        learning_rate=cfg.model.learning_rate,
        batch_size=cfg.model.batch_size,
        epochs=cfg.model.epochs,
        weight_decay=cfg.model.weight_decay,
    )

    predictor = SentimentPredictor(model_name_or_path=output_dir)
    test_preds = predictor.predict(test_df[cfg.data.text_col].tolist())
    eval_df = test_df.reset_index(drop=True).join(test_preds)
    report = classification_report(
        eval_df[cfg.data.label_col].astype(str).str.lower(),
        eval_df["predicted_label"],
        output_dict=True,
        zero_division=0,
    )

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(outputs_dir / "test_predictions.csv", index=False)
    with (outputs_dir / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"validation_metrics": train_output["metrics"], "test_classification_report": report},
            handle,
            indent=2,
        )

    print(f"Model saved to: {output_dir}")
    print(f"Validation accuracy: {train_output['metrics'].get('eval_accuracy', 'n/a')}")
    print(f"Test accuracy: {report.get('accuracy', 'n/a')}")
    print("Saved outputs/test_predictions.csv and outputs/train_metrics.json")


if __name__ == "__main__":
    main()
