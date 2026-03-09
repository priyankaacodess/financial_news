from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _build_label_mapping(labels: pd.Series) -> dict[str, int]:
    normalized = labels.astype(str).str.lower().str.strip()
    canonical_order = ["negative", "neutral", "positive"]
    present = [label for label in canonical_order if label in set(normalized)]
    remaining = sorted(set(normalized) - set(present))
    ordered_labels = present + remaining
    return {label: idx for idx, label in enumerate(ordered_labels)}


def _compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
    }


def train_transformer_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    output_dir: str | Path,
    model_name: str = "distilbert-base-uncased",
    max_length: int = 256,
    learning_rate: float = 2e-5,
    batch_size: int = 8,
    epochs: int = 2,
    weight_decay: float = 0.01,
) -> dict[str, Any]:
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    label2id = _build_label_mapping(train_df[label_col])
    id2label = {idx: label for label, idx in label2id.items()}

    train_data = train_df[[text_col, label_col]].copy()
    val_data = val_df[[text_col, label_col]].copy()
    train_data[label_col] = train_data[label_col].astype(str).str.lower().map(label2id)
    val_data[label_col] = val_data[label_col].astype(str).str.lower().map(label2id)

    train_dataset = Dataset.from_pandas(train_data, preserve_index=False).rename_column(label_col, "labels")
    val_dataset = Dataset.from_pandas(val_data, preserve_index=False).rename_column(label_col, "labels")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)
    train_dataset = train_dataset.remove_columns([text_col])
    val_dataset = val_dataset.remove_columns([text_col])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    args_kwargs = {
        "output_dir": str(output_path),
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "weight_decay": weight_decay,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "logging_steps": 50,
        "save_total_limit": 2,
        "report_to": "none",
    }
    # Transformers changed this arg name across versions.
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in ta_params:
        args_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in ta_params:
        args_kwargs["eval_strategy"] = "epoch"
    else:
        raise RuntimeError("TrainingArguments does not expose evaluation/eval strategy in this transformers version.")

    args = TrainingArguments(**args_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "compute_metrics": _compute_metrics,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    with (output_path / "label_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(label2id, handle, indent=2)

    return {"metrics": metrics, "label2id": label2id, "output_dir": str(output_path)}
