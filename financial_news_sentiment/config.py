from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    data_path: str = "data/sample_data/financial_news_sample.csv"
    text_col: str = "text"
    label_col: str = "label"
    date_col: str = "published_at"
    ticker_col: str = "ticker"
    test_size: float = 0.1
    val_size: float = 0.1
    random_state: int = 42


@dataclass
class ModelConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    learning_rate: float = 2e-5
    batch_size: int = 8
    epochs: int = 2
    weight_decay: float = 0.01
    output_dir: str = "models/financial-news-transformer"


@dataclass
class ProjectConfig:
    data: DataConfig
    model: ModelConfig


def _merge_config(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = default.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_project_config(path: str | Path | None = None) -> ProjectConfig:
    default = {
        "data": DataConfig().__dict__,
        "model": ModelConfig().__dict__,
    }
    if path is None:
        config = default
    else:
        cfg_path = Path(path)
        with cfg_path.open("r", encoding="utf-8") as handle:
            override = yaml.safe_load(handle) or {}
        config = _merge_config(default, override)
    return ProjectConfig(data=DataConfig(**config["data"]), model=ModelConfig(**config["model"]))
