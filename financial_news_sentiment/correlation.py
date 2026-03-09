from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd


@dataclass
class CorrelationResult:
    ticker: str
    observations: int
    lag_days: int
    pearson: float
    spearman: float


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    out = df.copy()
    out.columns = [
        "_".join(str(part) for part in col if part is not None and str(part) != "").strip("_")
        for col in out.columns.to_flat_index()
    ]
    return out


def _find_column(columns: list[str], *candidates: str) -> str | None:
    lower_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        hit = lower_map.get(candidate.lower())
        if hit:
            return hit
    return None


def _resolve_close_column(columns: list[str], ticker: str) -> str | None:
    ticker = ticker.upper()
    direct = _find_column(
        columns,
        "close",
        f"close_{ticker}",
        f"{ticker}_close",
        "adj close",
        "adj_close",
        f"adj close_{ticker}",
        f"adj_close_{ticker}",
        f"{ticker}_adj close",
        f"{ticker}_adj_close",
    )
    if direct:
        return direct
    for col in columns:
        lowered = col.lower()
        if "close" in lowered and "volume" not in lowered:
            return col
    return None


def fetch_price_history(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "yfinance is required for correlation analysis. Install it with: python -m pip install yfinance"
        ) from exc

    price = yf.download(
        ticker,
        start=(start - timedelta(days=5)).strftime("%Y-%m-%d"),
        end=(end + timedelta(days=5)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )
    if price.empty:
        raise ValueError(f"No market data found for ticker: {ticker}")

    price = _flatten_columns(price).reset_index()
    price = _flatten_columns(price)

    date_col = _find_column(price.columns.tolist(), "date", "datetime")
    close_col = _resolve_close_column(price.columns.tolist(), ticker=ticker)
    if date_col is None or close_col is None:
        raise ValueError(
            f"Could not identify required market columns for {ticker}. "
            f"Available columns: {list(price.columns)}"
        )

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(price[date_col], errors="coerce"),
            "close": pd.to_numeric(price[close_col], errors="coerce"),
        }
    ).dropna()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    frame["return_1d"] = frame["close"].pct_change()
    return frame[["date", "close", "return_1d"]].dropna()


def run_correlation_analysis(
    sentiment_df: pd.DataFrame,
    ticker: str,
    date_col: str = "published_at",
    ticker_col: str = "ticker",
    sentiment_col: str = "sentiment_score",
    lag_days: int = 1,
) -> tuple[CorrelationResult, pd.DataFrame]:
    if date_col not in sentiment_df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    if sentiment_col not in sentiment_df.columns:
        raise ValueError(f"Missing sentiment score column: {sentiment_col}")
    if ticker_col not in sentiment_df.columns:
        raise ValueError(f"Missing ticker column: {ticker_col}")

    filtered = sentiment_df[sentiment_df[ticker_col].astype(str).str.upper() == ticker.upper()].copy()
    if filtered.empty:
        raise ValueError(f"No sentiment records found for ticker {ticker}.")

    filtered[date_col] = pd.to_datetime(filtered[date_col], errors="coerce").dt.tz_localize(None)
    filtered = filtered.dropna(subset=[date_col, sentiment_col])
    if filtered.empty:
        raise ValueError("No valid dated sentiment rows after filtering.")

    daily_sentiment = (
        filtered.assign(date=filtered[date_col].dt.floor("D"))
        .groupby("date", as_index=False)[sentiment_col]
        .mean()
        .rename(columns={sentiment_col: "daily_sentiment"})
    )

    prices = fetch_price_history(
        ticker=ticker,
        start=daily_sentiment["date"].min(),
        end=daily_sentiment["date"].max(),
    )
    prices["future_return"] = prices["return_1d"].shift(-lag_days)

    merged = daily_sentiment.merge(prices[["date", "future_return"]], on="date", how="inner").dropna()
    if len(merged) < 3:
        raise ValueError("Insufficient overlapping rows to compute correlation (need at least 3).")

    pearson = float(np.corrcoef(merged["daily_sentiment"], merged["future_return"])[0, 1])
    spearman = float(pd.Series(merged["daily_sentiment"]).corr(merged["future_return"], method="spearman"))

    result = CorrelationResult(
        ticker=ticker.upper(),
        observations=int(len(merged)),
        lag_days=lag_days,
        pearson=pearson,
        spearman=spearman,
    )
    return result, merged
