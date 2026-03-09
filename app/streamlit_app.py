from __future__ import annotations

from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from financial_news_sentiment.correlation import run_correlation_analysis
from financial_news_sentiment.features import add_engineered_features
from financial_news_sentiment.inference import SentimentPredictor

st.set_page_config(
    page_title="Financial News Sentiment Lab",
    page_icon="📈",
    layout="wide",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif;
            background: radial-gradient(circle at 0% 0%, #f8f2dc 0, #fffaf0 35%, #eef6ff 100%);
            color: #0f172a;
        }
        .hero {
            background: linear-gradient(120deg, #09203f 0%, #537895 100%);
            border-radius: 18px;
            padding: 24px;
            color: #f8fafc;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25);
            margin-bottom: 18px;
            animation: fadeIn 700ms ease-in-out;
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
        }
        .hero p {
            margin: 8px 0 0 0;
            max-width: 860px;
            opacity: 0.95;
        }
        .stMetric {
            background: rgba(255,255,255,0.75);
            border: 1px solid rgba(2, 6, 23, 0.08);
            border-radius: 12px;
            padding: 8px;
        }
        code, pre {
            font-family: 'IBM Plex Mono', monospace !important;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px);}
            to { opacity: 1; transform: translateY(0);}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_predictor(model_name_or_path: str) -> SentimentPredictor:
    return SentimentPredictor(model_name_or_path=model_name_or_path)


def sentiment_color(label: str) -> str:
    if label == "positive":
        return "#0f766e"
    if label == "negative":
        return "#be123c"
    return "#475569"


def render_single_article_tab(predictor: SentimentPredictor) -> None:
    st.subheader("Single Article Test")
    text = st.text_area(
        "Paste a financial news headline/article",
        value="NVIDIA beats earnings estimates as AI chip demand surges and guidance is raised.",
        height=150,
    )
    if st.button("Analyze Sentiment", type="primary"):
        with st.spinner("Scoring text with transformer model..."):
            base_df = pd.DataFrame({"text": [text]})
            feature_df = add_engineered_features(base_df, text_col="text")
            pred = predictor.predict([text]).iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Label", str(pred["predicted_label"]).title())
        c2.metric("Confidence", f"{float(pred['confidence']):.2%}")
        c3.metric("Sentiment Score", f"{float(pred['sentiment_score']):.2f}")
        c4.metric("Keyword Signal", int(feature_df["keyword_signal_score"].iloc[0]))

        label = pred["predicted_label"]
        st.markdown(
            f"<div style='padding:12px;border-radius:10px;background:{sentiment_color(label)}12;"
            f"border-left:4px solid {sentiment_color(label)};'>"
            f"<b>Interpretation:</b> Model sentiment is <b>{label.upper()}</b> for this text."
            f"</div>",
            unsafe_allow_html=True,
        )

        prob_df = pd.DataFrame(
            {
                "Class": ["Negative", "Neutral", "Positive"],
                "Probability": [pred["score_negative"], pred["score_neutral"], pred["score_positive"]],
            }
        )
        chart = (
            alt.Chart(prob_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Class:N", sort=["Negative", "Neutral", "Positive"]),
                y=alt.Y("Probability:Q", axis=alt.Axis(format="%")),
                color=alt.Color("Class:N", scale=alt.Scale(range=["#be123c", "#64748b", "#0f766e"]), legend=None),
                tooltip=["Class", alt.Tooltip("Probability:Q", format=".3f")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)


def _prepare_uploaded_dataframe(uploaded_file) -> pd.DataFrame:
    raw_df = pd.read_csv(uploaded_file)
    if "text" not in raw_df.columns:
        fallback_cols = [c for c in ["headline", "title", "content", "body"] if c in raw_df.columns]
        if not fallback_cols:
            raise ValueError("CSV must contain `text` or one of headline/title/content/body.")
        raw_df["text"] = raw_df[fallback_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    return raw_df


def render_batch_tab(predictor: SentimentPredictor) -> None:
    st.subheader("Batch CSV Scoring")
    st.caption("Upload a CSV with columns like `text`, `published_at`, `ticker`, and optional `label`.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")

    if uploaded is not None:
        with st.spinner("Running feature engineering and model inference..."):
            df = _prepare_uploaded_dataframe(uploaded)
            feature_df = add_engineered_features(df, text_col="text")
            pred_df = predictor.predict(feature_df["text"].tolist())
            output = pd.concat([feature_df.reset_index(drop=True), pred_df], axis=1)
            st.session_state["latest_predictions"] = output

        st.success(f"Scored {len(output)} rows.")
        st.dataframe(output.head(20), use_container_width=True)

        dist = output["predicted_label"].value_counts().rename_axis("label").reset_index(name="count")
        donut = (
            alt.Chart(dist)
            .mark_arc(innerRadius=70)
            .encode(
                theta="count:Q",
                color=alt.Color(
                    "label:N",
                    scale=alt.Scale(domain=["negative", "neutral", "positive"], range=["#be123c", "#64748b", "#0f766e"]),
                    legend=alt.Legend(title="Sentiment"),
                ),
                tooltip=["label", "count"],
            )
            .properties(height=320)
        )
        st.altair_chart(donut, use_container_width=True)

        st.download_button(
            label="Download Predictions CSV",
            data=output.to_csv(index=False).encode("utf-8"),
            file_name="financial_news_predictions.csv",
            mime="text/csv",
        )


def render_correlation_tab() -> None:
    st.subheader("Sentiment vs Short-Term Stock Movement")
    default_df = st.session_state.get("latest_predictions")
    data_source = st.radio(
        "Choose sentiment data source",
        ["Use latest batch predictions", "Upload sentiment CSV"],
        horizontal=True,
    )

    sentiment_df = None
    if data_source == "Use latest batch predictions":
        sentiment_df = default_df
    else:
        uploaded = st.file_uploader("Upload sentiment CSV", type=["csv"], key="corr_csv")
        if uploaded is not None:
            sentiment_df = pd.read_csv(uploaded)

    if sentiment_df is None or len(sentiment_df) == 0:
        st.info("Load predictions first in Batch CSV tab, or upload a sentiment CSV.")
        return

    required = {"ticker", "published_at", "sentiment_score"}
    missing = sorted(required - set(sentiment_df.columns))
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    sentiment_df["published_at"] = pd.to_datetime(sentiment_df["published_at"], errors="coerce")
    tickers = sorted(sentiment_df["ticker"].dropna().astype(str).str.upper().unique().tolist())
    if not tickers:
        st.error("No tickers found in the sentiment data.")
        return

    c1, c2 = st.columns([2, 1])
    ticker = c1.selectbox("Ticker", tickers, index=0)
    lag_days = c2.slider("Lag (days)", min_value=1, max_value=5, value=1)

    if st.button("Run Correlation Analysis", type="primary"):
        with st.spinner("Fetching market data and computing correlation..."):
            try:
                result, merged = run_correlation_analysis(
                    sentiment_df=sentiment_df,
                    ticker=ticker,
                    date_col="published_at",
                    ticker_col="ticker",
                    sentiment_col="sentiment_score",
                    lag_days=lag_days,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return

        m1, m2, m3 = st.columns(3)
        m1.metric("Pearson", f"{result.pearson:.4f}")
        m2.metric("Spearman", f"{result.spearman:.4f}")
        m3.metric("Observations", result.observations)

        long_df = merged.melt(id_vars=["date"], value_vars=["daily_sentiment", "future_return"], var_name="series")
        chart = (
            alt.Chart(long_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("value:Q", title="Value"),
                color=alt.Color("series:N", scale=alt.Scale(range=["#0f766e", "#1d4ed8"])),
                tooltip=["date:T", "series:N", alt.Tooltip("value:Q", format=".4f")],
            )
            .properties(height=340)
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(merged.tail(20), use_container_width=True)


def main() -> None:
    apply_styles()
    st.markdown(
        """
        <div class="hero">
          <h1>Financial News Sentiment Analysis Lab</h1>
          <p>Transformer-powered sentiment scoring for financial news with engineered market signals and
          sentiment-to-return correlation analysis for real tickers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_default = "models/financial-news-transformer" if Path("models/financial-news-transformer").exists() else "ProsusAI/finbert"
    model_path = st.sidebar.text_input("Model path or HuggingFace model id", value=model_default)
    st.sidebar.caption("Tip: train your own model first, then point this to your `models/...` directory.")
    predictor = load_predictor(model_path)

    tab1, tab2, tab3 = st.tabs(["Live Test", "Batch CSV", "Correlation"])
    with tab1:
        render_single_article_tab(predictor)
    with tab2:
        render_batch_tab(predictor)
    with tab3:
        render_correlation_tab()


if __name__ == "__main__":
    main()
