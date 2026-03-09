.PHONY: install sample-data train evaluate predict correlate app test

install:
	python -m pip install -r requirements.txt

sample-data:
	python scripts/make_sample_dataset.py --rows 5000

train:
	python scripts/train_model.py --config configs/default.yaml

evaluate:
	python scripts/evaluate_model.py --model-path models/financial-news-transformer --data-path data/sample_data/financial_news_sample.csv

predict:
	python scripts/batch_predict.py --model-path models/financial-news-transformer --data-path data/sample_data/financial_news_sample.csv

correlate:
	python scripts/run_correlation.py --data-path outputs/batch_predictions.csv --ticker AAPL --model-path models/financial-news-transformer

app:
	python -m streamlit run app/streamlit_app.py

test:
	pytest -q
