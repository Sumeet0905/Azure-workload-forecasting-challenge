# Final Submission Checklist

## Required artifacts

- [x] `README.md` with setup and one-command forecast reproduction.
- [x] `requirements.txt` and `pyproject.toml`.
- [x] `src/` forecasting implementation.
- [x] `predictions.csv` with the required columns and 24 chronological rows.
- [x] `reports/final_report.md`.
- [x] `reports/final_report.pdf`.
- [x] EDA summaries and figures under `reports/eda/`.
- [x] `tests/` and test configuration.
- [x] `DATASET.md` and `DATA_CARD.md`.

## Before uploading

1. Download the official `data-v1` release from the challenge repository.
2. Extract it to `challenge_data/` and verify `SHA256SUMS`.
3. Create a clean Python 3.11+ environment and install `requirements.txt`.
4. Run `python -m src.predict --data challenge_data --output predictions.csv`.
5. Run `python -m pytest -q`.
6. Inspect the regenerated `predictions.csv` and confirm exactly 24 rows and the exact required column names.
7. Review the report and ensure you can explain the model selection, validation windows, holdout protocol, limitations, and operational recommendation.
8. Do not upload the raw challenge dataset or virtual environment.

## Honest reporting

The bundled predictions and metrics are based on the public `data-v1` release and the stated protocol. The report distinguishes rolling-origin validation from the post-hoc public-holdout audit. Do not present the post-hoc holdout audit as an unseen/private test result.
