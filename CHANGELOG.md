# Changelog

## 2.0.0 — 2026-11

Complete rewrite: from a notebook exercise to a tested forecasting package.

### Data
- 26 years of real daily weather for Marrakech (Open-Meteo / ERA5, 2000–2026, 9,813 days)
  instead of 20 hand-written rows without dates.
- `tempforecast fetch` downloads the latest archive and the provisional last days.

### Model
- Forecasts the daily **maximum and minimum** temperature **1 to 7 days ahead**.
- Leak-free features: recent temperatures, rolling averages, rain, wind, sunshine, season.
- Two baselines (persistence, climatology) and three learned models (ridge, gradient boosting, ensemble).
- 80% uncertainty band with conformal prediction, checked on the test years (≈ 80% coverage).

### Evaluation
- Chronological split (train 2000–2022, test 2023–2026) instead of a random shuffle.
- Tomorrow's max temperature: MAE 6.82 °C (v1) → 1.81 °C (v2).

### Delivery
- Python package with a CLI, a bilingual Streamlit dashboard and an executed notebook.
- Tests (including a quality gate on the real data and a headless dashboard run), Ruff, GitHub Actions.

## 1.0.0

Linear regression of the temperature on the day of the week, in a Google Colab notebook.
