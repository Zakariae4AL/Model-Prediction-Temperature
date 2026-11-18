"""Honest evaluation for time series: train on the past, test on later years."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import HORIZONS, TARGETS, make_targets
from .models import ConformalBand, EnsembleForecaster, all_models

TEST_START = "2023-01-01"


def split(df: pd.DataFrame, test_start: str = TEST_START) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: everything before `test_start` trains, everything after tests.

    Shuffling days (as the v1 notebook did with train_test_split) would let the model
    learn from days that come after the ones it is tested on.
    """
    start = pd.Timestamp(test_start)
    return df[df.index < start], df[df.index >= start]


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    ok = y_true.notna() & y_pred.notna()
    err = (y_pred[ok] - y_true[ok]).to_numpy(dtype=float)
    truth = y_true[ok].to_numpy(dtype=float)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "r2": float(1 - np.sum(err**2) / np.sum((truth - truth.mean()) ** 2)),
        "within_2c": float(np.mean(np.abs(err) <= 2.0)),
        "n": int(ok.sum()),
    }


def evaluate(df: pd.DataFrame, test_start: str = TEST_START, horizons=HORIZONS, models=None) -> pd.DataFrame:
    """Scores of every model, for every target and horizon, on the test years.

    Models are fitted on the training years only. To predict a test day, a model may use
    the observed weather up to the day the forecast is made, exactly like in real use.
    """
    train, test = split(df, test_start)
    models = models if models is not None else all_models()
    # Forecasts for the first test days are made from the last training days.
    context = df[df.index >= pd.Timestamp(test_start) - pd.Timedelta(days=60)]
    rows = []
    for model in models:
        model.fit(train)
        for h in horizons:
            pred = model.predict(context, h)
            truth = make_targets(context, h)
            issued = pred.index + pd.Timedelta(days=h)  # the day being forecast
            in_test = issued >= pd.Timestamp(test_start)
            for t in TARGETS:
                m = metrics(truth.loc[in_test, t], pred.loc[in_test, t])
                rows.append({"model": model.name, "target": t, "horizon": h, **m})
    return pd.DataFrame(rows)


def interval_coverage(
    df: pd.DataFrame, test_start: str = TEST_START, level: float = 0.8, horizons=(1, 3, 7)
) -> pd.DataFrame:
    """How often the real temperature falls inside the predicted band (should be close to `level`).

    The band is calibrated on the last training years only, then checked on the test years.
    """
    train, _ = split(df, test_start)
    band = ConformalBand(level=level).fit(lambda: EnsembleForecaster(tuple(horizons)), train)
    model = EnsembleForecaster(tuple(horizons)).fit(train)
    context = df[df.index >= pd.Timestamp(test_start) - pd.Timedelta(days=60)]
    rows = []
    for h in horizons:
        pred = model.predict(context, h)
        truth = make_targets(context, h)
        in_test = (context.index + pd.Timedelta(days=h)) >= pd.Timestamp(test_start)
        for t in TARGETS:
            err = (pred.loc[in_test, t] - truth.loc[in_test, t]).abs().dropna()
            width = band.half_width[(t, h)]
            rows.append(
                {"target": t, "horizon": h, "coverage": float((err <= width).mean()), "band": f"±{width:.1f} °C"}
            )
    return pd.DataFrame(rows)


def summary(scores: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """One line per model: MAE for tomorrow's max and min temperature, and skill vs persistence."""
    day = scores[scores["horizon"] == horizon].pivot(index="model", columns="target", values="mae")
    day.columns = [f"mae_{c}" for c in day.columns]
    if "Persistence" in day.index:
        for t in TARGETS:
            day[f"skill_{t}"] = 1 - day[f"mae_{t}"] / day.loc["Persistence", f"mae_{t}"]
    return day.sort_values("mae_tmax")
