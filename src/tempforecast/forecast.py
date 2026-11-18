"""Train on all available days and forecast the next week."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features import HORIZONS, TARGETS
from .models import ConformalBand, EnsembleForecaster

LEVEL = 0.8


@dataclass
class TrainedForecaster:
    """The production model and its calibrated uncertainty band."""

    model: EnsembleForecaster
    band: ConformalBand
    trained_until: pd.Timestamp


def train(history: pd.DataFrame, horizons=HORIZONS) -> TrainedForecaster:
    band = ConformalBand(level=LEVEL).fit(lambda: EnsembleForecaster(tuple(horizons)), history)
    model = EnsembleForecaster(tuple(horizons)).fit(history)
    return TrainedForecaster(model, band, history.index[-1])


def next_days(trained: TrainedForecaster, history: pd.DataFrame) -> pd.DataFrame:
    """Forecast for the days after the last day of `history`.

    One row per day: expected max/min temperature and the 80% band around each.
    """
    window = history.iloc[-60:]
    last = window.index[-1]
    rows = []
    for h in trained.model.horizons:
        pred = trained.model.predict(window, h).iloc[-1]
        row = {"date": last + pd.Timedelta(days=h), "horizon": h}
        for t in TARGETS:
            width = trained.band.half_width[(t, h)]
            row[t] = round(float(pred[t]), 1)
            row[f"{t}_low"] = round(float(pred[t]) - width, 1)
            row[f"{t}_high"] = round(float(pred[t]) + width, 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")
