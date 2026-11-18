"""Feature engineering for day-ahead forecasting.

Every feature for day *t* only uses information available at the end of day *t*,
so a model trained on these rows never sees the future it has to predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGETS = ("tmax", "tmin")
HORIZONS = tuple(range(1, 8))

LAGS = (0, 1, 2, 3, 6, 13)
WINDOWS = (3, 7, 30)


def _seasonal(index: pd.DatetimeIndex) -> pd.DataFrame:
    doy = index.day_of_year.to_numpy()
    angle = 2 * np.pi * doy / 365.25
    return pd.DataFrame(
        {
            "season_sin": np.sin(angle),
            "season_cos": np.cos(angle),
            "season_sin2": np.sin(2 * angle),
            "season_cos2": np.cos(2 * angle),
        },
        index=index,
    )


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature table: one row per day, describing that day and the days before it."""
    feats = {}
    for col in ("tmax", "tmin"):
        for lag in LAGS:
            feats[f"{col}_lag{lag}"] = df[col].shift(lag)
        for w in WINDOWS:
            feats[f"{col}_mean{w}"] = df[col].rolling(w, min_periods=w).mean()
        feats[f"{col}_std7"] = df[col].rolling(7, min_periods=7).std()
        feats[f"{col}_trend3"] = df[col] - df[col].shift(3)
    feats["range_lag0"] = df["tmax"] - df["tmin"]
    for col in ("rain", "wind", "radiation", "sunshine"):
        feats[f"{col}_lag0"] = df[col]
        feats[f"{col}_mean3"] = df[col].rolling(3, min_periods=3).mean()
    feats["rain_sum7"] = df["rain"].rolling(7, min_periods=7).sum()
    # Wind direction is an angle: 359° and 1° are neighbours, so encode it on the circle.
    rad = np.deg2rad(df["wind_dir"])
    feats["wind_dir_sin"] = np.sin(rad)
    feats["wind_dir_cos"] = np.cos(rad)
    out = pd.DataFrame(feats, index=df.index)
    out = out.join(_seasonal(df.index))
    out["year"] = df.index.year + (df.index.day_of_year - 1) / 365.25
    return out


def make_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Targets for day t: the temperature `horizon` days later."""
    return pd.DataFrame({t: df[t].shift(-horizon) for t in TARGETS}, index=df.index)


def training_table(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aligned features X and targets y, with incomplete rows removed."""
    x = make_features(df)
    y = make_targets(df, horizon)
    ok = x.notna().all(axis=1) & y.notna().all(axis=1)
    return x[ok], y[ok]


def feature_names() -> list[str]:
    sample = pd.DataFrame(
        0.0,
        index=pd.date_range("2000-01-01", periods=40, freq="D"),
        columns=["tmax", "tmin", "rain", "wind", "wind_dir", "radiation", "sunshine"],
    )
    return list(make_features(sample).columns)
