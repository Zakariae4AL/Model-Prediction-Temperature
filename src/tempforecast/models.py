"""Forecasting models: two baselines and two machine-learning models.

All models share the same interface: ``fit(history)`` on a daily DataFrame and
``predict(history, horizon)`` returning predictions for day t + horizon for every
day t of ``history``. Baselines are what any real model has to beat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import HORIZONS, TARGETS, make_features, make_targets


class Climatology:
    """The usual temperature for each day of the year, learned from the training years.

    Values are smoothed with a circular 31-day window so that a single unusual year
    does not create spikes.
    """

    def __init__(self, window: int = 31):
        self.window = window
        self.table: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> Climatology:
        doy = np.minimum(df.index.day_of_year, 365)
        means = df[list(TARGETS)].groupby(doy).mean().reindex(range(1, 366)).interpolate()
        padded = pd.concat([means.iloc[-self.window :], means, means.iloc[: self.window]])
        smooth = padded.rolling(self.window, center=True, min_periods=1).mean()
        self.table = smooth.iloc[self.window : -self.window]
        self.table.index = range(1, 366)
        return self

    def for_dates(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        assert self.table is not None, "fit() first"
        doy = np.minimum(dates.day_of_year, 365)
        return pd.DataFrame(self.table.loc[doy].to_numpy(), index=dates, columns=list(TARGETS))


class Persistence:
    """Baseline: the temperature in h days will be the same as today."""

    name = "Persistence"

    def fit(self, history: pd.DataFrame) -> Persistence:
        return self

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        return history[list(TARGETS)].copy()


class ClimatologyBaseline:
    """Baseline: the usual temperature for that date of the year."""

    name = "Climatology"

    def fit(self, history: pd.DataFrame) -> ClimatologyBaseline:
        self.clim = Climatology().fit(history)
        return self

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        target_dates = history.index + pd.Timedelta(days=horizon)
        out = self.clim.for_dates(target_dates)
        out.index = history.index
        return out


class WeekdayRegression:
    """The original v1 model: a linear regression on the day of the week (Monday=1 ... Sunday=7).

    Kept only to show how much the new approach improves on it.
    """

    name = "v1 · weekday regression"

    def fit(self, history: pd.DataFrame) -> WeekdayRegression:
        x = (history.index.day_of_week + 1).to_numpy().reshape(-1, 1)
        self.models = {t: LinearRegression().fit(x, history[t]) for t in TARGETS}
        return self

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        x = ((history.index + pd.Timedelta(days=horizon)).day_of_week + 1).to_numpy().reshape(-1, 1)
        return pd.DataFrame({t: self.models[t].predict(x) for t in TARGETS}, index=history.index)


@dataclass
class MLForecaster:
    """One regression model per target and horizon, on the engineered features.

    The model also gets the climatology of the target date as an input, so it mostly has
    to learn how far the coming days will drift from the seasonal normal.
    """

    kind: str = "boosting"  # "boosting" or "ridge"
    horizons: tuple[int, ...] = HORIZONS
    random_state: int = 42
    models: dict = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        return "Gradient boosting" if self.kind == "boosting" else "Ridge regression"

    def _estimator(self):
        if self.kind == "ridge":
            return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        return HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=self.random_state,
        )

    def _inputs(self, features: pd.DataFrame, horizon: int) -> pd.DataFrame:
        clim = self.clim.for_dates(features.index + pd.Timedelta(days=horizon))
        x = features.copy()
        for t in TARGETS:
            x[f"clim_{t}"] = clim[t].to_numpy()
        return x

    def fit(self, history: pd.DataFrame) -> MLForecaster:
        self.clim = Climatology().fit(history)
        features = make_features(history)
        self.models = {}
        for h in self.horizons:
            y = make_targets(history, h)
            ok = features.notna().all(axis=1) & y.notna().all(axis=1)
            x = self._inputs(features[ok], h)
            for t in TARGETS:
                self.models[(t, h)] = self._estimator().fit(x, y.loc[ok, t])
        return self

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        features = make_features(history)
        ok = features.notna().all(axis=1)
        x = self._inputs(features[ok], horizon)
        out = pd.DataFrame(index=history.index, columns=list(TARGETS), dtype=float)
        for t in TARGETS:
            out.loc[ok, t] = self.models[(t, horizon)].predict(x)
        return out


@dataclass
class EnsembleForecaster:
    """Average of ridge regression and gradient boosting.

    The linear model is steadier (it wins on minimum temperatures), the boosted trees catch
    non-linear effects (they win on maximum temperatures); their average beats both overall.
    """

    horizons: tuple[int, ...] = HORIZONS
    name: str = "Ensemble (ridge + boosting)"

    def fit(self, history: pd.DataFrame) -> EnsembleForecaster:
        self.members = [
            MLForecaster("ridge", self.horizons).fit(history),
            MLForecaster("boosting", self.horizons).fit(history),
        ]
        return self

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        preds = [m.predict(history, horizon) for m in self.members]
        return sum(preds) / len(preds)


@dataclass
class ConformalBand:
    """Uncertainty band from split conformal prediction.

    A copy of the model is trained on all but the last `calibration_years` of history and
    its errors on those years are measured. The band is the forecast plus or minus the
    `level` quantile of those absolute errors, per target and horizon: if the future
    behaves like the recent past, the real value falls inside about `level` of the time.
    """

    level: float = 0.8
    calibration_years: int = 3
    half_width: dict = field(default_factory=dict)

    def fit(self, make_model, history: pd.DataFrame) -> ConformalBand:
        cut = history.index[-1] - pd.DateOffset(years=self.calibration_years)
        model = make_model().fit(history[history.index <= cut])
        context = history[history.index > cut - pd.Timedelta(days=60)]
        for h in model.horizons:
            pred = model.predict(context, h)
            truth = make_targets(context, h)
            scored = (context.index + pd.Timedelta(days=h)) > cut
            for t in TARGETS:
                err = (pred.loc[scored, t] - truth.loc[scored, t]).abs().dropna()
                self.half_width[(t, h)] = float(np.quantile(err, self.level))
        return self


def all_models() -> list:
    return [
        WeekdayRegression(),
        Persistence(),
        ClimatologyBaseline(),
        MLForecaster("ridge"),
        MLForecaster("boosting"),
        EnsembleForecaster(),
    ]
