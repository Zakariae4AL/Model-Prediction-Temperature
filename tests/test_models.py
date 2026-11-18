import numpy as np
import pandas as pd
import pytest

from tempforecast import evaluate
from tempforecast.models import (
    Climatology,
    ConformalBand,
    EnsembleForecaster,
    MLForecaster,
    Persistence,
    WeekdayRegression,
)

from .conftest import synthetic_weather


def test_climatology_is_smooth_and_seasonal(weather):
    clim = Climatology().fit(weather)
    assert len(clim.table) == 365
    assert clim.table["tmax"].diff().abs().max() < 0.6  # no spikes from day to day
    summer = clim.for_dates(pd.DatetimeIndex(["2030-07-20"]))["tmax"].iloc[0]
    winter = clim.for_dates(pd.DatetimeIndex(["2030-01-20"]))["tmax"].iloc[0]
    assert summer > winter + 10


def test_persistence_repeats_today(weather):
    pred = Persistence().fit(weather).predict(weather, horizon=2)
    pd.testing.assert_series_equal(pred["tmax"], weather["tmax"])


def test_a_learned_model_beats_the_baselines():
    # Eight synthetic years: enough history to learn the seasons. The ensemble itself is
    # checked on the real Marrakech data in test_forecast.py (quality gate).
    scores = evaluate.evaluate(
        synthetic_weather(years=8),
        test_start="2022-01-01",
        horizons=(1,),
        models=[Persistence(), WeekdayRegression(), MLForecaster("ridge", (1,))],
    )
    mae = scores[scores["target"] == "tmax"].set_index("model")["mae"]
    assert mae["Ridge regression"] < mae["Persistence"] < mae["v1 · weekday regression"]


def test_ensemble_is_the_average_of_its_members(weather):
    model = EnsembleForecaster((1,)).fit(weather)
    ridge, boosting = (m.predict(weather, 1) for m in model.members)
    pd.testing.assert_frame_equal(model.predict(weather, 1), (ridge + boosting) / 2)


def test_conformal_band_widens_with_the_horizon(weather):
    band = ConformalBand(level=0.8, calibration_years=1).fit(lambda: MLForecaster("ridge", (1, 5)), weather)
    assert 0 < band.half_width[("tmax", 1)] < band.half_width[("tmax", 5)]


def test_evaluation_scores_only_future_days(weather):
    train, test = evaluate.split(weather, "2018-01-01")
    assert train.index.max() < test.index.min()
    scores = evaluate.evaluate(weather, "2018-01-01", horizons=(1, 2), models=[Persistence()])
    assert set(scores["horizon"]) == {1, 2}
    assert scores["n"].max() <= len(test)


def test_metrics():
    m = evaluate.metrics(pd.Series([10.0, 20.0, np.nan]), pd.Series([12.0, 19.0, 5.0]))
    assert m["mae"] == pytest.approx(1.5) and m["bias"] == pytest.approx(0.5) and m["n"] == 2
