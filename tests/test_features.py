import numpy as np
import pandas as pd

from tempforecast.features import feature_names, make_features, make_targets, training_table


def test_features_never_look_into_the_future(weather):
    """Changing every day after t must not change the features of day t."""
    t = weather.index[400]
    before = make_features(weather).loc[t]
    changed = weather.copy()
    changed.loc[changed.index > t, ["tmax", "tmin", "rain", "wind", "radiation"]] += 50
    after = make_features(changed).loc[t]
    pd.testing.assert_series_equal(before, after)


def test_targets_are_the_future_values(weather):
    y = make_targets(weather, horizon=3)
    assert y["tmax"].iloc[10] == weather["tmax"].iloc[13]
    assert y["tmin"].iloc[-3:].isna().all()


def test_training_table_is_aligned_and_complete(weather):
    x, y = training_table(weather, horizon=1)
    assert x.index.equals(y.index)
    assert not x.isna().any().any() and not y.isna().any().any()
    assert list(x.columns) == feature_names()


def test_wind_direction_is_encoded_on_a_circle(weather):
    w = weather.iloc[:40].copy()
    w["wind_dir"] = 359.0
    a = make_features(w).iloc[-1]
    w["wind_dir"] = 1.0
    b = make_features(w).iloc[-1]
    assert np.isclose(a["wind_dir_cos"], b["wind_dir_cos"]) and abs(a["wind_dir_sin"] - b["wind_dir_sin"]) < 0.05
