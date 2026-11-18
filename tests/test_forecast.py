import pandas as pd

from tempforecast import cli, data, forecast


def test_forecasts_the_next_seven_days(weather):
    trained = forecast.train(weather)
    table = forecast.next_days(trained, weather)
    assert len(table) == 7
    assert table.index[0] == weather.index[-1] + pd.Timedelta(days=1)
    assert (table.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    for t in ("tmax", "tmin"):
        assert (table[f"{t}_low"] <= table[t]).all() and (table[t] <= table[f"{t}_high"]).all()
    assert (table["tmax"] > table["tmin"]).all()


def test_cli_prints_a_forecast(weather, monkeypatch, capsys):
    monkeypatch.setattr(data, "load", lambda *a, **k: weather)
    assert cli.main(["forecast"]) == 0
    out = capsys.readouterr().out
    assert "forecast from the last observed day" in out and out.count("°C") >= 14


def test_quality_gate_on_real_data():
    """On the real Marrakech data, the model must clearly beat 'same as today' for tomorrow."""
    from tempforecast import evaluate
    from tempforecast.models import EnsembleForecaster, Persistence

    df = data.load(recent=None)
    scores = evaluate.evaluate(df, "2025-01-01", horizons=(1,), models=[Persistence(), EnsembleForecaster((1,))])
    mae = scores.pivot(index="model", columns="target", values="mae")
    assert (mae.loc["Ensemble (ridge + boosting)"] < 0.95 * mae.loc["Persistence"]).all()
