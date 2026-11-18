import io

import numpy as np
import pandas as pd
import pytest

from tempforecast import data

from .conftest import OPEN_METEO_CSV


def test_reads_an_open_meteo_export():
    df = data.read_open_meteo_csv(io.StringIO(OPEN_METEO_CSV))
    assert list(df.columns) == ["tmax", "tmin", "rain", "wind", "wind_dir", "radiation", "sunshine"]
    assert df.index[0] == pd.Timestamp("2026-01-01")
    assert df.loc["2026-01-01", "tmax"] == 18.1
    assert df.loc["2026-01-01", "sunshine"] == pytest.approx(10.0)  # seconds -> hours


def test_rejects_other_files():
    with pytest.raises(ValueError, match="Open-Meteo"):
        data.read_open_meteo_csv(io.StringIO("a\nb\nc\ndate,temp\n2026-01-01,3\n"))


def test_clean_fills_short_gaps_only_and_fixes_swapped_values():
    index = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-15"])
    df = pd.DataFrame({"tmax": [20.0, 5.0, 22.0, 25.0], "tmin": [10.0, 15.0, 12.0, 14.0]}, index=index)
    out = data.clean(df)
    assert len(out) == 15  # missing days are added back
    assert out.loc["2026-01-03", "tmax"] == pytest.approx(13.5)  # one-day gap interpolated
    assert np.isnan(out.loc["2026-01-10", "tmax"])  # a long gap is not invented
    assert out.loc["2026-01-02", "tmax"] == 15.0 and out.loc["2026-01-02", "tmin"] == 5.0


def test_load_adds_recent_days_as_provisional(tmp_path):
    archive = tmp_path / "archive.csv"
    archive.write_text(OPEN_METEO_CSV, encoding="utf-8")
    recent = tmp_path / "recent.csv"
    recent.write_text(
        OPEN_METEO_CSV.replace("2026-01-03,15.0", "2026-01-04,14.0").replace("2026-01-02,16.7", "2026-01-03,99.0"),
        encoding="utf-8",
    )
    df = data.load(archive, recent)
    assert df.index.max() == pd.Timestamp("2026-01-04")
    assert not df.loc["2026-01-03", "provisional"]
    assert df.loc["2026-01-03", "tmax"] == 15.0  # the archive wins where both exist
    assert df.loc["2026-01-04", "provisional"]


def test_urls_ask_for_every_variable_as_csv():
    url = data.archive_url(data.MARRAKECH, pd.Timestamp("2020-01-01").date(), pd.Timestamp("2020-12-31").date())
    assert url.startswith("https://archive-api.open-meteo.com/")
    for variable in data.DAILY_VARIABLES:
        assert variable in url
    assert "format=csv" in url and "past_days=14" in data.recent_url(data.MARRAKECH)


def test_refuses_insecure_downloads():
    with pytest.raises(ValueError, match="HTTPS"):
        data._download("http://example.com/data.csv", timeout=1)


def test_bundled_dataset_is_complete():
    df = data.load(recent=None)
    assert len(df) > 9000
    assert df[["tmax", "tmin"]].notna().all().all()
    assert (df["tmin"] <= df["tmax"]).all()
