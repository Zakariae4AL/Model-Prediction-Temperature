import numpy as np
import pandas as pd
import pytest


def synthetic_weather(years: int = 4, seed: int = 0) -> pd.DataFrame:
    """A small, realistic-looking daily series: seasonal cycle + persistent weather anomalies."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2015-01-01", periods=365 * years, freq="D")
    season = np.sin(2 * np.pi * (index.day_of_year - 110) / 365.25)
    anomaly = np.zeros(len(index))
    for i in range(1, len(index)):  # weather systems last a few days
        anomaly[i] = 0.75 * anomaly[i - 1] + rng.normal(0, 1.5)
    tmax = 27 + 9 * season + anomaly
    tmin = 12 + 7 * season + 0.7 * anomaly + rng.normal(0, 0.8, len(index))
    return pd.DataFrame(
        {
            "tmax": tmax,
            "tmin": tmin,
            "rain": rng.exponential(0.7, len(index)) * (rng.random(len(index)) < 0.15),
            "wind": 12 + rng.normal(0, 3, len(index)),
            "wind_dir": rng.uniform(0, 360, len(index)),
            "radiation": 20 + 6 * season + rng.normal(0, 2, len(index)),
            "sunshine": 10 + 2 * season + rng.normal(0, 1, len(index)),
            "provisional": False,
        },
        index=pd.DatetimeIndex(index, name="date"),
    )


@pytest.fixture
def weather() -> pd.DataFrame:
    return synthetic_weather()


# ruff: noqa: E501
OPEN_METEO_CSV = """latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation
31.599297,-7.976074,465.0,0,Africa/Casablanca,GMT

time,temperature_2m_max (°C),temperature_2m_min (°C),precipitation_sum (mm),wind_speed_10m_max (km/h),wind_direction_10m_dominant (°),shortwave_radiation_sum (MJ/m²),sunshine_duration (s)
2026-01-01,18.1,0.9,0.00,8.4,232,13.41,36000.00
2026-01-02,16.7,3.2,0.00,11.3,265,13.40,34609.68
2026-01-03,15.0,4.0,1.20,9.0,10,10.00,20000.00
"""
