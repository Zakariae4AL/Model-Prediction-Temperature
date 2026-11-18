"""Loading and downloading daily weather data from Open-Meteo."""

from __future__ import annotations

import io
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# The archive is complete about five days late; the forecast service fills that gap
# with its own analysis of the last days ("provisional" data).
RECENT_URL = "https://api.open-meteo.com/v1/forecast"

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "sunshine_duration",
]

# Short, unit-free column names used everywhere in the code.
COLUMNS = {
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "precipitation_sum": "rain",
    "wind_speed_10m_max": "wind",
    "wind_direction_10m_dominant": "wind_dir",
    "shortwave_radiation_sum": "radiation",
    "sunshine_duration": "sunshine",
}


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float
    timezone: str


MARRAKECH = Location("Marrakech", 31.63, -7.98, "Africa/Casablanca")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_DATA_PATH = DATA_DIR / "marrakech_daily.csv"
RECENT_DATA_PATH = DATA_DIR / "marrakech_recent.csv"


def read_open_meteo_csv(source: str | Path | io.TextIOBase) -> pd.DataFrame:
    """Read an Open-Meteo daily CSV export into a clean, date-indexed DataFrame.

    Open-Meteo files start with a small location block (two lines and a blank line)
    before the daily table; column names carry units such as "temperature_2m_max (°C)".
    """
    raw = pd.read_csv(source, skiprows=3)
    raw.columns = [c.split(" (")[0].strip() for c in raw.columns]
    missing = [v for v in DAILY_VARIABLES if v not in raw.columns]
    if "time" not in raw.columns or missing:
        raise ValueError(f"Not an Open-Meteo daily export: missing {['time', *missing]}")

    df = raw.rename(columns=COLUMNS).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[list(COLUMNS.values())].astype(float)
    df["sunshine"] = df["sunshine"] / 3600.0  # seconds -> hours
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate days, fill short gaps and reject impossible values."""
    df = df[~df.index.duplicated(keep="last")]
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full)
    df.index.name = "date"
    # Gaps of a few days are interpolated; longer gaps stay missing and are dropped later.
    df = df.interpolate(limit=3, limit_area="inside")
    swapped = df["tmin"] > df["tmax"]
    df.loc[swapped, ["tmin", "tmax"]] = df.loc[swapped, ["tmax", "tmin"]].to_numpy()
    out_of_range = (df["tmax"] > 60) | (df["tmin"] < -30)
    df.loc[out_of_range, ["tmax", "tmin"]] = float("nan")
    return df


def load(path: str | Path = DEFAULT_DATA_PATH, recent: str | Path | None = RECENT_DATA_PATH) -> pd.DataFrame:
    """Load the archive CSV, extended with the provisional recent days if they were downloaded.

    The result has a boolean ``provisional`` column: True for days that only come from
    the recent file. Archive values always win where both exist.
    """
    df = read_open_meteo_csv(Path(path))
    df["provisional"] = False
    if recent is not None and Path(recent).exists():
        newer = read_open_meteo_csv(Path(recent))
        newer = newer[newer.index > df.index.max()]
        if not newer.empty:
            newer["provisional"] = True
            df = clean_merge(df, newer)
    return df


def clean_merge(archive: pd.DataFrame, newer: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([archive, newer])
    flags = merged.pop("provisional")
    merged = clean(merged)
    merged["provisional"] = flags.reindex(merged.index).fillna(True).astype(bool)
    return merged


def archive_url(location: Location, start: date, end: date) -> str:
    query = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": location.timezone,
        "format": "csv",
    }
    return f"{ARCHIVE_URL}?{urllib.parse.urlencode(query)}"


def _download(url: str, timeout: float) -> str:
    if not url.startswith("https://"):
        raise ValueError(f"Refusing to download from a non-HTTPS URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "tempforecast"})  # noqa: S310 - https checked above
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read().decode("utf-8")


def recent_url(location: Location, past_days: int = 14) -> str:
    query = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "past_days": past_days,
        "forecast_days": 1,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": location.timezone,
        "format": "csv",
    }
    return f"{RECENT_URL}?{urllib.parse.urlencode(query)}"


def fetch_recent(
    location: Location = MARRAKECH, path: str | Path = RECENT_DATA_PATH, timeout: float = 60
) -> pd.DataFrame:
    """Download the last two weeks up to yesterday from the forecast service."""
    text = _download(recent_url(location), timeout)
    df = read_open_meteo_csv(io.StringIO(text))
    df = df[df.index < pd.Timestamp(date.today())]  # today is not over yet
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return df


def fetch(
    location: Location = MARRAKECH,
    start: date = date(2000, 1, 1),
    end: date | None = None,
    path: str | Path = DEFAULT_DATA_PATH,
    timeout: float = 60,
) -> pd.DataFrame:
    """Download the daily history from the Open-Meteo archive and save it as CSV.

    The archive (ERA5 reanalysis) is complete with a delay of about five days,
    so by default the download stops six days ago.
    """
    end = end or date.today() - timedelta(days=6)
    text = _download(archive_url(location, start, end), timeout)
    df = read_open_meteo_csv(io.StringIO(text))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return df
