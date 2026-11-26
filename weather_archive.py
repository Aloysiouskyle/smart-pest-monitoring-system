"""Load a dated 72-hour regional weather window for prototype evaluation.

Open-Meteo historical values are modelled/reanalysis estimates for a grid cell.
They are not readings from a sensor installed in a farmer's maize field.
"""

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from microclimate_sim import TelemetryRecord


EAT = ZoneInfo("Africa/Kampala")
ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
VARIABLES = (
    "temperature_2m", "relative_humidity_2m", "rain",
    "soil_moisture_0_to_7cm",
)


class WeatherDataError(ValueError):
    """A weather request or response cannot supply a complete window."""


@dataclass(frozen=True)
class HistoricalWindow:
    records: list[TelemetryRecord]
    requested_latitude: float
    requested_longitude: float
    grid_latitude: float
    grid_longitude: float
    first_date: date
    last_date: date
    source: str = "Open-Meteo Historical Weather API (regional reanalysis)"


@dataclass(frozen=True)
class ForecastWindow:
    records: list[TelemetryRecord]
    requested_latitude: float
    requested_longitude: float
    grid_latitude: float
    grid_longitude: float
    retrieved_at: datetime
    source: str = "Open-Meteo Weather Forecast API (regional weather-model outlook)"


def _measurement(value: object, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WeatherDataError(f"Missing or nonnumeric {name} in weather response")
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise WeatherDataError(f"Invalid {name} in weather response")
    return number


def _fetch_payload(url: str, opener: Callable) -> dict:
    try:
        with opener(url, timeout=15) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        raise WeatherDataError(f"Weather could not be loaded: {error}") from error
    if not isinstance(payload, dict) or payload.get("error"):
        raise WeatherDataError("Weather service returned an error")
    return payload


def _parse_records(payload: dict, expected_first: datetime | None = None):
    """Check complete hourly weather and preserve the service's grid coordinates."""
    if payload.get("timezone") != "Africa/Kampala":
        raise WeatherDataError("Weather response did not use Uganda local time")
    hourly = payload.get("hourly")
    units = payload.get("hourly_units")
    expected_units = {
        "temperature_2m": "°C", "relative_humidity_2m": "%",
        "rain": "mm", "soil_moisture_0_to_7cm": "m³/m³",
    }
    if not isinstance(hourly, dict) or not isinstance(units, dict):
        raise WeatherDataError("Weather response has no hourly measurements or units")
    if any(units.get(key) != value for key, value in expected_units.items()):
        raise WeatherDataError("Weather response has unexpected measurement units")
    timestamps = hourly.get("time")
    if not isinstance(timestamps, list) or len(timestamps) != 72:
        raise WeatherDataError("Weather response must contain exactly 72 hourly timestamps")
    if any(not isinstance(hourly.get(key), list) or len(hourly[key]) != 72 for key in VARIABLES):
        raise WeatherDataError("Weather response has incomplete measurements")

    records: list[TelemetryRecord] = []
    for index, stamp in enumerate(timestamps):
        try:
            local = datetime.fromisoformat(stamp)
        except (TypeError, ValueError) as error:
            raise WeatherDataError("Weather response has an invalid timestamp") from error
        if local.tzinfo is not None or local.minute or local.second or local.microsecond:
            raise WeatherDataError("Weather response has an invalid local hour")
        if index == 0 and expected_first is None:
            expected_first = local.replace(tzinfo=EAT)
        expected = expected_first + timedelta(hours=index)
        if local != expected.replace(tzinfo=None):
            raise WeatherDataError("Weather response has missing, duplicated or unordered hours")
        records.append(TelemetryRecord(
            timestamp=expected.isoformat(),
            temperature_c=_measurement(hourly["temperature_2m"][index], "temperature", -50, 60),
            relative_humidity_pct=_measurement(hourly["relative_humidity_2m"][index], "humidity", 0, 100),
            rainfall_mm_per_hour=_measurement(hourly["rain"][index], "rain", 0, 300),
            soil_moisture_pct=round(
                100 * _measurement(hourly["soil_moisture_0_to_7cm"][index], "soil moisture", 0, 1), 2
            ),
        ))
    return (records,
            _measurement(payload.get("latitude"), "grid latitude", -90, 90),
            _measurement(payload.get("longitude"), "grid longitude", -180, 180))


def load_historical_window(
    latitude: float,
    longitude: float,
    last_date: date,
    *,
    opener: Callable = urlopen,
    today: date | None = None,
) -> HistoricalWindow:
    """Fetch three complete calendar days ending on last_date in Uganda time.

    The date ends at least six days ago to allow delayed reanalysis products.
    Inject opener and today in tests; the default request has a 15 s timeout.
    """
    lat = _measurement(latitude, "latitude", -90, 90)
    lon = _measurement(longitude, "longitude", -180, 180)
    if isinstance(last_date, datetime) or not isinstance(last_date, date):
        raise WeatherDataError("Select a calendar date")
    today = today or datetime.now(EAT).date()
    if not date(2017, 1, 3) <= last_date <= today - timedelta(days=6):
        raise WeatherDataError("Select a date from 2017-01-03 through six days ago")
    first_date = last_date - timedelta(days=2)
    query = urlencode({
        "latitude": lat, "longitude": lon,
        "start_date": first_date.isoformat(), "end_date": last_date.isoformat(),
        "hourly": ",".join(VARIABLES), "timezone": "Africa/Kampala",
        "temperature_unit": "celsius", "precipitation_unit": "mm",
    })
    payload = _fetch_payload(f"{ENDPOINT}?{query}", opener)
    records, grid_lat, grid_lon = _parse_records(
        payload, datetime.combine(first_date, datetime.min.time(), EAT)
    )
    return HistoricalWindow(
        records, lat, lon,
        grid_lat, grid_lon,
        first_date, last_date,
    )


def load_weather_outlook(latitude: float, longitude: float, *, opener: Callable = urlopen) -> ForecastWindow:
    """Fetch the next 72 hourly regional weather-model estimates.

    This forecasts weather, not future pest counts or an outbreak.
    """
    lat = _measurement(latitude, "latitude", -90, 90)
    lon = _measurement(longitude, "longitude", -180, 180)
    query = urlencode({
        "latitude": lat, "longitude": lon, "hourly": ",".join(VARIABLES),
        "timezone": "Africa/Kampala", "forecast_hours": 72,
        "temperature_unit": "celsius", "precipitation_unit": "mm",
    })
    payload = _fetch_payload(f"{FORECAST_ENDPOINT}?{query}", opener)
    records, grid_lat, grid_lon = _parse_records(payload)
    return ForecastWindow(records, lat, lon, grid_lat, grid_lon, datetime.now(EAT))
