"""Check source dates, units and completeness before field-facing display."""

import json
import unittest
from datetime import date, datetime, timedelta
from io import BytesIO
from unittest.mock import Mock

from forecasting_agent import assess_risk
from weather_archive import WeatherDataError, load_historical_window, load_weather_outlook


def weather_payload():
    start = datetime(2026, 11, 17)
    return {
        "latitude": 0.31634447, "longitude": 32.765625,
        "timezone": "Africa/Kampala",
        "hourly_units": {
            "time": "iso8601", "temperature_2m": "°C", "relative_humidity_2m": "%",
            "rain": "mm", "soil_moisture_0_to_7cm": "m³/m³",
        },
        "hourly": {
            "time": [(start + timedelta(hours=i)).isoformat(timespec="minutes") for i in range(72)],
            "temperature_2m": [25.0] * 72,
            "relative_humidity_2m": [72] * 72,
            "rain": [0.1] * 72,
            "soil_moisture_0_to_7cm": [0.31] * 72,
        },
    }


def opener(payload):
    return Mock(return_value=BytesIO(json.dumps(payload).encode("utf-8")))


class HistoricalWindowTests(unittest.TestCase):
    def load(self, payload):
        return load_historical_window(0.35, 32.75, date(2026, 11, 19),
                                      opener=opener(payload), today=date(2026, 11, 26))

    def test_complete_window_flows_into_existing_agent(self):
        payload = weather_payload()
        open_mock = opener(payload)
        window = load_historical_window(0.35, 32.75, date(2026, 11, 19),
                                        opener=open_mock, today=date(2026, 11, 26))
        self.assertEqual(len(window.records), 72)
        self.assertEqual(window.records[0]["timestamp"], "2026-11-17T00:00:00+03:00")
        self.assertEqual(window.records[-1]["timestamp"], "2026-11-19T23:00:00+03:00")
        self.assertEqual(window.records[0]["soil_moisture_pct"], 31.0)
        self.assertEqual(window.grid_latitude, 0.31634447)
        self.assertEqual(assess_risk(window.records).matching_hours, 72)
        self.assertIn("start_date=2026-11-17", open_mock.call_args.args[0])

    def test_rejects_gaps_bad_units_and_missing_measurements(self):
        for change in ("gap", "unit", "missing", "null"):
            payload = weather_payload()
            if change == "gap":
                payload["hourly"]["time"][20] = payload["hourly"]["time"][19]
            elif change == "unit":
                payload["hourly_units"]["soil_moisture_0_to_7cm"] = "%"
            elif change == "missing":
                payload["hourly"]["rain"].pop()
            else:
                payload["hourly"]["temperature_2m"][5] = None
            with self.subTest(change=change), self.assertRaises(WeatherDataError):
                self.load(payload)

    def test_rejects_recent_dates_and_invalid_coordinates(self):
        with self.assertRaises(WeatherDataError):
            load_historical_window(0.35, 32.75, date(2026, 11, 25), today=date(2026, 11, 26))
        with self.assertRaises(WeatherDataError):
            load_historical_window(99, 32.75, date(2026, 11, 19), today=date(2026, 11, 26))

    def test_future_weather_window_is_weather_not_pest_forecast(self):
        payload = weather_payload()
        window = load_weather_outlook(0.35, 32.75, opener=opener(payload))
        self.assertEqual(len(window.records), 72)
        self.assertEqual(window.records[0]["timestamp"], "2026-11-17T00:00:00+03:00")
        self.assertIn("Weather Forecast API", window.source)
        self.assertIn("not a scientifically validated", assess_risk(window.records).limitation)
