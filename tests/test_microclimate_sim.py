

import csv
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import microclimate_sim as sim


BOUNDS = {
    "normal": ((19, 29), (55, 85), (0, 4), (34, 42)),
    "higher_risk": ((22, 32), (65, 95), (0, 6), (42, 52)),
}
MEASUREMENTS = (
    "temperature_c", "relative_humidity_pct", "rainfall_mm_per_hour", "soil_moisture_pct",
)


class MicroclimateTests(unittest.TestCase):
    def test_hourly_records_across_month_boundary(self):
        start = datetime(2026, 11, 30, 22, tzinfo=timezone.utc)
        for scenario in BOUNDS:
            with self.subTest(scenario=scenario):
                records = sim.generate_telemetry(scenario, start=start)
                self.assertEqual(len(records), 72)
                for hour, record in enumerate(records):
                    timestamp = datetime.fromisoformat(record["timestamp"])
                    self.assertEqual(timestamp, start + timedelta(hours=hour))
                    self.assertEqual(timestamp.utcoffset(), timedelta(hours=3))

    def test_schema_and_ranges(self):
        for scenario, bounds in BOUNDS.items():
            for seed in (0, 42, 123):
                with self.subTest(scenario=scenario, seed=seed):
                    for record in sim.generate_telemetry(scenario, seed=seed):
                        self.assertEqual(set(record), {"timestamp", *MEASUREMENTS})
                        for field, (low, high) in zip(MEASUREMENTS, bounds):
                            value = record[field]
                            self.assertIsInstance(value, float)
                            self.assertLessEqual(low, value)
                            self.assertLessEqual(value, high)

    def test_seed_and_scenario(self):
        for scenario in BOUNDS:
            first = sim.generate_telemetry(scenario, seed=7)
            self.assertEqual(first, sim.generate_telemetry(scenario, seed=7))
            self.assertNotEqual(first, sim.generate_telemetry(scenario, seed=8))
        self.assertNotEqual(sim.generate_telemetry("normal"), sim.generate_telemetry("higher_risk"))

    def test_invalid_inputs(self):
        for seed in (True, 1.5, "42"):
            with self.assertRaises(TypeError):
                sim.generate_telemetry("normal", seed=seed)
        for start in (datetime(2026, 1, 1), datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)):
            with self.assertRaises(ValueError):
                sim.generate_telemetry("normal", start=start)
        for scenario, fmt in (("../escape", "csv"), ("normal", "../escape")):
            with self.assertRaises(ValueError):
                sim.export_scenario(scenario, fmt)

    def test_exports_round_trip_and_replace(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            with patch.object(sim, "_PROJECT_ROOT", root):
                for scenario in BOUNDS:
                    expected = sim.generate_telemetry(scenario, seed=11)
                    for fmt in ("csv", "json"):
                        with self.subTest(scenario=scenario, fmt=fmt):
                            sim.export_scenario(scenario, fmt, seed=9)
                            path = sim.export_scenario(scenario, fmt, seed=11)
                            self.assertEqual(path, root / "data" / "generated" / f"{scenario}.{fmt}")
                            with path.open(encoding="utf-8", newline="") as source:
                                if fmt == "json":
                                    actual = json.load(source)
                                else:
                                    actual = list(csv.DictReader(source))
                                    for row in actual:
                                        for field in MEASUREMENTS:
                                            row[field] = float(row[field])
                            self.assertEqual(actual, expected)
