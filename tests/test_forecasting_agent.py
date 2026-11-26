"""WP 2.2 rule boundaries and compatibility with WP 2.1 and the dashboard."""

import unittest
from collections import defaultdict

from forecasting_agent import RiskRules, assess_risk
from microclimate_sim import generate_telemetry


def window(matching_hours=0):
    rows = generate_telemetry("normal")
    for i, row in enumerate(rows):
        row.update(temperature_c=25, relative_humidity_pct=72 if i < matching_hours else 71.99)
    return rows


class ForecastingTests(unittest.TestCase):
    def test_fusion_at_inclusive_cutoffs(self):
        for hours, score, expected in (
            (17, 0.49999, "Low"), (18, 0.49999, "Moderate"),
            (17, 0.50, "Moderate"), (18, 0.50, "High"),
            (0, 0, "Low"), (72, 1, "High"),
        ):
            with self.subTest(hours=hours, score=score):
                result = assess_risk(window(hours), {"fall-armyworm-egg": [score]})
                self.assertEqual(result.risk_level, expected)
                self.assertEqual(result.matching_hours, hours)
                self.assertEqual(result.strongest_faw_confidence, score)
                self.assertEqual(len(result.contributing_factors), 2)
                self.assertIn("not a scientifically validated 48–72-hour pest forecast", result.limitation)

    def test_missing_image_and_processed_empty(self):
        for detections, status in ((None, "no_image"), ({}, "zero_detections")):
            result = assess_risk(window(72), detections)
            self.assertEqual(result.risk_level, "Moderate")
            self.assertEqual(result.image_status, status)
            self.assertIsNone(result.strongest_faw_confidence)

    def test_other_classes_and_repeated_boxes(self):
        detections = defaultdict(list, {"healthy-maize": [1], "maize-streak-disease": [0.99]})
        result = assess_risk(window(), detections)
        self.assertEqual((result.risk_level, result.image_status), ("Low", "detections_present"))
        for label in ("fall-armyworm-egg", "fall-armyworm-frass", "fall-armyworm-larva",
                      "fall-armyworm-larval-damage"):
            with self.subTest(label=label):
                detections[label] = [0.4, 0.75, 0.1]
                result = assess_risk(window(), detections)
                self.assertEqual(result.risk_level, "Moderate")
                self.assertEqual(result.strongest_faw_confidence, 0.75)
                self.assertIn(label, result.contributing_factors[1])
                del detections[label]

    def test_real_simulator_and_dashboard_format(self):
        boxes = defaultdict(list, {"fall-armyworm-larval-damage": [0.92945],
                                  "maize-streak-disease": [0.10267]})
        for scenario, expected in (("normal", "Moderate"), ("higher_risk", "High")):
            with self.subTest(scenario=scenario):
                result = assess_risk(generate_telemetry(scenario, seed=42), boxes)
                self.assertEqual(result.telemetry_hours, 72)
                self.assertEqual(result.risk_level, expected)

    def test_custom_rules(self):
        rows = window(17)
        detections = {"fall-armyworm-larva": [0.4]}
        self.assertEqual(assess_risk(rows, detections).risk_level, "Low")
        rules = RiskRules(min_temperature_c=25, min_humidity_pct=72,
                          min_matching_hours=17, min_detection_confidence=0.4)
        self.assertEqual(assess_risk(rows, detections, rules=rules).risk_level, "High")
        for key, value in (("min_matching_hours", True), ("min_matching_hours", 73),
                           ("min_detection_confidence", -0.1), ("min_temperature_c", 61),
                           ("min_humidity_pct", float("nan"))):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                RiskRules(**{key: value})

    def test_incomplete_and_invalid_windows(self):
        rows = window()
        for invalid in (None, [], rows[:-1], rows + rows[:1], [None] + rows[1:]):
            with self.assertRaises(ValueError):
                assess_risk(invalid)
        for stamp in (None, "invalid", "2026-11-18T00:00:00", "2026-11-18T00:30:00+03:00",
                      rows[0]["timestamp"], rows[2]["timestamp"]):
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                assess_risk([rows[0], dict(rows[1], timestamp=stamp), *rows[2:]])

    def test_invalid_measurements(self):
        for key, value in (("temperature_c", 61), ("temperature_c", "25"),
                           ("relative_humidity_pct", float("nan")), ("rainfall_mm_per_hour", -1),
                           ("soil_moisture_pct", float("inf")), ("soil_moisture_pct", True)):
            rows = window()
            rows[25][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                assess_risk(rows)
        rows = window()
        del rows[-1]["rainfall_mm_per_hour"]
        with self.assertRaises(ValueError):
            assess_risk(rows)

    def test_invalid_detections_including_irrelevant_classes(self):
        for invalid in ([], {"armyworm": [1]}, {"fall-armyworm-egg": []},
                        {"healthy-maize": 0.5}, {"fall-armyworm-frass": [-0.001]},
                        {"maize-streak-disease": [1.001]}, {"healthy-maize": [True]},
                        {"fall-armyworm-larva": [0.9, float("nan")]}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                assess_risk(window(), invalid)
