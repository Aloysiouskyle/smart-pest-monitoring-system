"""Exercise the bilingual SMS previews using real forecasting results."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import advisory_agent as agent
from forecasting_agent import RiskRules, assess_risk
from microclimate_sim import generate_telemetry


class AdvisoryTests(unittest.TestCase):
    def test_all_forecast_and_photo_combinations(self):
        for scenario in ("normal", "higher_risk"):
            rows = generate_telemetry(scenario, seed=42)
            climate = scenario == "higher_risk"
            for boxes, en, lg, visual in (
                (None, "No photo submitted", "Tewali kifaananyi", False),
                ({}, "Photo check flagged nothing", "Tewali kyazuuliddwa", False),
                ({"healthy-maize": [1]}, "other maize features", "kiraze ebirala", False),
                ({"fall-armyworm-frass": [0.49]}, "too weak to count", "tekategeerekeka", False),
                ({"fall-armyworm-egg": [0.5]}, "possible armyworm signs", "kiyinza okulaga", True),
            ):
                with self.subTest(scenario=scenario, boxes=boxes):
                    assessment = assess_risk(rows, boxes)
                    result = agent.build_advisory(assessment)
                    level = ("Low", "Moderate", "High")[int(climate) + int(visual)]
                    luganda_level = {"Low": "butono", "Moderate": "bwa wakati", "High": "bungi"}
                    self.assertIs(result.assessment, assessment)
                    self.assertEqual(assessment.risk_level, level)
                    self.assertIn(level.lower(), result.english)
                    self.assertIn(luganda_level[level], result.luganda)
                    self.assertIn(en, result.english)
                    self.assertIn(lg, result.luganda)
                    self.assertTrue(result.luganda_requires_review)
                    for text in (result.english, result.luganda):
                        self.assertLessEqual(len(text), 160)
                        self.assertRegex(text, r"\A[A-Za-z0-9 .,:'!?-]+\Z")

    def test_custom_visual_cutoff_distinguishes_two_moderate_results(self):
        boxes = {"fall-armyworm-larva": [0.6]}
        for scenario, cutoff, wording in (
            ("higher_risk", 0.7, "Risk uses weather only"),
            ("normal", 0.6, "possible armyworm signs"),
        ):
            forecast = assess_risk(generate_telemetry(scenario), boxes,
                                   rules=RiskRules(min_detection_confidence=cutoff))
            self.assertEqual(forecast.risk_level, "Moderate")
            self.assertIn(wording, agent.build_advisory(forecast).english)

    def test_invalid_assessment(self):
        assessment = assess_risk(generate_telemetry("normal"))
        with self.assertRaises(TypeError):
            agent.build_advisory({"risk_level": "Low"})
        for changes in (
            {"risk_level": "Severe"}, {"image_status": "unknown"}, {"telemetry_hours": 70},
            {"matching_hours": -1}, {"risk_level": "High"}, {"visual_rule_met": 1},
            {"strongest_faw_confidence": 0.7}, {"visual_rule_met": True},
            {"image_status": "detections_present", "strongest_faw_confidence": float("nan")},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                agent.build_advisory(replace(assessment, **changes))

    def test_template_limits_at_render_time(self):
        forecast = assess_risk(generate_telemetry("normal"))
        original = agent.build_advisory(forecast).english
        action = agent._ACTIONS["en"]["Low"][1]
        exact = action + "a" * (160 - len(original))
        for replacement in (exact, exact + "a", "Check [maize].", "Farmer’s maize"):
            with self.subTest(action=replacement), patch.dict(
                agent._ACTIONS["en"], {"Low": (agent._ACTIONS["en"]["Low"][0], replacement)}
            ):
                if replacement == exact:
                    self.assertEqual(len(agent.build_advisory(forecast).english), 160)
                else:
                    with self.assertRaises(ValueError):
                        agent.build_advisory(forecast)
