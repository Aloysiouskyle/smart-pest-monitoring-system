"""Exercise the Streamlit entry point without requiring trained weights."""

from io import BytesIO
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from forecasting_agent import LIMITATION


class DashboardTests(unittest.TestCase):
    def setUp(self):
        folder = self.enterContext(TemporaryDirectory())
        script = Path(folder) / "app.py"
        copyfile(Path(__file__).resolve().parents[1] / "app.py", script)
        self.app = AppTest.from_file(str(script), default_timeout=60)
        self.weights = script.parent / "runs/detect/pest_detection_model-6/weights/best.pt"
        self.weights.parent.mkdir(parents=True)
        photo = BytesIO()
        Image.new("RGB", (32, 32)).save(photo, format="PNG")
        self.photo = photo.getvalue()

    def assert_assessment(self, level, evidence):
        self.assertFalse(self.app.exception)
        self.assertEqual(self.metric("Simulated Fall Armyworm risk"), level)
        self.assertIn(evidence, self.markdown_text())

    def metric(self, label):
        return next(m.value for m in self.app.metric if m.label == label)

    def markdown_text(self):
        return "\n".join(m.value for m in self.app.markdown)

    def research_view(self):
        return self.app.selectbox(key="view").select("Developer/Research View")

    def upload_photo(self):
        self.app.file_uploader[0].upload("maize.png", self.photo).run()

    def test_farmer_view_default_without_photo(self):
        self.app.run()
        self.assertEqual(self.app.selectbox(key="view").value, "Farmer View")
        self.assert_assessment("Low", "No photo submitted")
        self.assertIn("Keep checking under maize leaves", self.markdown_text())
        self.assertNotIn("Okugezesa", self.markdown_text())
        self.assertEqual([h.value for h in self.app.subheader].count("Recommended action"), 1)
        self.assertTrue(any("unvalidated" in c.value for c in self.app.caption))
        self.assertFalse(self.app.sidebar.slider)
        self.assertIn("Simulated weather readings", [e.label for e in self.app.expander])

    def test_farmer_language_selection_shows_only_draft_luganda(self):
        self.app.run()
        self.app.selectbox(key="language").select("Draft Luganda").run()
        self.assert_assessment("Low", "Tewali kifaananyi")
        self.assertIn("Keberanga wansi", self.markdown_text())
        self.assertNotIn("Keep checking under maize leaves", self.markdown_text())
        self.assertTrue(any("awaiting human review" in c.value for c in self.app.caption))

    def test_research_view_without_photo_and_scenario_selection(self):
        self.weights.touch()
        self.weights.with_suffix(".onnx").touch()
        self.app.run()
        self.research_view().run()
        self.assert_assessment("Low", "No photo submitted")
        self.assertIn("Okugezesa", self.markdown_text())
        self.assertIn("requires at least 18", self.markdown_text())
        self.assertIn("confidence ≥ 50%", self.markdown_text())
        self.assertIn(LIMITATION, [c.value for c in self.app.caption])
        self.assertEqual(sum("GSM-7" in c.value for c in self.app.caption), 2)
        self.assertEqual(self.app.selectbox(key="backend").options, ["PyTorch (.pt)", "ONNX (.onnx)"])
        self.app.selectbox(key="backend").select("ONNX (.onnx)")
        self.app.selectbox(key="scenario").select("Higher-risk scenario").run()
        self.assert_assessment("Moderate", "No photo submitted")
        self.assertTrue(any("best.onnx" in c.value for c in self.app.caption))
        self.assertEqual(len(self.app.dataframe[0].value), 72)

    def test_missing_model_with_upload(self):
        self.app.run()
        self.upload_photo()
        self.assert_assessment("Low", "Photo check unavailable")
        self.assertTrue(any("photo check is unavailable" in w.value for w in self.app.warning))
        self.research_view().run()
        self.assert_assessment("Low", "Photo check unavailable")
        self.assertTrue(any("weights are missing" in e.value for e in self.app.error))

    def test_research_empty_result_then_failed_prediction(self):
        self.weights.touch()
        result = Mock(boxes=None, speed={})
        result.plot.return_value = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch("ultralytics.YOLO"), patch("inference.predict_image", return_value=result) as predict:
            self.app.run()
            self.research_view().run()
            self.upload_photo()
            self.assert_assessment("Low", "Photo check flagged nothing")
            self.assertEqual(self.metric("Detections"), "0")
            predict.side_effect = RuntimeError("test prediction failure")
            self.app.slider(key="confidence").set_value(0.2).run()
            self.assert_assessment("Low", "Photo check unavailable")
            self.assertTrue(any("test prediction failure" in e.value for e in self.app.error))
            self.assertFalse(any(m.label == "Detections" for m in self.app.metric))

    def test_farmer_failed_prediction_keeps_risk_and_action_visible(self):
        self.weights.touch()
        with patch("ultralytics.YOLO"), patch("inference.predict_image", side_effect=RuntimeError("test prediction failure")):
            self.app.run()
            self.upload_photo()
        self.assert_assessment("Low", "Photo check unavailable")
        self.assertTrue(any("photo check is unavailable" in w.value for w in self.app.warning))
        self.assertFalse(self.app.error)
        self.assertIn("Keep checking under maize leaves", self.markdown_text())

    def test_farmer_detection_language_and_optional_heatmap(self):
        self.weights.touch()
        result = self.detection_result()
        with (
            patch("ultralytics.YOLO"),
            patch("inference.predict_image", return_value=result) as predict,
            patch("inference.generate_gradcam", return_value=Image.new("RGB", (32, 32))) as cam,
        ):
            self.app.run()
            self.upload_photo()
            self.assert_assessment("Moderate", "Possible Fall Armyworm leaf damage")
            self.assertNotIn("Message for the farmer", self.markdown_text())
            cam.assert_not_called()
            self.app.selectbox(key="language").select("Draft Luganda").run()
            self.assert_assessment("Moderate", "Okugezesa")
            self.app.checkbox(key="farmer_heatmap").check().run()
            cam.assert_called_once()
            predict.assert_called_once()

    def test_gradcam_failure_preserves_positive_detection(self):
        self.weights.touch()
        result = self.detection_result()
        with (
            patch("ultralytics.YOLO", side_effect=[Mock(), Mock()]),
            patch("inference.predict_image", return_value=result) as predict,
            patch("inference.generate_gradcam", side_effect=RuntimeError("test heatmap failure")) as cam,
        ):
            self.app.run()
            self.research_view().run()
            self.app.selectbox(key="scenario").select("Higher-risk scenario")
            self.upload_photo()
            self.assert_assessment("High", "Photo suggests possible armyworm signs")
            self.assertTrue(any("test heatmap failure" in e.value for e in self.app.error))
            self.assertIsNot(predict.call_args.args[0], cam.call_args.args[0])
            self.assertEqual(self.metric("Detections"), "1")
            self.assertEqual(self.metric("Inference"), "12.5 ms")
            box = self.app.dataframe[0].value.iloc[0]
            self.assertEqual(box["Class"], "fall-armyworm-larval-damage")
            self.assertAlmostEqual(box["Confidence"], 0.8)
            self.assertEqual(box["x2"], 20)
            self.assertIn("Message for the farmer", self.markdown_text())

    @staticmethod
    def detection_result():
        from ultralytics.engine.results import Boxes

        result = Mock(names={0: "fall-armyworm-larval-damage"}, speed={"inference": 12.5})
        result.boxes = Boxes(np.array([[0, 0, 20, 20, 0.8, 0]]), (32, 32))
        result.plot.return_value = np.zeros((32, 32, 3), dtype=np.uint8)
        return result
