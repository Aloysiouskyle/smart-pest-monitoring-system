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
        app = self.app
        self.assertFalse(app.exception)
        self.assertEqual(next(m.value for m in app.metric if m.label == "Simulated Fall Armyworm risk"), level)
        self.assertTrue(any(evidence in m.value for m in app.markdown))
        captions = [c.value for c in app.caption]
        self.assertIn(LIMITATION, captions)
        self.assertTrue(any("human review" in text for text in captions))
        self.assertEqual(sum("characters · GSM-7" in text for text in captions), 2)

    def test_no_upload_and_scenario_selection(self):
        self.weights.touch()
        self.app.run()
        self.assert_assessment("Low", "No photo submitted")
        self.app.selectbox(key="scenario").select("Higher-risk scenario").run()
        self.assert_assessment("Moderate", "No photo submitted")

    def test_missing_model_with_upload(self):
        self.app.run()
        self.app.file_uploader[0].upload("maize.png", self.photo).run()
        self.assertTrue(any("weights are missing" in w.value for w in self.app.warning))
        self.assert_assessment("Low", "Photo check unavailable")

    def test_processed_empty_then_failed_prediction(self):
        self.weights.touch()
        result = Mock(boxes=None, speed={})
        result.plot.return_value = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch("ultralytics.YOLO"), patch("inference.predict_image", return_value=result) as predict:
            self.app.run()
            self.app.file_uploader[0].upload("maize.png", self.photo).run()
            self.assert_assessment("Low", "Photo check flagged nothing")
            self.assertEqual(next(m.value for m in self.app.metric if m.label == "Detections"), "0")
            predict.side_effect = RuntimeError("test prediction failure")
            self.app.slider[0].set_value(0.2).run()
            self.assert_assessment("Low", "Photo check unavailable")
            self.assertTrue(any("test prediction failure" in e.value for e in self.app.error))

    def test_gradcam_failure_preserves_positive_detection(self):
        from ultralytics.engine.results import Boxes

        self.weights.touch()
        result = Mock(names={0: "fall-armyworm-larva"}, speed={})
        result.boxes = Boxes(np.array([[0, 0, 20, 20, 0.8, 0]]), (32, 32))
        result.plot.return_value = np.zeros((32, 32, 3), dtype=np.uint8)
        with (
            patch("ultralytics.YOLO", side_effect=[Mock(), Mock()]),
            patch("inference.predict_image", return_value=result) as predict,
            patch("inference.generate_gradcam", side_effect=RuntimeError("test heatmap failure")) as cam,
        ):
            self.app.run()
            self.app.selectbox(key="scenario").select("Higher-risk scenario")
            self.app.file_uploader[0].upload("maize.png", self.photo).run()
            self.assert_assessment("High", "Photo suggests possible armyworm signs")
            self.assertTrue(any("test heatmap failure" in e.value for e in self.app.error))
            self.assertIsNot(predict.call_args.args[0], cam.call_args.args[0])
