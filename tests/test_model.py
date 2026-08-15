import unittest
from pathlib import Path

from longieye.domain import LongitudinalCase, VisitMeasurements
from longieye.model import DemoRiskModel
from longieye.service import RiskPredictionService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RiskPredictionService(
            DemoRiskModel.from_path(PROJECT_ROOT / "configs" / "demo_model.json")
        )

    def test_returns_bounded_two_eye_demo_probabilities_and_safety_metadata(self):
        case = LongitudinalCase(
            y1=VisitMeasurements(0, 145, 38, 105, 68, 62, 0, 23.50, 23.45),
            y2=VisitMeasurements(0, 151, 43, 108, 70, 65, 1, 23.82, 23.74),
        )
        result = self.service.predict(case, case_id="unit-test")

        self.assertEqual(result["case_id"], "unit-test")
        self.assertFalse(result["clinical_use"])
        self.assertEqual(result["model"]["model_stage"], "demo_synthetic")
        self.assertEqual(set(result["predictions"]), {"od", "os"})
        for prediction in result["predictions"].values():
            self.assertGreaterEqual(prediction["demo_probability"], 0.0)
            self.assertLessEqual(prediction["demo_probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
