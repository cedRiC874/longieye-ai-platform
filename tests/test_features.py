import unittest

from longieye.domain import LongitudinalCase, VisitMeasurements
from longieye.features import FEATURE_ORDER, extract_features, ordered_values


class FeatureTests(unittest.TestCase):
    def test_extracts_static_sex_and_eight_deltas_in_contract_order(self):
        y1 = VisitMeasurements(0, 145, 38, 105, 68, 62, 0, 23.50, 23.45)
        y2 = VisitMeasurements(0, 151, 43, 108, 70, 65, 1, 23.82, 23.74)
        features = extract_features(LongitudinalCase(y1=y1, y2=y2))

        self.assertEqual(tuple(features), FEATURE_ORDER)
        self.assertAlmostEqual(features["height_delta_cm"], 6.0)
        self.assertAlmostEqual(features["axial_length_od_delta_mm"], 0.32)
        self.assertAlmostEqual(features["axial_length_os_delta_mm"], 0.29)
        self.assertEqual(ordered_values(features), list(features.values()))


if __name__ == "__main__":
    unittest.main()
