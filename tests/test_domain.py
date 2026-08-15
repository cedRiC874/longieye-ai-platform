import unittest

from longieye.domain import LongitudinalCase, VisitMeasurements


def visit(**overrides):
    values = {
        "sex_code": 0,
        "height_cm": 145.0,
        "weight_kg": 38.0,
        "sbp_mmhg": 105.0,
        "dbp_mmhg": 68.0,
        "waist_cm": 62.0,
        "wears_glasses": 0,
        "axial_length_od_mm": 23.5,
        "axial_length_os_mm": 23.45,
    }
    values.update(overrides)
    return VisitMeasurements(**values)


class DomainTests(unittest.TestCase):
    def test_rejects_out_of_range_axial_length(self):
        with self.assertRaisesRegex(ValueError, "axial_length_od_mm"):
            visit(axial_length_od_mm=42.0)

    def test_rejects_sex_change_between_visits(self):
        with self.assertRaisesRegex(ValueError, "必须保持不变"):
            LongitudinalCase(y1=visit(sex_code=0), y2=visit(sex_code=1))


if __name__ == "__main__":
    unittest.main()
