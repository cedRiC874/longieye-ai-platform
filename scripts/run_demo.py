"""Run one local prediction without starting the web server."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from longieye.domain import LongitudinalCase, VisitMeasurements  # noqa: E402
from longieye.model import DemoRiskModel  # noqa: E402
from longieye.service import RiskPredictionService  # noqa: E402


def main() -> None:
    case = LongitudinalCase(
        y1=VisitMeasurements(
            sex_code=0,
            height_cm=145.0,
            weight_kg=38.0,
            sbp_mmhg=105.0,
            dbp_mmhg=68.0,
            waist_cm=62.0,
            wears_glasses=0,
            axial_length_od_mm=23.50,
            axial_length_os_mm=23.45,
        ),
        y2=VisitMeasurements(
            sex_code=0,
            height_cm=151.0,
            weight_kg=43.0,
            sbp_mmhg=108.0,
            dbp_mmhg=70.0,
            waist_cm=65.0,
            wears_glasses=1,
            axial_length_od_mm=23.82,
            axial_length_os_mm=23.74,
        ),
    )
    model = DemoRiskModel.from_path(PROJECT_ROOT / "configs" / "demo_model.json")
    result = RiskPredictionService(model).predict(case, case_id="demo-001")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
