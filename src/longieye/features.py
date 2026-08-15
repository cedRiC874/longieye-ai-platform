"""Feature contract for the public engineering demo."""

from __future__ import annotations

from collections.abc import Mapping

from .domain import LongitudinalCase


FEATURE_ORDER = (
    "sex_y1",
    "height_delta_cm",
    "weight_delta_kg",
    "sbp_delta_mmhg",
    "dbp_delta_mmhg",
    "waist_delta_cm",
    "wears_glasses_delta",
    "axial_length_od_delta_mm",
    "axial_length_os_delta_mm",
)


def extract_features(case: LongitudinalCase) -> dict[str, float]:
    """Return static sex plus eight Y2-minus-Y1 changes.

    Spherical-equivalent measurements are deliberately excluded, matching the
    leakage-avoidance boundary of the source research.
    """

    y1, y2 = case.y1, case.y2
    return {
        "sex_y1": float(y1.sex_code),
        "height_delta_cm": float(y2.height_cm - y1.height_cm),
        "weight_delta_kg": float(y2.weight_kg - y1.weight_kg),
        "sbp_delta_mmhg": float(y2.sbp_mmhg - y1.sbp_mmhg),
        "dbp_delta_mmhg": float(y2.dbp_mmhg - y1.dbp_mmhg),
        "waist_delta_cm": float(y2.waist_cm - y1.waist_cm),
        "wears_glasses_delta": float(y2.wears_glasses - y1.wears_glasses),
        "axial_length_od_delta_mm": float(
            y2.axial_length_od_mm - y1.axial_length_od_mm
        ),
        "axial_length_os_delta_mm": float(
            y2.axial_length_os_mm - y1.axial_length_os_mm
        ),
    }


def ordered_values(features: Mapping[str, float]) -> list[float]:
    missing = [name for name in FEATURE_ORDER if name not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return [float(features[name]) for name in FEATURE_ORDER]
