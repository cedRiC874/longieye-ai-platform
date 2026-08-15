"""Privacy-safe domain objects used by both the API and model pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _bounded(name: str, value: float, lower: float, upper: float) -> None:
    if not isfinite(float(value)) or not lower <= float(value) <= upper:
        raise ValueError(f"`{name}` 必须在 {lower} 到 {upper} 之间。")


def _binary(name: str, value: int) -> None:
    if value not in (0, 1):
        raise ValueError(f"`{name}` 必须编码为 0 或 1。")


@dataclass(frozen=True)
class VisitMeasurements:
    """One anonymized clinical visit.

    The bounds are input-quality guards for the engineering demo. They are not
    diagnostic thresholds and must not be interpreted as clinical guidance.
    """

    sex_code: int
    height_cm: float
    weight_kg: float
    sbp_mmhg: float
    dbp_mmhg: float
    waist_cm: float
    wears_glasses: int
    axial_length_od_mm: float
    axial_length_os_mm: float

    def __post_init__(self) -> None:
        _binary("sex_code", self.sex_code)
        _binary("wears_glasses", self.wears_glasses)
        _bounded("height_cm", self.height_cm, 80.0, 220.0)
        _bounded("weight_kg", self.weight_kg, 15.0, 250.0)
        _bounded("sbp_mmhg", self.sbp_mmhg, 60.0, 240.0)
        _bounded("dbp_mmhg", self.dbp_mmhg, 30.0, 160.0)
        _bounded("waist_cm", self.waist_cm, 30.0, 200.0)
        _bounded("axial_length_od_mm", self.axial_length_od_mm, 15.0, 35.0)
        _bounded("axial_length_os_mm", self.axial_length_os_mm, 15.0, 35.0)


@dataclass(frozen=True)
class LongitudinalCase:
    """A paired Y1/Y2 record with no direct participant identifiers."""

    y1: VisitMeasurements
    y2: VisitMeasurements
    followup_months: int = 12

    def __post_init__(self) -> None:
        if self.y1.sex_code != self.y2.sex_code:
            raise ValueError("`sex_code` 在 Y1 与 Y2 之间必须保持不变。")
        if self.followup_months != 12:
            raise ValueError("演示特征契约要求两次随访间隔为12个月。")
