"""Stable contracts shared by public demo and offline research adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .features import FEATURE_ORDER


FEATURE_CONTRACT_VERSION = "longitudinal-static-sex-delta8-v1"
OUTPUT_CONTRACT_VERSION = "bilateral-non-clinical-score-v1"
OUTPUT_ORDER = ("od", "os")


class ModelContractError(ValueError):
    """Raised when a backend violates the shared inference contract."""


@dataclass(frozen=True)
class AdapterReadiness:
    """Cached, privacy-safe readiness state for an already loaded backend."""

    status: str
    self_test: str
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"ready", "not_ready"}:
            raise ValueError("status must be ready or not_ready")
        if self.self_test not in {"passed", "failed", "not_run"}:
            raise ValueError("self_test has an unsupported value")
        if self.status == "ready" and self.self_test != "passed":
            raise ValueError("a ready backend must have passed its self-test")


@runtime_checkable
class RiskModelBackend(Protocol):
    """Minimal model boundary used by the application service."""

    @property
    def metadata(self) -> Mapping[str, object]: ...

    def predict(self, features: Mapping[str, float]) -> Mapping[str, float]: ...

    def readiness(self) -> AdapterReadiness: ...


def validated_feature_values(features: Mapping[str, float]) -> tuple[float, ...]:
    """Validate and order one feature mapping without echoing submitted values."""

    if not isinstance(features, Mapping):
        raise ModelContractError("features must be a mapping")
    if set(features) != set(FEATURE_ORDER):
        raise ModelContractError("features do not match the versioned feature contract")

    values: list[float] = []
    for name in FEATURE_ORDER:
        raw_value = features[name]
        if isinstance(raw_value, bool):
            raise ModelContractError(f"feature {name} must be a finite number")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ModelContractError(
                f"feature {name} must be a finite number"
            ) from exc
        if not math.isfinite(value):
            raise ModelContractError(f"feature {name} must be a finite number")
        values.append(value)
    return tuple(values)


def validated_scores(scores: Mapping[str, float]) -> dict[str, float]:
    """Return exact OD/OS scores after enforcing finite probability-like bounds."""

    if not isinstance(scores, Mapping) or set(scores) != set(OUTPUT_ORDER):
        raise ModelContractError("backend output must contain exactly od and os")
    validated: dict[str, float] = {}
    for name in OUTPUT_ORDER:
        raw_value = scores[name]
        if isinstance(raw_value, bool):
            raise ModelContractError("backend output must contain finite scores")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ModelContractError(
                "backend output must contain finite scores"
            ) from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ModelContractError("backend output scores must be between 0 and 1")
        validated[name] = value
    return validated


def validated_runtime_output(values: Sequence[float]) -> dict[str, float]:
    """Map a two-value runtime result to the public bilateral score contract."""

    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != len(OUTPUT_ORDER)
    ):
        raise ModelContractError("runtime output must contain two scores")
    return validated_scores(dict(zip(OUTPUT_ORDER, values, strict=True)))
