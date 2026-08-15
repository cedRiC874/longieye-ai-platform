"""Small, inspectable inference backend for the synthetic demo model."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .features import FEATURE_ORDER, ordered_values


class ModelConfigError(ValueError):
    """Raised when a model artifact violates the public feature contract."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelConfigError(f"{name} must be a JSON object")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ModelConfigError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelConfigError(f"{name} must be a finite number") from exc
    if not math.isfinite(converted):
        raise ModelConfigError(f"{name} must be a finite number")
    return converted


def _finite_vector(value: object, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ModelConfigError(f"{name} must be a numeric array")
    return [
        _finite_number(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    ]


def _sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-value))


class DemoRiskModel:
    """Two-head logistic model trained only on deterministic synthetic data."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        config = _mapping(config, "model config")
        raw_features = config.get("feature_order")
        if not isinstance(raw_features, Sequence) or isinstance(
            raw_features, (str, bytes)
        ):
            raise ModelConfigError("feature_order must be a string array")
        configured_features = tuple(raw_features)
        if configured_features != FEATURE_ORDER:
            raise ModelConfigError(
                "Model feature_order does not match the application contract"
            )

        normalization = _mapping(config.get("normalization"), "normalization")
        self.means = _finite_vector(normalization.get("mean"), "normalization.mean")
        self.stds = _finite_vector(normalization.get("std"), "normalization.std")
        if len(self.means) != len(FEATURE_ORDER) or len(self.stds) != len(
            FEATURE_ORDER
        ):
            raise ModelConfigError("Normalization vectors have an invalid length")
        if any(value <= 0.0 for value in self.stds):
            raise ModelConfigError("Normalization standard deviations must be positive")

        heads = _mapping(config.get("heads"), "heads")
        if set(heads) != {"od", "os"}:
            raise ModelConfigError("Exactly two model heads, od and os, are required")
        self.heads: dict[str, dict[str, object]] = {}
        for name, raw_head in heads.items():
            head = _mapping(raw_head, f"heads.{name}")
            coefficients = _finite_vector(
                head.get("coefficients"), f"heads.{name}.coefficients"
            )
            if len(coefficients) != len(FEATURE_ORDER):
                raise ModelConfigError(f"Head {name} has an invalid coefficient count")
            self.heads[name] = {
                "intercept": _finite_number(
                    head.get("intercept"), f"heads.{name}.intercept"
                ),
                "coefficients": coefficients,
            }

        self.metadata = dict(_mapping(config.get("metadata"), "metadata"))
        if self.metadata.get("model_stage") != "demo_synthetic":
            raise ModelConfigError("Public scaffold accepts only demo_synthetic models")
        for field in ("model_id", "training_data"):
            if not isinstance(self.metadata.get(field), str) or not self.metadata[
                field
            ].strip():
                raise ModelConfigError(f"metadata.{field} must be a non-empty string")
        if self.metadata.get("clinical_use") is not False:
            raise ModelConfigError("metadata.clinical_use must be false")

    @classmethod
    def from_path(cls, path: str | Path) -> "DemoRiskModel":
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                return cls(json.load(handle))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelConfigError("Unable to load the model artifact") from exc

    def predict(self, features: Mapping[str, float]) -> dict[str, float]:
        values = ordered_values(features)
        standardized = [
            (value - mean) / std
            for value, mean, std in zip(values, self.means, self.stds, strict=True)
        ]
        output: dict[str, float] = {}
        for name, head in self.heads.items():
            coefficients: Sequence[float] = head["coefficients"]
            logit = float(head["intercept"]) + sum(
                float(weight) * value
                for weight, value in zip(coefficients, standardized, strict=True)
            )
            output[name] = _sigmoid(logit)
        return output
