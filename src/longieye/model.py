"""Small, inspectable inference backend for the synthetic demo model."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .features import FEATURE_ORDER
from .model_contract import (
    AdapterReadiness,
    FEATURE_CONTRACT_VERSION,
    OUTPUT_CONTRACT_VERSION,
    validated_feature_values,
    validated_scores,
)


class ModelConfigError(ValueError):
    """Raised when a model artifact violates the public feature contract."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelConfigError(f"{name} must be a JSON object")
    return value


def _finite_number(value: object, name: str) -> float:
    if type(value) not in {int, float}:
        raise ModelConfigError(f"{name} must be a finite number")
    converted = float(value)
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
        contract_defaults = {
            "adapter_kind": "json_logistic",
            "adapter_version": "1",
            "framework": "python_stdlib",
            "framework_version": "1",
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "output_contract_version": OUTPUT_CONTRACT_VERSION,
        }
        for field, expected in contract_defaults.items():
            configured = self.metadata.get(field)
            if configured != expected or not isinstance(configured, str):
                raise ModelConfigError(f"metadata.{field} violates the model contract")

        try:
            self.predict({name: 0.0 for name in FEATURE_ORDER})
        except (ModelConfigError, ValueError) as exc:
            raise ModelConfigError("Model self-test failed") from exc

    @classmethod
    def from_path(cls, path: str | Path) -> "DemoRiskModel":
        try:
            artifact_bytes = Path(path).read_bytes()
            model = cls(json.loads(artifact_bytes.decode("utf-8")))
            model.metadata["artifact_sha256"] = hashlib.sha256(
                artifact_bytes
            ).hexdigest()
            return model
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ModelConfigError("Unable to load the model artifact") from None

    def predict(self, features: Mapping[str, float]) -> dict[str, float]:
        values = validated_feature_values(features)
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
        return validated_scores(output)

    def readiness(self) -> AdapterReadiness:
        return AdapterReadiness(status="ready", self_test="passed")
