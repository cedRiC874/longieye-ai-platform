"""Application service that keeps model details out of the API layer."""

from __future__ import annotations

from .domain import LongitudinalCase
from .features import extract_features
from .model import DemoRiskModel
from .telemetry import normalize_request_id


DISCLAIMER = (
    "Synthetic engineering demonstration only; not validated for clinical use."
)


class RiskPredictionService:
    def __init__(self, model: DemoRiskModel) -> None:
        self.model = model

    def predict(
        self,
        case: LongitudinalCase,
        case_id: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, object]:
        features = extract_features(case)
        probabilities = self.model.predict(features)
        return {
            "request_id": normalize_request_id(request_id),
            "case_id": case_id,
            "model": {
                "model_id": self.model.metadata.get("model_id"),
                "model_stage": self.model.metadata.get("model_stage"),
                "training_data": self.model.metadata.get("training_data"),
            },
            "predictions": {
                eye: {"demo_probability": round(probability, 6)}
                for eye, probability in probabilities.items()
            },
            "derived_features": {
                name: round(value, 6) for name, value in features.items()
            },
            "clinical_use": False,
            "disclaimer": DISCLAIMER,
        }
