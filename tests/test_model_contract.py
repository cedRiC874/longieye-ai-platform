import copy
import math
from pathlib import Path

import pytest

from longieye.features import FEATURE_ORDER
from longieye.model import DemoRiskModel
from longieye.model_contract import (
    AdapterReadiness,
    ModelContractError,
    RiskModelBackend,
    validated_feature_values,
    validated_runtime_output,
    validated_scores,
)
from longieye.service import RiskPredictionService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def valid_features():
    return {name: float(index) for index, name in enumerate(FEATURE_ORDER)}


def test_demo_backend_satisfies_shared_contract_without_mutating_input():
    model = DemoRiskModel.from_path(PROJECT_ROOT / "configs" / "demo_model.json")
    features = valid_features()
    before = copy.deepcopy(features)

    scores = model.predict(features)

    assert isinstance(model, RiskModelBackend)
    assert features == before
    assert set(scores) == {"od", "os"}
    assert all(0.0 <= score <= 1.0 for score in scores.values())
    assert model.readiness().status == "ready"


@pytest.mark.parametrize("bad_value", [True, math.nan, math.inf, "not-a-number"])
def test_feature_contract_rejects_non_finite_or_non_numeric_values(bad_value):
    features = valid_features()
    features[FEATURE_ORDER[0]] = bad_value

    with pytest.raises(ModelContractError):
        validated_feature_values(features)


def test_feature_contract_rejects_missing_and_extra_keys():
    missing = valid_features()
    missing.pop(FEATURE_ORDER[0])
    extra = valid_features() | {"unexpected": 1.0}

    with pytest.raises(ModelContractError):
        validated_feature_values(missing)
    with pytest.raises(ModelContractError):
        validated_feature_values(extra)


@pytest.mark.parametrize(
    "scores",
    [
        {"od": 0.1},
        {"od": 0.1, "os": 0.2, "extra": 0.3},
        {"od": math.nan, "os": 0.2},
        {"od": -0.1, "os": 0.2},
        {"od": 0.1, "os": True},
    ],
)
def test_output_contract_fails_closed(scores):
    with pytest.raises(ModelContractError):
        validated_scores(scores)


@pytest.mark.parametrize("output", [None, 1, object(), (value for value in [0.1, 0.2])])
def test_runtime_output_normalizes_non_sequences_to_contract_error(output):
    with pytest.raises(ModelContractError):
        validated_runtime_output(output)


def test_public_demo_service_refuses_research_stage_even_if_backend_is_ready():
    class ResearchBackend:
        metadata = {
            "model_id": "research-test",
            "model_stage": "research_locked",
            "training_data": "test-only",
            "clinical_use": False,
        }

        def predict(self, features):
            return {"od": 0.1, "os": 0.2}

        def readiness(self):
            return AdapterReadiness(status="ready", self_test="passed")

    with pytest.raises(ValueError, match="demo_synthetic"):
        RiskPredictionService(ResearchBackend())
