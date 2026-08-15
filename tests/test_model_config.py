import copy
import json
from pathlib import Path

import pytest

from longieye.model import DemoRiskModel, ModelConfigError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def valid_config():
    return json.loads((PROJECT_ROOT / "configs" / "demo_model.json").read_text())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cfg: cfg["normalization"]["mean"].__setitem__(0, float("nan")), "finite number"),
        (lambda cfg: cfg["heads"]["od"].pop("intercept"), "heads.od.intercept"),
        (lambda cfg: cfg["heads"]["os"]["coefficients"].__setitem__(0, "1.0"), "finite number"),
        (lambda cfg: cfg["metadata"].__setitem__("clinical_use", True), "clinical_use"),
        (lambda cfg: cfg.__setitem__("normalization", []), "normalization must"),
    ],
)
def test_rejects_malformed_artifacts_at_load_time(mutation, message):
    config = copy.deepcopy(valid_config())
    mutation(config)
    with pytest.raises(ModelConfigError, match=message):
        DemoRiskModel(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("feature_contract_version", "wrong-v99"),
        ("output_contract_version", "wrong-v99"),
        ("adapter_kind", "unknown"),
        ("adapter_version", 1),
        ("framework", "unknown"),
    ],
)
def test_demo_metadata_cannot_override_versioned_backend_contract(field, value):
    config = valid_config()
    config["metadata"][field] = value

    with pytest.raises(ModelConfigError, match=field):
        DemoRiskModel(config)
