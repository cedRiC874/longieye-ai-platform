import pytest

from scripts.train_demo_model import canonicalize_artifact_numbers


def test_artifact_number_canonicalization_absorbs_platform_noise():
    windows_value = 0.14725998716523656
    linux_value = 0.14725998716523653

    assert canonicalize_artifact_numbers(windows_value) == canonicalize_artifact_numbers(
        linux_value
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_artifact_number_canonicalization_rejects_non_finite_values(value):
    with pytest.raises(ValueError, match="non-finite"):
        canonicalize_artifact_numbers({"coefficient": value})


def test_artifact_number_canonicalization_preserves_non_float_values():
    payload = {"clinical_use": False, "samples": 2000, "model_id": "demo"}

    assert canonicalize_artifact_numbers(payload) == payload
