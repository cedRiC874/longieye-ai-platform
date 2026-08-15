from pathlib import Path

import pytest

from longieye.comparison import (
    AdapterComparisonError,
    build_engineering_comparison,
    comparison_markdown,
)
from longieye.features import FEATURE_ORDER
from longieye.model import DemoRiskModel
from longieye.model_contract import (
    AdapterReadiness,
    FEATURE_CONTRACT_VERSION,
    OUTPUT_CONTRACT_VERSION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FixedResearchBackend:
    metadata = {
        "model_id": "synthetic-research-contract-fixture",
        "model_stage": "research_locked",
        "adapter_kind": "test_vector_runtime",
        "adapter_version": "1",
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
        "artifact_sha256": "a" * 64,
        "preprocessing_sha256": "b" * 64,
        "model_card_sha256": "c" * 64,
        "golden_cases_sha256": "d" * 64,
        "approval_request_sha256": "e" * 64,
        "manifest_sha256": "1" * 64,
        "loader_contract_version": "research-package-loader-v1",
        "source_commit": "f" * 40,
        "approval_receipt_id": "SYNTHETIC-RECEIPT",
        "approval_scope": "local_adapter_evaluation",
        "approval_issued_at_utc": "2026-01-01T00:00:00+00:00",
        "approval_expires_at_utc": "2099-01-01T00:00:00+00:00",
        "framework": "pure_python_test_double",
        "framework_version": "1",
        "clinical_use": False,
    }

    def predict(self, features):
        return {"od": 0.4, "os": 0.6}

    def readiness(self):
        return AdapterReadiness(status="ready", self_test="passed")


def feature_vector():
    return {name: 0.0 for name in FEATURE_ORDER}


def research_adapter():
    return FixedResearchBackend()


def test_report_separates_modes_and_never_stores_scores_or_model_metrics():
    demo = DemoRiskModel.from_path(PROJECT_ROOT / "configs" / "demo_model.json")

    report = build_engineering_comparison(
        demo, research_adapter(), feature_vector(), warmup=1, iterations=3
    )

    assert report["scope"] == "loaded_adapter_contract_and_runtime_only"
    assert set(report["modes"]) == {"synthetic_demo", "research_adapter"}
    assert report["modes"]["synthetic_demo"]["runtime"]["iterations"] == 3
    assert len(report["modes"]["synthetic_demo"]["artifact_sha256"]) == 64
    assert report["modes"]["research_adapter"]["preprocessing_sha256"] == "b" * 64
    assert report["modes"]["research_adapter"]["contract_checks"] == {
        "exact_bilateral_outputs": True,
        "finite_bounded_scores": True,
        "deterministic_repeat": True,
        "input_unchanged": True,
    }
    assert report["metric_namespaces"]["synthetic_sanity_metrics"][
        "included"
    ] is False
    assert report["metric_namespaces"]["authorized_research_metrics"][
        "included"
    ] is False
    assert report["stores_model_outputs"] is False
    assert report["authorization_verification"] == {
        "performed_by_builder": False,
        "standard_cli_requires_verified_package": True,
        "reason": (
            "the generic builder records backend evidence but is not an "
            "authorization security boundary"
        ),
    }
    assert "predictions" not in str(report).lower()
    assert "auc" not in str(report).lower()
    markdown = comparison_markdown(report)
    assert "只比较已加载适配器的合同与本机顺序运行时间" in markdown
    assert "authorized_research_metrics`：未包含" in markdown
    assert "builder 不验证授权" in markdown


def test_report_refuses_to_put_same_stage_in_both_namespaces():
    demo = DemoRiskModel.from_path(PROJECT_ROOT / "configs" / "demo_model.json")

    with pytest.raises(AdapterComparisonError):
        build_engineering_comparison(
            demo, demo, feature_vector(), warmup=1, iterations=1
        )

    incomplete = research_adapter()
    incomplete.metadata = dict(incomplete.metadata)
    incomplete.metadata.pop("approval_request_sha256")
    with pytest.raises(AdapterComparisonError, match="approval_request_sha256"):
        build_engineering_comparison(
            demo, incomplete, feature_vector(), warmup=1, iterations=1
        )

    injected = research_adapter()
    injected.metadata = dict(injected.metadata)
    injected.metadata["model_id"] = "safe\n| AUC | 0.999 |"
    with pytest.raises(AdapterComparisonError, match="model_id"):
        build_engineering_comparison(
            demo, injected, feature_vector(), warmup=1, iterations=1
        )
