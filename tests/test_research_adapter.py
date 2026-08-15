import hashlib
import io
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pytest

from longieye.features import FEATURE_ORDER
from longieye.model import DemoRiskModel
from longieye.research import (
    ARTIFACT_MAX_BYTES,
    SUPPORTED_TORCH_VERSION,
    ApprovalReceipt,
    ExternalJsonApprovalPolicy,
    GoldenCase,
    ResearchArtifactError,
    ResearchManifest,
    ResearchModelAdapter,
    golden_cases_sha256,
    preprocessing_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = PROJECT_ROOT / "configs" / "research_manifest.template.json"


def manifest_payload():
    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))


def default_golden_cases():
    return (
        GoldenCase((0.0,) * len(FEATURE_ORDER), (0.2, 0.8)),
        GoldenCase(tuple(float(index) for index in range(len(FEATURE_ORDER))), (0.3, 0.7)),
        GoldenCase((1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 1.0, 0.4, -0.2), (0.4, 0.6)),
    )


def set_preprocessing(payload, means, stds):
    payload["preprocessing"] = {
        "kind": "standardize",
        "mean": list(means),
        "std": list(stds),
        "sha256": preprocessing_sha256(means, stds),
    }


def make_loadable_payload(
    artifact: bytes,
    *,
    cases=None,
    means=None,
    stds=None,
):
    payload = manifest_payload()
    model_id = "synthetic-pytorch-fixture-v1"
    payload["model"]["model_id"] = model_id
    payload["provenance"]["source_commit"] = "1" * 40
    payload["authorization_request"] = {
        "reference_id": "REQ-synthetic-test",
        "requested_scope": "local_adapter_evaluation",
    }
    payload["artifact"]["sha256"] = hashlib.sha256(artifact).hexdigest()
    model_card = (
        f"# {model_id}\n\nmodel_id: {model_id}\nclinical_use: false\n"
        "Synthetic adapter fixture for contract testing only.\n"
    ).encode()
    payload["provenance"]["model_card_sha256"] = hashlib.sha256(
        model_card
    ).hexdigest()
    selected_cases = tuple(cases or default_golden_cases())
    tolerance = 1e-6
    payload["self_test"] = {
        "status": "exported",
        "tolerance": tolerance,
        "cases": [
            {
                "features": list(case.features),
                "expected_scores": list(case.expected_scores),
            }
            for case in selected_cases
        ],
        "sha256": golden_cases_sha256(tolerance, selected_cases),
    }
    if means is not None or stds is not None:
        set_preprocessing(
            payload,
            means if means is not None else [0.0] * len(FEATURE_ORDER),
            stds if stds is not None else [1.0] * len(FEATURE_ORDER),
        )
    return payload, model_card


def write_package(tmp_path, artifact=b"synthetic-test-artifact", **kwargs):
    payload, model_card = make_loadable_payload(artifact, **kwargs)
    (tmp_path / "research_model.pt").write_bytes(artifact)
    (tmp_path / "MODEL_CARD_RESEARCH.md").write_bytes(model_card)
    (tmp_path / "manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return payload


def external_policy_for(package_path, receipt_path):
    manifest = ResearchManifest.from_path(package_path / "manifest.json")
    request = manifest.approval_request()
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_id": "SYNTHETIC-EXTERNAL-CI",
                "request_sha256": request.request_sha256,
                "approved_scope": request.requested_scope,
                "issued_at_utc": "2026-01-01T00:00:00+00:00",
                "expires_at_utc": "2099-01-01T00:00:00+00:00",
                "revoked": False,
            }
        ),
        encoding="utf-8",
    )
    return ExternalJsonApprovalPolicy(receipt_path)


class MatchingPolicy:
    def __init__(self):
        self.requests = []

    def authorize(self, request):
        self.requests.append(request)
        return ApprovalReceipt(
            receipt_id="SYNTHETIC-TEST-RECEIPT",
            request_sha256=request.request_sha256,
            approved_scope=request.requested_scope,
            issued_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
            expires_at_utc=datetime(2099, 1, 1, tzinfo=timezone.utc),
            revoked=False,
        )


class DenyPolicy:
    def __init__(self):
        self.called = False

    def authorize(self, request):
        self.called = True
        return None


def optional_torch():
    try:
        import torch
    except ImportError:
        if os.getenv("LONGIEYE_REQUIRE_TORCH") == "1":
            pytest.fail("PyTorch is required in the research-adapter CI job")
        pytest.skip("optional PyTorch runtime is not installed")
    assert str(torch.__version__).split("+", 1)[0] == SUPPORTED_TORCH_VERSION
    return torch


def test_public_template_is_structurally_valid_but_cannot_request_approval():
    manifest = ResearchManifest.from_path(TEMPLATE_PATH)

    assert manifest.authorization_reference is None
    assert manifest.self_test_status == "not_exported"
    with pytest.raises(ResearchArtifactError) as exc_info:
        manifest.approval_request()
    assert exc_info.value.code == "approval_required"

    payload = manifest_payload()
    constant_cases = (
        GoldenCase((0.0,) * len(FEATURE_ORDER), (0.2, 0.8)),
        GoldenCase(tuple(float(i) for i in range(len(FEATURE_ORDER))), (0.3, 0.8)),
        GoldenCase((1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 1.0, 0.4, -0.2), (0.4, 0.8)),
    )
    payload["self_test"] = {
        "status": "exported",
        "tolerance": 1e-6,
        "cases": [
            {"features": list(case.features), "expected_scores": list(case.expected_scores)}
            for case in constant_cases
        ],
        "sha256": golden_cases_sha256(1e-6, constant_cases),
    }
    with pytest.raises(ResearchArtifactError) as coverage_error:
        ResearchManifest.from_mapping(payload)
    assert coverage_error.value.code == "manifest_invalid"

    injected_identifier = manifest_payload()
    injected_identifier["model"]["model_id"] = "safe\n| AUC | 0.999 |"
    with pytest.raises(ResearchArtifactError) as identifier_error:
        ResearchManifest.from_mapping(injected_identifier)
    assert identifier_error.value.code == "manifest_invalid"


def test_complete_package_requires_external_approval_before_file_checks(tmp_path):
    write_package(tmp_path)
    (tmp_path / "research_model.pt").unlink()
    policy = DenyPolicy()

    with pytest.raises(ResearchArtifactError) as exc_info:
        ResearchModelAdapter.from_package(tmp_path, approval_policy=policy)

    assert exc_info.value.code == "approval_required"
    assert policy.called is True


def test_approval_receipt_must_bind_exact_manifest_request(tmp_path):
    payload = write_package(tmp_path)
    original_request = ResearchManifest.from_mapping(payload).approval_request()
    changed_payload = json.loads(json.dumps(payload))
    changed_payload["model"]["training_data"] = "different-approved-source"
    changed_request = ResearchManifest.from_mapping(
        changed_payload
    ).approval_request()
    assert changed_request.request_sha256 != original_request.request_sha256

    class WrongReceiptPolicy(MatchingPolicy):
        def authorize(self, request):
            receipt = super().authorize(request)
            return ApprovalReceipt(
                receipt_id=receipt.receipt_id,
                request_sha256="f" * 64,
                approved_scope=receipt.approved_scope,
                issued_at_utc=receipt.issued_at_utc,
                expires_at_utc=receipt.expires_at_utc,
                revoked=False,
            )

    with pytest.raises(ResearchArtifactError) as exc_info:
        ResearchModelAdapter.from_package(
            tmp_path, approval_policy=WrongReceiptPolicy()
        )

    assert exc_info.value.code == "approval_invalid"

    class WrongTypePolicy:
        def authorize(self, request):
            return object()

    with pytest.raises(ResearchArtifactError) as type_error:
        ResearchModelAdapter.from_package(
            tmp_path, approval_policy=WrongTypePolicy()
        )
    assert type_error.value.code == "approval_invalid"


@pytest.mark.parametrize(
    ("issued_at", "expires_at", "revoked"),
    [
        (
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2021, 1, 1, tzinfo=timezone.utc),
            False,
        ),
        (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2099, 1, 1, tzinfo=timezone.utc),
            True,
        ),
    ],
)
def test_expired_or_revoked_receipt_is_rejected(
    tmp_path, issued_at, expires_at, revoked
):
    write_package(tmp_path)

    class InvalidLifecyclePolicy(MatchingPolicy):
        def authorize(self, request):
            receipt = super().authorize(request)
            return ApprovalReceipt(
                receipt_id=receipt.receipt_id,
                request_sha256=receipt.request_sha256,
                approved_scope=receipt.approved_scope,
                issued_at_utc=issued_at,
                expires_at_utc=expires_at,
                revoked=revoked,
            )

    with pytest.raises(ResearchArtifactError) as exc_info:
        ResearchModelAdapter.from_package(
            tmp_path, approval_policy=InvalidLifecyclePolicy()
        )

    assert exc_info.value.code == "approval_invalid"


def test_artifact_digest_mismatch_fails_after_external_approval(tmp_path):
    write_package(tmp_path)
    (tmp_path / "research_model.pt").write_bytes(b"tampered")

    with pytest.raises(ResearchArtifactError) as exc_info:
        ResearchModelAdapter.from_package(
            tmp_path, approval_policy=MatchingPolicy()
        )

    assert exc_info.value.code == "artifact_digest_mismatch"


def test_model_card_digest_and_completion_are_both_enforced(tmp_path):
    payload = write_package(tmp_path)
    card_path = tmp_path / "MODEL_CARD_RESEARCH.md"
    card_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ResearchArtifactError) as digest_error:
        ResearchModelAdapter.from_package(
            tmp_path, approval_policy=MatchingPolicy()
        )
    assert digest_error.value.code == "model_card_digest_mismatch"

    incomplete = b"synthetic-pytorch-fixture-v1 <required>"
    card_path.write_bytes(incomplete)
    payload["provenance"]["model_card_sha256"] = hashlib.sha256(
        incomplete
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResearchArtifactError) as content_error:
        ResearchModelAdapter.from_package(
            tmp_path, approval_policy=MatchingPolicy()
        )
    assert content_error.value.code == "model_card_incomplete"

    missing_machine_id = (
        b"# synthetic-pytorch-fixture-v1\n\nclinical_use: false\n"
        b"Completed synthetic test card.\n"
    )
    card_path.write_bytes(missing_machine_id)
    payload["provenance"]["model_card_sha256"] = hashlib.sha256(
        missing_machine_id
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResearchArtifactError) as machine_line_error:
        ResearchModelAdapter.from_package(
            tmp_path, approval_policy=MatchingPolicy()
        )
    assert machine_line_error.value.code == "model_card_incomplete"


def test_package_rejects_extra_files_and_oversized_artifact(tmp_path):
    write_package(tmp_path)
    (tmp_path / "participants.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ResearchArtifactError) as inventory_error:
        ResearchModelAdapter.from_package(
            tmp_path, approval_policy=MatchingPolicy()
        )
    assert inventory_error.value.code == "package_inventory_invalid"

    other = tmp_path / "oversized"
    other.mkdir()
    large_artifact = b"x" * (ARTIFACT_MAX_BYTES + 1)
    write_package(other, artifact=large_artifact)
    with pytest.raises(ResearchArtifactError) as size_error:
        ResearchModelAdapter.from_package(
            other, approval_policy=MatchingPolicy()
        )
    assert size_error.value.code == "package_file_too_large"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.update(schema_version=True),
        lambda payload: payload["adapter"].update(input_shape=[True, 9]),
        lambda payload: payload["feature_contract"].update(followup_months=12.0),
        lambda payload: payload["feature_contract"]["order"].reverse(),
        lambda payload: payload["preprocessing"]["std"].__setitem__(0, 0),
        lambda payload: payload["preprocessing"]["mean"].__setitem__(0, "0"),
        lambda payload: payload["preprocessing"]["mean"].__setitem__(0, math.nan),
        lambda payload: payload["output_contract"].update(order=["os", "od"]),
        lambda payload: payload["model"].update(clinical_use=True),
        lambda payload: payload["provenance"].update(model_card="../private.md"),
    ],
)
def test_manifest_contract_rejects_wrong_types_and_mismatches(mutator):
    payload = manifest_payload()
    mutator(payload)

    with pytest.raises(ResearchArtifactError) as exc_info:
        ResearchManifest.from_mapping(payload)
    assert exc_info.value.code == "manifest_invalid"


def test_external_json_receipt_is_valid_only_outside_package(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    write_package(package)
    manifest = ResearchManifest.from_path(package / "manifest.json")
    request = manifest.approval_request()
    receipt_payload = {
        "schema_version": 1,
        "receipt_id": "SYNTHETIC-EXTERNAL-1",
        "request_sha256": request.request_sha256,
        "approved_scope": request.requested_scope,
        "issued_at_utc": "2026-01-01T00:00:00+00:00",
        "expires_at_utc": "2099-01-01T00:00:00+00:00",
        "revoked": False,
    }
    external_receipt = tmp_path / "approval.json"
    external_receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
    policy = ExternalJsonApprovalPolicy(external_receipt)
    assert policy.authorize(request).request_sha256 == request.request_sha256

    external_receipt.write_bytes(b'{"schema_version":' + b"9" * 5000 + b"}")
    with pytest.raises(ResearchArtifactError) as integer_error:
        policy.authorize(request)
    assert integer_error.value.code == "approval_invalid"

    external_receipt.write_bytes(b"[" * 5000 + b"0" + b"]" * 5000)
    with pytest.raises(ResearchArtifactError) as recursion_error:
        policy.authorize(request)
    assert recursion_error.value.code == "approval_invalid"

    inside_receipt = package / "approval.json"
    inside_receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
    with pytest.raises(ResearchArtifactError) as exc_info:
        ResearchModelAdapter.from_package(
            package, approval_policy=ExternalJsonApprovalPolicy(inside_receipt)
        )
    assert exc_info.value.code == "approval_invalid"


def test_safe_errors_suppress_private_path_exception_chains(tmp_path):
    private_path = tmp_path / "private-person-name" / "missing.json"

    with pytest.raises(ResearchArtifactError) as exc_info:
        ResearchManifest.from_path(private_path)

    rendered = "".join(
        traceback.format_exception(
            type(exc_info.value), exc_info.value, exc_info.value.__traceback__
        )
    )
    assert exc_info.value.__cause__ is None
    assert "private-person-name" not in rendered

    huge_integer_json = b'{"schema_version":' + b"9" * 5000 + b"}"
    with pytest.raises(ResearchArtifactError) as integer_error:
        ResearchManifest.from_bytes(huge_integer_json)
    assert integer_error.value.code == "manifest_unreadable"

    deeply_nested_json = b"[" * 5000 + b"0" + b"]" * 5000
    with pytest.raises(ResearchArtifactError) as recursion_error:
        ResearchManifest.from_bytes(deeply_nested_json)
    assert recursion_error.value.code == "manifest_unreadable"


def test_full_adapter_parity_with_temporary_synthetic_pytorch_state_dict(
    tmp_path,
):
    torch = optional_torch()
    demo_config = json.loads(
        (PROJECT_ROOT / "configs" / "demo_model.json").read_text(encoding="utf-8")
    )
    artifact_buffer = io.BytesIO()
    torch.save(
        {
            "weight": torch.tensor(
                [
                    demo_config["heads"]["od"]["coefficients"],
                    demo_config["heads"]["os"]["coefficients"],
                ],
                dtype=torch.float32,
            ),
            "bias": torch.tensor(
                [
                    demo_config["heads"]["od"]["intercept"],
                    demo_config["heads"]["os"]["intercept"],
                ],
                dtype=torch.float32,
            ),
        },
        artifact_buffer,
    )
    artifact = artifact_buffer.getvalue()
    demo = DemoRiskModel(demo_config)
    raw_vectors = (
        (0.0,) * len(FEATURE_ORDER),
        tuple(float(index) for index in range(len(FEATURE_ORDER))),
        (1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 1.0, 0.4, -0.2),
    )
    cases = tuple(
        GoldenCase(
            features=vector,
            expected_scores=(
                demo.predict(dict(zip(FEATURE_ORDER, vector, strict=True)))["od"],
                demo.predict(dict(zip(FEATURE_ORDER, vector, strict=True)))["os"],
            ),
        )
        for vector in raw_vectors
    )
    package_path = tmp_path / "package"
    package_path.mkdir()
    write_package(
        package_path,
        artifact=artifact,
        cases=cases,
        means=demo_config["normalization"]["mean"],
        stds=demo_config["normalization"]["std"],
    )

    adapter = ResearchModelAdapter.from_package(
        package_path,
        approval_policy=external_policy_for(
            package_path, tmp_path / "approval.json"
        ),
    )

    wrong_cases = list(cases)
    wrong_cases[0] = GoldenCase(
        wrong_cases[0].features,
        (wrong_cases[0].expected_scores[0] + 0.01, wrong_cases[0].expected_scores[1]),
    )
    bad_golden_package = tmp_path / "bad-golden"
    bad_golden_package.mkdir()
    write_package(
        bad_golden_package,
        artifact=artifact,
        cases=wrong_cases,
        means=demo_config["normalization"]["mean"],
        stds=demo_config["normalization"]["std"],
    )
    with pytest.raises(ResearchArtifactError) as golden_error:
        ResearchModelAdapter.from_package(
            bad_golden_package,
            approval_policy=external_policy_for(
                bad_golden_package, tmp_path / "bad-golden-approval.json"
            ),
        )
    assert golden_error.value.code == "self_test_failed"

    for vector in raw_vectors:
        features = dict(zip(FEATURE_ORDER, vector, strict=True))
        assert adapter.predict(features) == pytest.approx(
            demo.predict(features), abs=1e-6
        )
    assert adapter.readiness().self_test == "passed"
    assert adapter.metadata["approval_receipt_id"] == "SYNTHETIC-EXTERNAL-CI"

    report_dir = tmp_path / "comparison-report"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "compare_adapters.py"),
            str(package_path),
            "--approval-receipt",
            str(tmp_path / "approval.json"),
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--output-dir",
            str(report_dir),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    comparison_json = json.loads(
        (report_dir / "adapter_comparison.json").read_text(encoding="utf-8")
    )
    comparison_markdown = (report_dir / "adapter_comparison.md").read_text(
        encoding="utf-8"
    )
    assert comparison_json["authorization_verification"][
        "performed_by_builder"
    ] is False
    assert comparison_json["modes"]["research_adapter"][
        "approval_receipt_id"
    ] == "SYNTHETIC-EXTERNAL-CI"
    assert "builder 不验证授权" in comparison_markdown
    assert "AUC 0.999" not in comparison_markdown

    with torch.no_grad():
        adapter._runtime._model.weight[0, 0] += 1.0
    assert adapter.readiness().status == "not_ready"
    with pytest.raises(ResearchArtifactError) as exc_info:
        adapter.predict(dict(zip(FEATURE_ORDER, raw_vectors[0], strict=True)))
    assert exc_info.value.code == "runtime_inference_failed"

    revocation_adapter = ResearchModelAdapter.from_package(
        package_path,
        approval_policy=external_policy_for(
            package_path, tmp_path / "approval.json"
        ),
    )
    receipt_payload = json.loads(
        (tmp_path / "approval.json").read_text(encoding="utf-8")
    )
    receipt_payload["revoked"] = True
    (tmp_path / "approval.json").write_text(
        json.dumps(receipt_payload), encoding="utf-8"
    )
    revoked_readiness = revocation_adapter.readiness()
    assert revoked_readiness.status == "not_ready"
    assert revoked_readiness.error_code == "approval_invalid"
    with pytest.raises(ResearchArtifactError) as revoked_error:
        revocation_adapter.predict(
            dict(zip(FEATURE_ORDER, raw_vectors[0], strict=True))
        )
    assert revoked_error.value.code == "approval_invalid"


@pytest.mark.parametrize(
    "bad_state",
    ["missing_key", "wrong_shape", "wrong_dtype", "non_finite"],
)
def test_verified_package_rejects_malformed_synthetic_state_dicts(
    tmp_path, bad_state
):
    torch = optional_torch()
    weight = torch.zeros((2, len(FEATURE_ORDER)), dtype=torch.float32)
    bias = torch.zeros((2,), dtype=torch.float32)
    state = {"weight": weight, "bias": bias}
    if bad_state == "missing_key":
        state.pop("bias")
    elif bad_state == "wrong_shape":
        state["weight"] = torch.zeros((1, len(FEATURE_ORDER)))
    elif bad_state == "wrong_dtype":
        state["bias"] = bias.to(dtype=torch.float64)
    elif bad_state == "non_finite":
        state["weight"][0, 0] = math.nan
    artifact_buffer = io.BytesIO()
    torch.save(state, artifact_buffer)

    package_path = tmp_path / "package"
    package_path.mkdir()
    write_package(package_path, artifact=artifact_buffer.getvalue())

    with pytest.raises(ResearchArtifactError) as exc_info:
        ResearchModelAdapter.from_package(
            package_path,
            approval_policy=external_policy_for(
                package_path, tmp_path / "approval.json"
            ),
        )
    assert exc_info.value.code == "runtime_load_failed"
    assert exc_info.value.__cause__ is None
