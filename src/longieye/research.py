"""Fail-closed, offline adapter boundary for authorized research artifacts.

The public API never imports this module.  Package metadata is not trusted to
approve itself: callers must provide an approval policy backed by a receipt
stored outside the package.  Files are bounded, hashed once into memory and the
same verified bytes are passed to PyTorch, avoiding a hash/load path race.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import os
import re
import stat
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from .features import FEATURE_ORDER
from .model_contract import (
    AdapterReadiness,
    FEATURE_CONTRACT_VERSION,
    ModelContractError,
    OUTPUT_CONTRACT_VERSION,
    validated_feature_values,
    validated_runtime_output,
)


SUPPORTED_ARCHITECTURE = "bilateral_linear_v1"
SUPPORTED_TORCH_VERSION = "2.13.0"
SUPPORTED_APPROVAL_SCOPE = "local_adapter_evaluation"
RESEARCH_LOADER_CONTRACT_VERSION = "research-package-loader-v1"
MANIFEST_MAX_BYTES = 64 * 1024
MODEL_CARD_MAX_BYTES = 256 * 1024
ARTIFACT_MAX_BYTES = 1024 * 1024
FLOAT32_MAX = 3.4028235e38
ZERO_SHA256 = "0" * 64

_SAFE_MESSAGES = {
    "manifest_unreadable": "无法读取研究工件清单。",
    "manifest_invalid": "研究工件清单不符合版本化合同。",
    "approval_required": "研究工件缺少包外可信审批。",
    "approval_invalid": "研究工件审批回执无效、过期或不匹配。",
    "package_inventory_invalid": "研究工件包包含未获准的文件或目录。",
    "package_file_invalid": "研究工件包文件类型或路径无效。",
    "package_file_too_large": "研究工件包文件超过允许大小。",
    "artifact_unreadable": "无法读取研究模型工件。",
    "artifact_digest_mismatch": "研究模型工件完整性校验失败。",
    "model_card_unreadable": "无法读取研究模型卡。",
    "model_card_digest_mismatch": "研究模型卡完整性校验失败。",
    "model_card_incomplete": "研究模型卡仍含未完成占位内容。",
    "runtime_unavailable": "当前环境未安装锁定版本的 PyTorch 运行时。",
    "runtime_load_failed": "研究模型运行时加载失败。",
    "runtime_inference_failed": "研究模型运行时推理失败。",
    "self_test_failed": "研究模型适配器黄金向量自检失败。",
}


class ResearchArtifactError(RuntimeError):
    """Stable, non-path-leaking error raised by the research package gate."""

    def __init__(self, code: str) -> None:
        if code not in _SAFE_MESSAGES:
            raise ValueError("unsupported research artifact error code")
        self.code = code
        super().__init__(_SAFE_MESSAGES[code])


class VectorRuntime(Protocol):
    def infer(self, values: Sequence[float]) -> Sequence[float]: ...

    def integrity_ok(self) -> bool: ...


class ApprovalPolicy(Protocol):
    """Trust boundary implemented by an access-controlled external system."""

    def authorize(self, request: "ApprovalRequest") -> "ApprovalReceipt | None": ...


def _invalid() -> ResearchArtifactError:
    return ResearchArtifactError("manifest_invalid")


def _object(value: object, expected_keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise _invalid()
    return value


def _string(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid()
    return value.strip()


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _string(value)


def _identifier(value: object) -> str:
    identifier = _string(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", identifier):
        raise _invalid()
    return identifier


def _single_line_text(value: object) -> str:
    text = _string(value)
    if len(text) > 256 or any(ord(character) < 32 for character in text):
        raise _invalid()
    return text


def _exact_int(value: object, expected: int) -> int:
    if type(value) is not int or value != expected:
        raise _invalid()
    return value


def _number(value: object, *, positive: bool = False) -> float:
    if type(value) not in {int, float}:
        raise _invalid()
    converted = float(value)
    if not math.isfinite(converted) or (positive and converted <= 0.0):
        raise _invalid()
    return converted


def _numeric_array(
    value: object,
    length: int,
    *,
    positive: bool = False,
    bounded_score: bool = False,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise _invalid()
    converted = tuple(_number(item, positive=positive) for item in value)
    if bounded_score and any(not 0.0 <= item <= 1.0 for item in converted):
        raise _invalid()
    if any(abs(item) > FLOAT32_MAX for item in converted):
        raise _invalid()
    return converted


def _sha256(value: object) -> str:
    digest = _string(value).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise _invalid()
    return digest


def _safe_filename(value: object) -> str:
    filename = _string(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", filename):
        raise _invalid()
    if filename in {".", ".."}:
        raise _invalid()
    return filename


def _canonical_sha256(payload: object) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise _invalid() from None
    return hashlib.sha256(encoded).hexdigest()


def preprocessing_sha256(
    means: Sequence[float], stds: Sequence[float]
) -> str:
    return _canonical_sha256(
        {
            "kind": "standardize",
            "mean": [float(value) for value in means],
            "std": [float(value) for value in stds],
        }
    )


@dataclass(frozen=True)
class GoldenCase:
    features: tuple[float, ...]
    expected_scores: tuple[float, float]


def golden_cases_sha256(
    tolerance: float, cases: Sequence[GoldenCase]
) -> str:
    return _canonical_sha256(
        {
            "tolerance": float(tolerance),
            "cases": [
                {
                    "features": list(case.features),
                    "expected_scores": list(case.expected_scores),
                }
                for case in cases
            ],
        }
    )


@dataclass(frozen=True)
class ApprovalRequest:
    reference_id: str
    requested_scope: str
    model_id: str
    source_commit: str
    artifact_sha256: str
    preprocessing_sha256: str
    model_card_sha256: str
    golden_cases_sha256: str
    manifest_sha256: str
    loader_contract_version: str

    @property
    def request_sha256(self) -> str:
        return _canonical_sha256(
            {
                "reference_id": self.reference_id,
                "requested_scope": self.requested_scope,
                "model_id": self.model_id,
                "source_commit": self.source_commit,
                "artifact_sha256": self.artifact_sha256,
                "preprocessing_sha256": self.preprocessing_sha256,
                "model_card_sha256": self.model_card_sha256,
                "golden_cases_sha256": self.golden_cases_sha256,
                "manifest_sha256": self.manifest_sha256,
                "loader_contract_version": self.loader_contract_version,
            }
        )


@dataclass(frozen=True)
class ApprovalReceipt:
    receipt_id: str
    request_sha256: str
    approved_scope: str
    issued_at_utc: datetime
    expires_at_utc: datetime
    revoked: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ApprovalReceipt":
        try:
            receipt = _object(
                raw,
                {
                    "schema_version",
                    "receipt_id",
                    "request_sha256",
                    "approved_scope",
                    "issued_at_utc",
                    "expires_at_utc",
                    "revoked",
                },
            )
            _exact_int(receipt["schema_version"], 1)
            receipt_id = _string(receipt["receipt_id"])
            if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", receipt_id):
                raise _invalid()
            request_digest = _sha256(receipt["request_sha256"])
            scope = _string(receipt["approved_scope"])
            issued_at = _parse_utc(receipt["issued_at_utc"])
            expires_at = _parse_utc(receipt["expires_at_utc"])
            if type(receipt["revoked"]) is not bool:
                raise _invalid()
            return cls(
                receipt_id=receipt_id,
                request_sha256=request_digest,
                approved_scope=scope,
                issued_at_utc=issued_at,
                expires_at_utc=expires_at,
                revoked=receipt["revoked"],
            )
        except ResearchArtifactError:
            raise ResearchArtifactError("approval_invalid") from None


class ExternalJsonApprovalPolicy:
    """Read one receipt from a protected path outside the research package.

    Filesystem access control is the trust assumption.  This class does not
    claim that an unsigned JSON receipt authenticates an institutional issuer.
    """

    def __init__(self, receipt_path: str | Path) -> None:
        self.receipt_path = Path(receipt_path).resolve()

    def authorize(self, request: ApprovalRequest) -> ApprovalReceipt | None:
        try:
            raw_bytes = _read_regular_file(
                self.receipt_path,
                MANIFEST_MAX_BYTES,
                unreadable_code="approval_invalid",
            )
            raw = json.loads(raw_bytes.decode("utf-8"))
            if not isinstance(raw, Mapping):
                raise ResearchArtifactError("approval_invalid")
            return ApprovalReceipt.from_mapping(raw)
        except (
            OSError,
            UnicodeError,
            ValueError,
            RecursionError,
            json.JSONDecodeError,
            ResearchArtifactError,
        ):
            raise ResearchArtifactError("approval_invalid") from None


@dataclass(frozen=True)
class ResearchManifest:
    model_id: str
    training_data: str
    architecture: str
    adapter_kind: str
    adapter_version: str
    framework: str
    framework_version: str
    dtype: str
    feature_contract_version: str
    means: tuple[float, ...]
    stds: tuple[float, ...]
    preprocessing_sha256: str
    output_contract_version: str
    artifact_filename: str
    artifact_sha256: str
    source_commit: str
    model_card: str
    model_card_sha256: str
    authorization_reference: str | None
    requested_scope: str
    self_test_status: str
    self_test_tolerance: float
    golden_cases: tuple[GoldenCase, ...]
    golden_cases_sha256: str
    manifest_sha256: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ResearchManifest":
        root = _object(
            raw,
            {
                "schema_version",
                "model",
                "adapter",
                "feature_contract",
                "preprocessing",
                "output_contract",
                "artifact",
                "provenance",
                "authorization_request",
                "self_test",
            },
        )
        _exact_int(root["schema_version"], 1)

        model = _object(
            root["model"],
            {
                "model_id",
                "model_stage",
                "training_data",
                "architecture",
                "clinical_use",
            },
        )
        if model["model_stage"] != "research_locked":
            raise _invalid()
        if model["clinical_use"] is not False:
            raise _invalid()
        architecture = _string(model["architecture"])
        if architecture != SUPPORTED_ARCHITECTURE:
            raise _invalid()

        adapter = _object(
            root["adapter"],
            {
                "kind",
                "version",
                "framework",
                "framework_version",
                "device",
                "dtype",
                "input_shape",
                "output_shape",
            },
        )
        expected_strings = {
            "kind": "pytorch_state_dict",
            "version": "1",
            "framework": "pytorch",
            "framework_version": SUPPORTED_TORCH_VERSION,
            "device": "cpu",
            "dtype": "float32",
        }
        if any(
            not isinstance(adapter[field], str)
            or adapter[field] != expected
            for field, expected in expected_strings.items()
        ):
            raise _invalid()
        input_shape = adapter["input_shape"]
        output_shape = adapter["output_shape"]
        if (
            not isinstance(input_shape, list)
            or len(input_shape) != 2
            or type(input_shape[0]) is not int
            or type(input_shape[1]) is not int
            or input_shape != [1, len(FEATURE_ORDER)]
            or not isinstance(output_shape, list)
            or len(output_shape) != 2
            or type(output_shape[0]) is not int
            or type(output_shape[1]) is not int
            or output_shape != [1, 2]
        ):
            raise _invalid()

        feature_contract = _object(
            root["feature_contract"], {"version", "order", "followup_months"}
        )
        if feature_contract["version"] != FEATURE_CONTRACT_VERSION:
            raise _invalid()
        raw_order = feature_contract["order"]
        if (
            not isinstance(raw_order, list)
            or any(not isinstance(item, str) for item in raw_order)
            or tuple(raw_order) != FEATURE_ORDER
        ):
            raise _invalid()
        _exact_int(feature_contract["followup_months"], 12)

        preprocessing = _object(
            root["preprocessing"], {"kind", "mean", "std", "sha256"}
        )
        if preprocessing["kind"] != "standardize":
            raise _invalid()
        means = _numeric_array(preprocessing["mean"], len(FEATURE_ORDER))
        stds = _numeric_array(
            preprocessing["std"], len(FEATURE_ORDER), positive=True
        )
        preprocessing_digest = preprocessing_sha256(means, stds)
        if _sha256(preprocessing["sha256"]) != preprocessing_digest:
            raise _invalid()

        output_contract = _object(
            root["output_contract"], {"version", "order", "kind"}
        )
        if (
            output_contract["version"] != OUTPUT_CONTRACT_VERSION
            or output_contract["order"] != ["od", "os"]
            or any(
                not isinstance(item, str)
                for item in output_contract.get("order", [])
            )
            or output_contract["kind"] != "logits_sigmoid"
        ):
            raise _invalid()

        artifact = _object(root["artifact"], {"filename", "sha256"})
        provenance = _object(
            root["provenance"],
            {"source_commit", "model_card", "model_card_sha256"},
        )
        source_commit = _string(provenance["source_commit"]).lower()
        if not re.fullmatch(r"[0-9a-f]{40,64}", source_commit):
            raise _invalid()

        authorization = _object(
            root["authorization_request"], {"reference_id", "requested_scope"}
        )
        reference_id = _optional_string(authorization["reference_id"])
        requested_scope = _string(authorization["requested_scope"])
        if requested_scope != SUPPORTED_APPROVAL_SCOPE:
            raise _invalid()

        self_test = _object(
            root["self_test"], {"status", "tolerance", "cases", "sha256"}
        )
        status_value = _string(self_test["status"])
        tolerance = _number(self_test["tolerance"], positive=True)
        if tolerance > 0.001 or not isinstance(self_test["cases"], list):
            raise _invalid()
        raw_cases = self_test["cases"]
        golden_cases: tuple[GoldenCase, ...]
        golden_digest = _sha256(self_test["sha256"])
        if status_value == "not_exported":
            if raw_cases or golden_digest != ZERO_SHA256:
                raise _invalid()
            golden_cases = ()
        elif status_value == "exported":
            parsed_cases: list[GoldenCase] = []
            for raw_case in raw_cases:
                case = _object(raw_case, {"features", "expected_scores"})
                features = _numeric_array(
                    case["features"], len(FEATURE_ORDER)
                )
                scores = _numeric_array(
                    case["expected_scores"], 2, bounded_score=True
                )
                parsed_cases.append(
                    GoldenCase(features=features, expected_scores=(scores[0], scores[1]))
                )
            golden_cases = tuple(parsed_cases)
            _validate_golden_coverage(golden_cases, tolerance)
            if golden_cases_sha256(tolerance, golden_cases) != golden_digest:
                raise _invalid()
        else:
            raise _invalid()

        artifact_filename = _safe_filename(artifact["filename"])
        model_card = _safe_filename(provenance["model_card"])
        if len({"manifest.json", artifact_filename, model_card}) != 3:
            raise _invalid()

        return cls(
            model_id=_identifier(model["model_id"]),
            training_data=_single_line_text(model["training_data"]),
            architecture=architecture,
            adapter_kind=expected_strings["kind"],
            adapter_version=expected_strings["version"],
            framework=expected_strings["framework"],
            framework_version=expected_strings["framework_version"],
            dtype=expected_strings["dtype"],
            feature_contract_version=_string(feature_contract["version"]),
            means=means,
            stds=stds,
            preprocessing_sha256=preprocessing_digest,
            output_contract_version=_string(output_contract["version"]),
            artifact_filename=artifact_filename,
            artifact_sha256=_sha256(artifact["sha256"]),
            source_commit=source_commit,
            model_card=model_card,
            model_card_sha256=_sha256(provenance["model_card_sha256"]),
            authorization_reference=(
                _identifier(reference_id) if reference_id is not None else None
            ),
            requested_scope=requested_scope,
            self_test_status=status_value,
            self_test_tolerance=tolerance,
            golden_cases=golden_cases,
            golden_cases_sha256=golden_digest,
            manifest_sha256=_canonical_sha256(raw),
        )

    @classmethod
    def from_bytes(cls, raw_bytes: bytes) -> "ResearchManifest":
        try:
            raw = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeError, ValueError, RecursionError, json.JSONDecodeError):
            raise ResearchArtifactError("manifest_unreadable") from None
        if not isinstance(raw, Mapping):
            raise _invalid()
        return cls.from_mapping(raw)

    @classmethod
    def from_path(cls, path: str | Path) -> "ResearchManifest":
        raw_bytes = _read_regular_file(
            Path(path), MANIFEST_MAX_BYTES, unreadable_code="manifest_unreadable"
        )
        return cls.from_bytes(raw_bytes)

    def approval_request(self) -> ApprovalRequest:
        if (
            not self.authorization_reference
            or self.authorization_reference.startswith("NOT_")
            or self.model_id.startswith("REPLACE_")
            or set(self.source_commit) == {"0"}
            or self.artifact_sha256 == ZERO_SHA256
            or self.model_card_sha256 == ZERO_SHA256
            or self.self_test_status != "exported"
            or self.golden_cases_sha256 == ZERO_SHA256
        ):
            raise ResearchArtifactError("approval_required")
        return ApprovalRequest(
            reference_id=self.authorization_reference,
            requested_scope=self.requested_scope,
            model_id=self.model_id,
            source_commit=self.source_commit,
            artifact_sha256=self.artifact_sha256,
            preprocessing_sha256=self.preprocessing_sha256,
            model_card_sha256=self.model_card_sha256,
            golden_cases_sha256=self.golden_cases_sha256,
            manifest_sha256=self.manifest_sha256,
            loader_contract_version=RESEARCH_LOADER_CONTRACT_VERSION,
        )


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise _invalid()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise _invalid() from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _invalid()
    return parsed.astimezone(timezone.utc)


def _validate_golden_coverage(
    cases: Sequence[GoldenCase], tolerance: float
) -> None:
    if len(cases) < 3 or len({case.features for case in cases}) != len(cases):
        raise _invalid()
    if not any(any(value != 0.0 for value in case.features) for case in cases):
        raise _invalid()
    od_index = FEATURE_ORDER.index("axial_length_od_delta_mm")
    os_index = FEATURE_ORDER.index("axial_length_os_delta_mm")
    if not any(case.features[od_index] != case.features[os_index] for case in cases):
        raise _invalid()
    if not any(
        abs(case.expected_scores[0] - case.expected_scores[1]) > tolerance
        for case in cases
    ):
        raise _invalid()
    if not all(
        max(case.expected_scores[index] for case in cases)
        - min(case.expected_scores[index] for case in cases)
        > tolerance
        for index in range(2)
    ):
        raise _invalid()


def _read_regular_file(
    path: Path,
    max_bytes: int,
    *,
    unreadable_code: str,
) -> bytes:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ResearchArtifactError("package_file_invalid")
        if metadata.st_size <= 0:
            raise ResearchArtifactError(unreadable_code)
        if metadata.st_size > max_bytes:
            raise ResearchArtifactError("package_file_too_large")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_size != metadata.st_size
                or (opened.st_dev, opened.st_ino)
                != (metadata.st_dev, metadata.st_ino)
            ):
                raise ResearchArtifactError("package_file_invalid")
            content = handle.read(max_bytes + 1)
        if len(content) != metadata.st_size or len(content) > max_bytes:
            raise ResearchArtifactError("package_file_too_large")
        return content
    except ResearchArtifactError:
        raise
    except OSError:
        raise ResearchArtifactError(unreadable_code) from None


def _validate_receipt(
    request: ApprovalRequest,
    receipt: ApprovalReceipt | None,
    *,
    now_utc: datetime | None = None,
) -> ApprovalReceipt:
    if receipt is None:
        raise ResearchArtifactError("approval_required")
    if not isinstance(receipt, ApprovalReceipt):
        raise ResearchArtifactError("approval_invalid")
    if (
        not isinstance(receipt.receipt_id, str)
        or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", receipt.receipt_id)
        or not isinstance(receipt.request_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", receipt.request_sha256)
        or not isinstance(receipt.approved_scope, str)
        or not isinstance(receipt.issued_at_utc, datetime)
        or not isinstance(receipt.expires_at_utc, datetime)
        or type(receipt.revoked) is not bool
        or receipt.issued_at_utc.tzinfo is None
        or receipt.issued_at_utc.utcoffset() is None
        or receipt.expires_at_utc.tzinfo is None
        or receipt.expires_at_utc.utcoffset() is None
    ):
        raise ResearchArtifactError("approval_invalid")
    now = now_utc or datetime.now(timezone.utc)
    if (
        not isinstance(now, datetime)
        or now.tzinfo is None
        or now.utcoffset() is None
    ):
        raise ResearchArtifactError("approval_invalid")
    now = now.astimezone(timezone.utc)
    if (
        receipt.revoked
        or receipt.request_sha256 != request.request_sha256
        or receipt.approved_scope != request.requested_scope
        or receipt.issued_at_utc > now
        or receipt.expires_at_utc <= now
        or receipt.expires_at_utc <= receipt.issued_at_utc
    ):
        raise ResearchArtifactError("approval_invalid")
    return receipt


def _validate_package_inventory(root: Path, expected_names: set[str]) -> None:
    try:
        inventory: dict[str, tuple[bool, bool]] = {}
        with os.scandir(root) as entries:
            for entry in entries:
                if len(inventory) >= len(expected_names):
                    raise ResearchArtifactError("package_inventory_invalid")
                inventory[entry.name] = (
                    entry.is_symlink(),
                    entry.is_file(follow_symlinks=False),
                )
    except ResearchArtifactError:
        raise
    except OSError:
        raise ResearchArtifactError("package_inventory_invalid") from None
    if set(inventory) != expected_names or any(
        is_symlink or not is_file for is_symlink, is_file in inventory.values()
    ):
        raise ResearchArtifactError("package_inventory_invalid")


@dataclass(frozen=True)
class _VerifiedResearchPackage:
    manifest: ResearchManifest
    artifact_bytes: bytes
    approval_receipt: ApprovalReceipt
    approval_request: ApprovalRequest
    approval_policy: ApprovalPolicy


def _verify_research_package(
    package_dir: str | Path,
    approval_policy: ApprovalPolicy,
) -> _VerifiedResearchPackage:
    try:
        supplied_root = Path(package_dir)
        root_metadata = supplied_root.lstat()
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
            root_metadata.st_mode
        ):
            raise ResearchArtifactError("package_file_invalid")
        root = supplied_root.resolve(strict=True)
    except ResearchArtifactError:
        raise
    except OSError:
        raise ResearchArtifactError("manifest_unreadable") from None

    manifest_bytes = _read_regular_file(
        root / "manifest.json",
        MANIFEST_MAX_BYTES,
        unreadable_code="manifest_unreadable",
    )
    manifest = ResearchManifest.from_bytes(manifest_bytes)
    request = manifest.approval_request()
    try:
        receipt = approval_policy.authorize(request)
    except ResearchArtifactError:
        raise
    except Exception:
        raise ResearchArtifactError("approval_invalid") from None
    receipt = _validate_receipt(request, receipt)

    if isinstance(approval_policy, ExternalJsonApprovalPolicy):
        try:
            approval_policy.receipt_path.relative_to(root)
        except ValueError:
            pass
        else:
            raise ResearchArtifactError("approval_invalid")

    expected_names = {
        "manifest.json",
        manifest.model_card,
        manifest.artifact_filename,
    }
    _validate_package_inventory(root, expected_names)

    model_card_bytes = _read_regular_file(
        root / manifest.model_card,
        MODEL_CARD_MAX_BYTES,
        unreadable_code="model_card_unreadable",
    )
    if hashlib.sha256(model_card_bytes).hexdigest() != manifest.model_card_sha256:
        raise ResearchArtifactError("model_card_digest_mismatch")
    _validate_model_card(model_card_bytes, manifest.model_id)

    artifact_bytes = _read_regular_file(
        root / manifest.artifact_filename,
        ARTIFACT_MAX_BYTES,
        unreadable_code="artifact_unreadable",
    )
    if hashlib.sha256(artifact_bytes).hexdigest() != manifest.artifact_sha256:
        raise ResearchArtifactError("artifact_digest_mismatch")
    _validate_package_inventory(root, expected_names)

    return _VerifiedResearchPackage(
        manifest=manifest,
        artifact_bytes=artifact_bytes,
        approval_receipt=receipt,
        approval_request=request,
        approval_policy=approval_policy,
    )


def _validate_model_card(content: bytes, model_id: str) -> None:
    try:
        text = content.decode("utf-8").strip()
    except UnicodeError:
        raise ResearchArtifactError("model_card_incomplete") from None
    lowered = text.casefold()
    forbidden_markers = (
        "<required",
        "not_reviewed",
        "未填写",
    )
    if (
        len(text) < 20
        or not re.search(
            rf"^model_id:\s*{re.escape(model_id)}\s*$", text, flags=re.MULTILINE
        )
        or not re.search(
            r"^clinical_use:\s*false\s*$", text, flags=re.MULTILINE
        )
        or any(marker in lowered for marker in forbidden_markers)
    ):
        raise ResearchArtifactError("model_card_incomplete")


class _TorchStateDictRuntime:
    """Internal runtime constructed only from a package verified in this module."""

    def __init__(self, torch_module: Any, model: Any, feature_count: int) -> None:
        self._torch = torch_module
        self._model = model
        self._feature_count = feature_count
        self._parameter_sha256 = self._current_parameter_sha256()

    @classmethod
    def _from_verified_package(
        cls,
        verified: _VerifiedResearchPackage,
        *,
        _token: object,
    ) -> "_TorchStateDictRuntime":
        if _token is not _CONSTRUCTION_TOKEN:
            raise TypeError("research runtime requires a verified package")
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            raise ResearchArtifactError("runtime_unavailable") from None

        try:
            installed_version = str(torch.__version__).split("+", 1)[0]
            if installed_version != SUPPORTED_TORCH_VERSION:
                raise ValueError("unsupported torch version")
            state_dict = torch.load(
                io.BytesIO(verified.artifact_bytes),
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(state_dict, Mapping) or set(state_dict) != {
                "weight",
                "bias",
            }:
                raise ValueError("unexpected state dict keys")
            weight = state_dict["weight"]
            bias = state_dict["bias"]
            feature_count = len(FEATURE_ORDER)
            if tuple(weight.shape) != (2, feature_count) or tuple(bias.shape) != (2,):
                raise ValueError("unexpected state dict shape")
            if weight.dtype != torch.float32 or bias.dtype != torch.float32:
                raise ValueError("unexpected state dict dtype")
            if not bool(torch.isfinite(weight).all()) or not bool(
                torch.isfinite(bias).all()
            ):
                raise ValueError("non-finite state dict")

            model = torch.nn.Linear(feature_count, 2, bias=True)
            model.load_state_dict(state_dict, strict=True)
            model.to(device="cpu", dtype=torch.float32)
            model.requires_grad_(False)
            model.eval()
            return cls(torch, model, feature_count)
        except Exception:
            raise ResearchArtifactError("runtime_load_failed") from None

    def _current_parameter_sha256(self) -> str:
        raw_weight = self._model.weight
        raw_bias = self._model.bias
        if (
            self._model.training
            or raw_weight.device.type != "cpu"
            or raw_bias.device.type != "cpu"
            or any(parameter.requires_grad for parameter in self._model.parameters())
        ):
            raise ValueError("runtime state changed")
        weight = raw_weight.detach().cpu()
        bias = raw_bias.detach().cpu()
        if (
            tuple(weight.shape) != (2, self._feature_count)
            or tuple(bias.shape) != (2,)
            or weight.dtype != self._torch.float32
            or bias.dtype != self._torch.float32
            or not bool(self._torch.isfinite(weight).all())
            or not bool(self._torch.isfinite(bias).all())
        ):
            raise ValueError("runtime parameters changed")
        values = weight.reshape(-1).tolist() + bias.reshape(-1).tolist()
        packed = struct.pack(f"<{len(values)}f", *values)
        return hashlib.sha256(packed).hexdigest()

    def integrity_ok(self) -> bool:
        try:
            return self._current_parameter_sha256() == self._parameter_sha256
        except Exception:
            return False

    def infer(self, values: Sequence[float]) -> Sequence[float]:
        try:
            if not self.integrity_ok():
                raise ValueError("runtime parameters changed")
            if isinstance(values, (str, bytes)):
                raise ValueError("invalid runtime input")
            converted = tuple(float(value) for value in values)
            if len(converted) != self._feature_count or any(
                not math.isfinite(value) or abs(value) > FLOAT32_MAX
                for value in converted
            ):
                raise ValueError("invalid runtime input")
            tensor = self._torch.tensor(
                [list(converted)], dtype=self._torch.float32, device="cpu"
            )
            with self._torch.inference_mode():
                output = self._torch.sigmoid(self._model(tensor))
            if tuple(output.shape) != (1, 2):
                raise ValueError("unexpected runtime output shape")
            return output.detach().cpu().reshape(-1).tolist()
        except Exception:
            raise ResearchArtifactError("runtime_inference_failed") from None


_CONSTRUCTION_TOKEN = object()


class ResearchModelAdapter:
    """Offline backend created only from a verified package and locked runtime."""

    def __init__(
        self,
        verified: _VerifiedResearchPackage,
        runtime: VectorRuntime,
        *,
        _token: object,
    ) -> None:
        if _token is not _CONSTRUCTION_TOKEN:
            raise TypeError("use ResearchModelAdapter.from_package")
        manifest = verified.manifest
        approval_request = verified.approval_request
        approval_receipt = verified.approval_receipt
        self._manifest = manifest
        self._runtime = runtime
        self._approval_request = approval_request
        self._approval_receipt = approval_receipt
        self._approval_policy = verified.approval_policy
        self.metadata = MappingProxyType(
            {
                "model_id": manifest.model_id,
                "model_stage": "research_locked",
                "training_data": manifest.training_data,
                "clinical_use": False,
                "adapter_kind": manifest.adapter_kind,
                "adapter_version": manifest.adapter_version,
                "feature_contract_version": manifest.feature_contract_version,
                "output_contract_version": manifest.output_contract_version,
                "framework": manifest.framework,
                "framework_version": manifest.framework_version,
                "artifact_sha256": manifest.artifact_sha256,
                "preprocessing_sha256": manifest.preprocessing_sha256,
                "model_card_sha256": manifest.model_card_sha256,
                "golden_cases_sha256": manifest.golden_cases_sha256,
                "source_commit": manifest.source_commit,
                "approval_request_sha256": approval_request.request_sha256,
                "manifest_sha256": manifest.manifest_sha256,
                "loader_contract_version": RESEARCH_LOADER_CONTRACT_VERSION,
                "approval_receipt_id": approval_receipt.receipt_id,
                "approval_scope": approval_receipt.approved_scope,
                "approval_issued_at_utc": approval_receipt.issued_at_utc.isoformat(),
                "approval_expires_at_utc": approval_receipt.expires_at_utc.isoformat(),
            }
        )
        self._readiness = AdapterReadiness(status="not_ready", self_test="not_run")
        self._run_self_test()

    def _approval_valid(self, *, refresh_policy: bool) -> bool:
        try:
            receipt = self._approval_receipt
            if refresh_policy:
                receipt = self._approval_policy.authorize(self._approval_request)
                receipt = _validate_receipt(self._approval_request, receipt)
                if receipt != self._approval_receipt:
                    return False
            else:
                _validate_receipt(self._approval_request, receipt)
            return True
        except Exception:
            return False

    @classmethod
    def from_package(
        cls,
        package_dir: str | Path,
        *,
        approval_policy: ApprovalPolicy,
    ) -> "ResearchModelAdapter":
        verified = _verify_research_package(package_dir, approval_policy)
        runtime = _TorchStateDictRuntime._from_verified_package(
            verified, _token=_CONSTRUCTION_TOKEN
        )
        return cls(verified, runtime, _token=_CONSTRUCTION_TOKEN)

    def _run_self_test(self) -> None:
        try:
            for case in self._manifest.golden_cases:
                features = dict(zip(FEATURE_ORDER, case.features, strict=True))
                first = self.predict(features)
                second = self.predict(features)
                if first != second:
                    raise ValueError("non-deterministic runtime")
                for name, expected in zip(
                    ("od", "os"), case.expected_scores, strict=True
                ):
                    if abs(first[name] - expected) > self._manifest.self_test_tolerance:
                        raise ValueError("golden output mismatch")
        except Exception:
            self._readiness = AdapterReadiness(
                status="not_ready", self_test="failed", error_code="self_test_failed"
            )
            raise ResearchArtifactError("self_test_failed") from None
        self._readiness = AdapterReadiness(status="ready", self_test="passed")

    def predict(self, features: Mapping[str, float]) -> dict[str, float]:
        if self._readiness.status == "not_ready" and self._readiness.self_test == "failed":
            raise ResearchArtifactError(
                self._readiness.error_code or "runtime_inference_failed"
            )
        if not self._approval_valid(refresh_policy=False):
            raise ResearchArtifactError("approval_invalid")
        values = validated_feature_values(features)
        standardized = tuple(
            (value - mean) / std
            for value, mean, std in zip(
                values, self._manifest.means, self._manifest.stds, strict=True
            )
        )
        if any(
            not math.isfinite(value) or abs(value) > FLOAT32_MAX
            for value in standardized
        ):
            raise ResearchArtifactError("runtime_inference_failed")
        try:
            if not self._runtime.integrity_ok():
                raise ResearchArtifactError("runtime_inference_failed")
            return validated_runtime_output(self._runtime.infer(standardized))
        except ResearchArtifactError:
            raise
        except (ModelContractError, TypeError, ValueError):
            raise ResearchArtifactError("runtime_inference_failed") from None

    def readiness(self) -> AdapterReadiness:
        if self._readiness.status == "ready":
            if not self._approval_valid(refresh_policy=True):
                self._readiness = AdapterReadiness(
                    status="not_ready",
                    self_test="failed",
                    error_code="approval_invalid",
                )
                return self._readiness
            try:
                runtime_intact = self._runtime.integrity_ok()
            except Exception:
                runtime_intact = False
            if not runtime_intact:
                self._readiness = AdapterReadiness(
                    status="not_ready",
                    self_test="failed",
                    error_code="runtime_inference_failed",
                )
        return self._readiness
