"""Engineering-only comparison reports with separated evidence namespaces."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from types import MappingProxyType

from .benchmarking import summarize_latencies
from .model_contract import (
    FEATURE_CONTRACT_VERSION,
    OUTPUT_CONTRACT_VERSION,
    RiskModelBackend,
    validated_feature_values,
    validated_scores,
)


class AdapterComparisonError(ValueError):
    """Raised when two modes cannot be compared under one engineering contract."""


def _metadata_string(backend: RiskModelBackend, field: str) -> str:
    value = backend.metadata.get(field)
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}", value)
    ):
        raise AdapterComparisonError(f"backend metadata is missing {field}")
    return value


def benchmark_adapter(
    backend: RiskModelBackend,
    features: Mapping[str, float],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    """Measure an already-loaded adapter; never persist its score values."""

    if warmup < 1 or iterations < 1:
        raise AdapterComparisonError("warmup and iterations must be positive")
    ordered_features = validated_feature_values(features)
    before = dict(features)
    immutable_features = MappingProxyType(dict(features))

    baseline = validated_scores(backend.predict(immutable_features))
    repeated = validated_scores(backend.predict(immutable_features))
    if baseline != repeated:
        raise AdapterComparisonError("backend output is not deterministic")
    if dict(features) != before:
        raise AdapterComparisonError("backend mutated its input")

    for _ in range(warmup):
        validated_scores(backend.predict(immutable_features))

    latencies_ms: list[float] = []
    total_started = time.perf_counter()
    for _ in range(iterations):
        started = time.perf_counter_ns()
        validated_scores(backend.predict(immutable_features))
        latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    total_seconds = time.perf_counter() - total_started
    if dict(features) != before:
        raise AdapterComparisonError("backend mutated its input")

    result = {
        "model_id": _metadata_string(backend, "model_id"),
        "model_stage": _metadata_string(backend, "model_stage"),
        "adapter_kind": _metadata_string(backend, "adapter_kind"),
        "adapter_version": _metadata_string(backend, "adapter_version"),
        "feature_contract_version": _metadata_string(
            backend, "feature_contract_version"
        ),
        "output_contract_version": _metadata_string(
            backend, "output_contract_version"
        ),
        "contract_checks": {
            "exact_bilateral_outputs": True,
            "finite_bounded_scores": True,
            "deterministic_repeat": True,
            "input_unchanged": True,
        },
        "measurement": {
            "already_loaded": True,
            "sequential": True,
            "warmup_iterations": warmup,
            "input_vector_sha256": hashlib.sha256(
                json.dumps(
                    list(ordered_features), separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
            ).hexdigest(),
        },
        "runtime": summarize_latencies(latencies_ms, total_seconds),
    }
    for field in (
        "artifact_sha256",
        "preprocessing_sha256",
        "model_card_sha256",
        "golden_cases_sha256",
        "source_commit",
        "approval_request_sha256",
        "manifest_sha256",
        "loader_contract_version",
        "framework",
        "framework_version",
        "approval_receipt_id",
        "approval_scope",
        "approval_issued_at_utc",
        "approval_expires_at_utc",
    ):
        value = backend.metadata.get(field)
        if isinstance(value, str) and value:
            result[field] = value
    return result


def build_engineering_comparison(
    synthetic_backend: RiskModelBackend,
    research_backend: RiskModelBackend,
    features: Mapping[str, float],
    *,
    warmup: int = 20,
    iterations: int = 200,
) -> dict[str, object]:
    """Compare runtime contracts while explicitly excluding model-quality metrics."""

    if synthetic_backend.metadata.get("model_stage") != "demo_synthetic":
        raise AdapterComparisonError("synthetic namespace requires demo_synthetic")
    if research_backend.metadata.get("model_stage") != "research_locked":
        raise AdapterComparisonError("research namespace requires research_locked")
    for backend in (synthetic_backend, research_backend):
        if backend.metadata.get("clinical_use") is not False:
            raise AdapterComparisonError("comparison accepts non-clinical models only")
        if backend.metadata.get("feature_contract_version") != FEATURE_CONTRACT_VERSION:
            raise AdapterComparisonError("feature contracts do not match")
        if backend.metadata.get("output_contract_version") != OUTPUT_CONTRACT_VERSION:
            raise AdapterComparisonError("output contracts do not match")
        artifact_digest = backend.metadata.get("artifact_sha256")
        if not isinstance(artifact_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", artifact_digest
        ):
            raise AdapterComparisonError("comparison requires artifact digests")
        if backend.readiness().status != "ready":
            raise AdapterComparisonError("comparison requires ready backends")

    for field in (
        "preprocessing_sha256",
        "model_card_sha256",
        "golden_cases_sha256",
        "approval_request_sha256",
        "manifest_sha256",
    ):
        digest = research_backend.metadata.get(field)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise AdapterComparisonError(
                f"research backend metadata is missing valid {field}"
            )
    source_commit = research_backend.metadata.get("source_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40,64}", source_commit
    ):
        raise AdapterComparisonError("research backend is missing a source commit")
    _metadata_string(research_backend, "approval_receipt_id")
    _metadata_string(research_backend, "approval_scope")
    _metadata_string(research_backend, "approval_issued_at_utc")
    _metadata_string(research_backend, "approval_expires_at_utc")
    _metadata_string(research_backend, "loader_contract_version")
    _metadata_string(research_backend, "framework")
    _metadata_string(research_backend, "framework_version")

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "loaded_adapter_contract_and_runtime_only",
        "clinical_use": False,
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        },
        "modes": {
            "synthetic_demo": benchmark_adapter(
                synthetic_backend,
                features,
                warmup=warmup,
                iterations=iterations,
            ),
            "research_adapter": benchmark_adapter(
                research_backend,
                features,
                warmup=warmup,
                iterations=iterations,
            ),
        },
        "metric_namespaces": {
            "synthetic_sanity_metrics": {
                "included": False,
                "reason": "not part of adapter engineering comparison",
            },
            "authorized_research_metrics": {
                "included": False,
                "reason": "requires separately authorized aggregate evidence",
            },
        },
        "authorization_verification": {
            "performed_by_builder": False,
            "standard_cli_requires_verified_package": True,
            "reason": (
                "the generic builder records backend evidence but is not an "
                "authorization security boundary"
            ),
        },
        "stores_model_outputs": False,
    }


def comparison_markdown(report: Mapping[str, object]) -> str:
    """Render the engineering report without adding model-quality claims."""

    modes = report["modes"]
    rows: list[str] = []
    labels = {
        "synthetic_demo": "公开合成 JSON",
        "research_adapter": "研究合同后端（builder 不验证授权）",
    }
    for namespace in ("synthetic_demo", "research_adapter"):
        result = modes[namespace]
        runtime = result["runtime"]
        rows.append(
            "| {label} | `{model_id}` | `{adapter_kind}` | {iterations} | "
            "{p50_ms:.3f} | {p95_ms:.3f} | {p99_ms:.3f} |".format(
                label=labels[namespace],
                model_id=result["model_id"],
                adapter_kind=result["adapter_kind"],
                **runtime,
            )
        )
    return "\n".join(
        [
            "# LongiEye adapter 工程比较",
            "",
            f"生成时间：`{report['generated_at_utc']}`",
            f"Python：`{report['environment']['python']}`",
            f"平台：`{report['environment']['platform']}`",
            "",
            "本报告只比较已加载适配器的合同与本机顺序运行时间。",
            "通用 builder 不验证授权；标准 CLI 仅在研究包门禁通过后调用它。",
            "不保存模型输出，不包含或比较合成/研究 AUC 等模型质量指标。",
            "",
            "## 研究后端追溯记录",
            "",
            f"- Receipt ID：`{modes['research_adapter']['approval_receipt_id']}`",
            f"- Scope：`{modes['research_adapter']['approval_scope']}`",
            f"- 有效期：`{modes['research_adapter']['approval_issued_at_utc']}` 至 `{modes['research_adapter']['approval_expires_at_utc']}`",
            f"- Source commit：`{modes['research_adapter']['source_commit']}`",
            f"- Request SHA-256：`{modes['research_adapter']['approval_request_sha256']}`",
            f"- Manifest SHA-256：`{modes['research_adapter']['manifest_sha256']}`",
            f"- Artifact SHA-256：`{modes['research_adapter']['artifact_sha256']}`",
            "",
            "| 隔离命名空间 | 模型 | Adapter | 迭代 | P50 ms | P95 ms | P99 ms |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "`synthetic_sanity_metrics`：未包含。",
            "`authorized_research_metrics`：未包含；必须由另行授权的聚合证据提供。",
            "",
        ]
    )
