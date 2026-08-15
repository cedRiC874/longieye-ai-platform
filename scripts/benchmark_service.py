"""Benchmark core and in-process ASGI inference on one deterministic request."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("LONGIEYE_LOG_LEVEL", "WARNING")

import httpx  # noqa: E402
import psutil  # noqa: E402

from app.main import SERVICE, app  # noqa: E402
from longieye.benchmarking import summarize_latencies  # noqa: E402
from longieye.domain import LongitudinalCase, VisitMeasurements  # noqa: E402


def load_payload() -> dict[str, object]:
    return json.loads((PROJECT_ROOT / "examples" / "request.json").read_text())


def payload_to_case(payload: dict[str, object]) -> LongitudinalCase:
    return LongitudinalCase(
        y1=VisitMeasurements(**payload["y1"]),
        y2=VisitMeasurements(**payload["y2"]),
        followup_months=int(payload["followup_months"]),
    )


def memory_summary(
    callback, iterations: int
) -> dict[str, float | int]:
    """Measure memory in a separate loop so tracing does not bias latency."""

    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    peak_rss = baseline_rss
    tracemalloc.start()
    for index in range(iterations):
        callback()
        if index % 10 == 0 or index == iterations - 1:
            peak_rss = max(peak_rss, process.memory_info().rss)
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "memory_iterations": iterations,
        "rss_baseline_mb": round(baseline_rss / 1_048_576.0, 6),
        "rss_peak_mb": round(peak_rss / 1_048_576.0, 6),
        "rss_peak_delta_mb": round((peak_rss - baseline_rss) / 1_048_576.0, 6),
        "python_tracemalloc_peak_mb": round(traced_peak / 1_048_576.0, 6),
    }


def benchmark_core(
    case: LongitudinalCase, warmup: int, iterations: int, memory_iterations: int
) -> dict[str, float | int | str]:
    for _ in range(warmup):
        SERVICE.predict(case, case_id="benchmark")

    latencies_ms: list[float] = []
    total_started = time.perf_counter()
    for _ in range(iterations):
        started = time.perf_counter_ns()
        SERVICE.predict(case, case_id="benchmark")
        latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    total_seconds = time.perf_counter() - total_started

    summary = summarize_latencies(latencies_ms, total_seconds)
    summary.update(
        {
            "mode": "core_service",
            "warmup_iterations": warmup,
            "errors": 0,
        }
    )
    summary.update(
        memory_summary(
            lambda: SERVICE.predict(case, case_id="benchmark"), memory_iterations
        )
    )
    return summary


async def benchmark_api(
    payload: dict[str, object],
    warmup: int,
    iterations: int,
    memory_iterations: int,
) -> dict[str, float | int | str]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://benchmark"
    ) as client:
        for _ in range(warmup):
            response = await client.post("/predict", json=payload)
            response.raise_for_status()

        latencies_ms: list[float] = []
        total_started = time.perf_counter()
        for _ in range(iterations):
            started = time.perf_counter_ns()
            response = await client.post("/predict", json=payload)
            response.raise_for_status()
            latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        total_seconds = time.perf_counter() - total_started

        process = psutil.Process()
        baseline_rss = process.memory_info().rss
        peak_rss = baseline_rss
        tracemalloc.start()
        for index in range(memory_iterations):
            response = await client.post("/predict", json=payload)
            response.raise_for_status()
            if index % 10 == 0 or index == memory_iterations - 1:
                peak_rss = max(peak_rss, process.memory_info().rss)
        _, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    summary = summarize_latencies(latencies_ms, total_seconds)
    summary.update(
        {
            "mode": "in_process_asgi",
            "warmup_iterations": warmup,
            "errors": 0,
            "memory_iterations": memory_iterations,
            "rss_baseline_mb": round(baseline_rss / 1_048_576.0, 6),
            "rss_peak_mb": round(peak_rss / 1_048_576.0, 6),
            "rss_peak_delta_mb": round(
                (peak_rss - baseline_rss) / 1_048_576.0, 6
            ),
            "python_tracemalloc_peak_mb": round(
                traced_peak / 1_048_576.0, 6
            ),
        }
    )
    return summary


def markdown_report(report: dict[str, object]) -> str:
    rows = []
    for result in report["results"]:
        rows.append(
            "| {mode} | {iterations} | {p50_ms:.3f} | {p95_ms:.3f} | "
            "{p99_ms:.3f} | {throughput_requests_per_second:.1f} | "
            "{rss_peak_delta_mb:.3f} | {python_tracemalloc_peak_mb:.3f} |".format(
                **result
            )
        )
    return "\n".join(
        [
            "# LongiEye local benchmark",
            "",
            f"Generated: `{report['generated_at_utc']}`",
            f"Model: `{report['model_id']}`",
            f"Artifact SHA-256: `{report['model_artifact_sha256']}`",
            f"Python: `{report['environment']['python']}`",
            f"Platform: `{report['environment']['platform']}`",
            "",
            "| Mode | Iterations | P50 ms | P95 ms | P99 ms | Requests/s | RSS delta MB | Python peak MB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "`core_service` measures feature extraction, inference and response assembly.",
            "`in_process_asgi` additionally measures validation, middleware and JSON handling.",
            "Both modes run in one process and exclude network, proxy and container overhead.",
            "Timing runs with memory tracing disabled; memory is measured in a separate loop.",
            "RSS is sampled from the process and Python peak uses `tracemalloc`.",
            "These are local engineering measurements for a synthetic model, not clinical metrics.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--core-iterations", type=int, default=5000)
    parser.add_argument("--api-iterations", type=int, default=500)
    parser.add_argument("--memory-iterations", type=int, default=500)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "benchmarks"
    )
    args = parser.parse_args()
    if min(
        args.warmup,
        args.core_iterations,
        args.api_iterations,
        args.memory_iterations,
    ) < 1:
        raise ValueError("All iteration counts must be positive")

    payload = load_payload()
    case = payload_to_case(payload)
    model_path = PROJECT_ROOT / "configs" / "demo_model.json"
    report: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": SERVICE.model.metadata.get("model_id"),
        "model_artifact_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "model_stage": SERVICE.model.metadata.get("model_stage"),
        "clinical_use": False,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "logical_cpu_count": os.cpu_count(),
        },
        "measurement": {
            "transport": "httpx.ASGITransport",
            "sequential": True,
            "warmup_iterations": args.warmup,
            "core_iterations": args.core_iterations,
            "api_iterations": args.api_iterations,
            "memory_iterations": args.memory_iterations,
            "timing_memory_tracking_enabled": False,
            "percentile_method": "linear interpolation over sorted samples",
        },
        "results": [
            benchmark_core(
                case,
                args.warmup,
                args.core_iterations,
                args.memory_iterations,
            ),
            asyncio.run(
                benchmark_api(
                    payload,
                    args.warmup,
                    args.api_iterations,
                    args.memory_iterations,
                )
            ),
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "latest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (args.output_dir / "latest.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(markdown_report(report))


if __name__ == "__main__":
    main()
