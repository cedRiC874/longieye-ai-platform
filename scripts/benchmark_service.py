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
    mode_labels = {
        "core_service": "核心服务（core_service）",
        "in_process_asgi": "进程内 ASGI（in_process_asgi）",
    }
    rows = []
    for result in report["results"]:
        rows.append(
            "| {mode_label} | {iterations} | {p50_ms:.3f} | {p95_ms:.3f} | "
            "{p99_ms:.3f} | {throughput_requests_per_second:.1f} | "
            "{rss_peak_delta_mb:.3f} | {python_tracemalloc_peak_mb:.3f} |".format(
                mode_label=mode_labels.get(result["mode"], result["mode"]),
                **result
            )
        )
    return "\n".join(
        [
            "# LongiEye 本机基准",
            "",
            f"生成时间：`{report['generated_at_utc']}`",
            f"模型：`{report['model_id']}`",
            f"模型制品 SHA-256：`{report['model_artifact_sha256']}`",
            f"Python: `{report['environment']['python']}`",
            f"运行平台：`{report['environment']['platform']}`",
            "",
            "| 测量路径 | 迭代次数 | P50 ms | P95 ms | P99 ms | 每秒请求数 | RSS 增量 MB | Python 峰值 MB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "`core_service` 测量特征提取、模型推理和响应组装。",
            "`in_process_asgi` 还包含请求校验、中间件和 JSON 处理。",
            "两种模式均在单进程内顺序运行，不包含网络、反向代理和容器开销。",
            "延迟计时阶段关闭内存追踪；内存数据在独立循环中测量。",
            "RSS 来自进程采样，Python 峰值使用 `tracemalloc` 测量。",
            "这些是合成模型的本机工程测量结果，不是临床性能指标。",
            "",
        ]
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="运行 LongiEye 本机工程基准。")
    parser.add_argument("--warmup", type=int, default=50, help="预热迭代次数。")
    parser.add_argument(
        "--core-iterations", type=int, default=5000, help="核心服务计时次数。"
    )
    parser.add_argument(
        "--api-iterations", type=int, default=500, help="进程内 API 计时次数。"
    )
    parser.add_argument(
        "--memory-iterations", type=int, default=500, help="内存测量迭代次数。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmarks",
        help="报告输出目录。",
    )
    args = parser.parse_args()
    if min(
        args.warmup,
        args.core_iterations,
        args.api_iterations,
        args.memory_iterations,
    ) < 1:
        raise ValueError("所有迭代次数都必须为正整数。")

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
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (args.output_dir / "latest.md").write_text(
        markdown_report(report), encoding="utf-8", newline="\n"
    )
    print(markdown_report(report))


if __name__ == "__main__":
    main()
