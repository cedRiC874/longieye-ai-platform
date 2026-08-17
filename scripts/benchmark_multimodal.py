"""Benchmark synthetic image fusion and exact structured fallback paths."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from longieye.benchmarking import summarize_latencies  # noqa: E402
from longieye import __version__  # noqa: E402
from longieye.features import extract_features  # noqa: E402
from longieye.fusion import (  # noqa: E402
    FUSION_CONTRACT_VERSION,
    IMAGE_EMBEDDING_CONTRACT_VERSION,
    StructuredAnchoredFusionAdapter,
)
from longieye.model import DemoRiskModel  # noqa: E402
from scripts.run_demo import load_case  # noqa: E402
from scripts.run_multimodal_demo import load_fixture_images  # noqa: E402


SCENARIOS = {
    "both_images": ("both", "multimodal", 2),
    "missing_os": ("missing-os", "partial_fallback", 1),
    "missing_both": ("missing-both", "structured_fallback", 0),
}


def benchmark_scenario(
    adapter: StructuredAnchoredFusionAdapter,
    features: dict[str, float],
    scenario_name: str,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float | int | str]:
    fixture_scenario, expected_mode, active_branches = SCENARIOS[scenario_name]
    images = load_fixture_images(fixture_scenario)

    initial = adapter.predict_with_images(features, images)
    if initial.mode != expected_mode:
        raise RuntimeError("multimodal benchmark scenario contract failed")
    for _ in range(warmup):
        adapter.predict_with_images(features, images)

    latencies_ms: list[float] = []
    total_started = time.perf_counter()
    for _ in range(iterations):
        started = time.perf_counter_ns()
        adapter.predict_with_images(features, images)
        latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    total_seconds = time.perf_counter() - total_started

    summary: dict[str, float | int | str] = summarize_latencies(
        latencies_ms, total_seconds
    )
    summary.update(
        {
            "scenario": scenario_name,
            "expected_result_mode": expected_mode,
            "synthetic_image_branches": active_branches,
            "structured_fallback_branches": 2 - active_branches,
            "warmup_iterations": warmup,
            "errors": 0,
        }
    )
    return summary


def markdown_report(report: dict[str, object]) -> str:
    labels = {
        "both_images": "双眼合成图像",
        "missing_os": "缺失 OS（单眼回退）",
        "missing_both": "双眼缺失（完整回退）",
    }
    rows = []
    for result in report["results"]:
        rows.append(
            "| {label} | {iterations} | {p50_ms:.3f} | {p95_ms:.3f} | "
            "{p99_ms:.3f} | {throughput_requests_per_second:.1f} | "
            "{synthetic_image_branches} | {structured_fallback_branches} |".format(
                label=labels[result["scenario"]], **result
            )
        )
    return "\n".join(
        [
            "# LongiEye 全合成多模态本机基准",
            "",
            f"生成时间：`{report['generated_at_utc']}`",
            f"包版本：`{report['package_version']}`",
            f"融合合同：`{report['fusion_contract_version']}`",
            f"Python：`{report['environment']['python']}`",
            f"运行平台：`{report['environment']['platform']}`",
            "",
            "| 场景 | 迭代 | P50 ms | P95 ms | P99 ms | 顺序吞吐量/秒 | 图像分支 | 回退分支 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "计时范围包含质量检查、32×32 预处理、确定性统计编码和有界融合，"
            "不包含文件读取。",
            "所有模式均为单进程顺序本机测量，不包含网络、容器或并发负载。",
            "报告不保存图像、路径、像素、embedding、逐例分数、病例别名或 AUC。",
            "这些结果只说明全合成工程路径与回退开销，不是临床性能指标。",
            "",
        ]
    )


def build_report(warmup: int, iterations: int) -> dict[str, object]:
    case, _ = load_case(PROJECT_ROOT / "examples" / "request.json")
    features = extract_features(case)
    model = DemoRiskModel.from_path(PROJECT_ROOT / "configs" / "demo_model.json")
    adapter = StructuredAnchoredFusionAdapter(model)
    return {
        "schema_version": 1,
        "package_version": __version__,
        "fusion_contract_version": FUSION_CONTRACT_VERSION,
        "image_embedding_contract_version": IMAGE_EMBEDDING_CONTRACT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_stage": "demo_multimodal_synthetic",
        "image_source": "procedurally_generated_synthetic_only",
        "clinical_use": False,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "measurement": {
            "sequential": True,
            "includes_file_io": False,
            "warmup_iterations": warmup,
            "iterations_per_scenario": iterations,
            "persists_case_outputs": False,
        },
        "results": [
            benchmark_scenario(
                adapter,
                features,
                scenario,
                warmup=warmup,
                iterations=iterations,
            )
            for scenario in SCENARIOS
        ],
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="测量全合成图像融合与结构化回退路径。"
    )
    parser.add_argument("--warmup", type=int, default=50, help="每个场景预热次数。")
    parser.add_argument(
        "--iterations", type=int, default=1000, help="每个场景计时次数。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmarks",
        help="报告输出目录。",
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise SystemExit("预热和计时次数必须为正整数。")

    report = build_report(args.warmup, args.iterations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "multimodal_latest.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown = markdown_report(report)
    (args.output_dir / "multimodal_latest.md").write_text(
        markdown,
        encoding="utf-8",
        newline="\n",
    )
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
