import pytest

from longieye.benchmarking import percentile, summarize_latencies
from scripts.benchmark_service import markdown_report


def test_percentile_interpolates_and_validates_input():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)
    with pytest.raises(ValueError):
        percentile([], 0.50)


def test_latency_summary_reports_tail_and_throughput():
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0], total_seconds=0.02)
    assert summary["iterations"] == 4
    assert summary["p50_ms"] == 2.5
    assert summary["p95_ms"] == 3.85
    assert summary["throughput_requests_per_second"] == 200.0


def test_markdown_report_uses_chinese_demo_labels():
    result = {
        "mode": "core_service",
        "iterations": 10,
        "p50_ms": 1.0,
        "p95_ms": 2.0,
        "p99_ms": 3.0,
        "throughput_requests_per_second": 100.0,
        "rss_peak_delta_mb": 0.5,
        "python_tracemalloc_peak_mb": 0.25,
    }
    report = {
        "generated_at_utc": "2026-08-15T00:00:00+00:00",
        "model_id": "demo-model",
        "model_artifact_sha256": "abc123",
        "environment": {"python": "3.12", "platform": "test-platform"},
        "results": [result],
    }

    markdown = markdown_report(report)

    assert markdown.startswith("# LongiEye 本机基准")
    assert "核心服务（core_service）" in markdown
    assert "不是临床性能指标" in markdown
