import pytest

from longieye.benchmarking import percentile, summarize_latencies


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
