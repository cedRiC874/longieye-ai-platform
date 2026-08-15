"""Small dependency-free helpers for reproducible latency summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile for an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_latencies(
    latencies_ms: Sequence[float], total_seconds: float
) -> dict[str, float | int]:
    if total_seconds <= 0.0:
        raise ValueError("total_seconds must be positive")
    return {
        "iterations": len(latencies_ms),
        "p50_ms": round(percentile(latencies_ms, 0.50), 6),
        "p95_ms": round(percentile(latencies_ms, 0.95), 6),
        "p99_ms": round(percentile(latencies_ms, 0.99), 6),
        "mean_ms": round(sum(latencies_ms) / len(latencies_ms), 6),
        "throughput_requests_per_second": round(
            len(latencies_ms) / total_seconds, 3
        ),
    }
