"""Inference latency profiler for PerceptionMetrics GUI.

Records per-image wall-clock inference time and computes
summary statistics: mean, median, P95, P99, min, max, FPS.
"""
from __future__ import annotations
import time
import statistics
from dataclasses import dataclass, field
from typing import List


@dataclass
class LatencyReport:
    latencies_ms: List[float] = field(default_factory=list)

    def record(self, elapsed_seconds: float) -> None:
        self.latencies_ms.append(elapsed_seconds * 1000.0)

    def summary(self) -> dict:
        if not self.latencies_ms:
            return {}
        n = len(self.latencies_ms)
        mean = statistics.mean(self.latencies_ms)
        sorted_lat = sorted(self.latencies_ms)
        p95 = sorted_lat[max(0, int(0.95 * n) - 1)]
        p99 = sorted_lat[max(0, int(0.99 * n) - 1)]
        return {
            "images_evaluated": n,
            "mean_ms":   round(mean, 2),
            "median_ms": round(statistics.median(self.latencies_ms), 2),
            "p95_ms":    round(p95, 2),
            "p99_ms":    round(p99, 2),
            "min_ms":    round(min(self.latencies_ms), 2),
            "max_ms":    round(max(self.latencies_ms), 2),
            "fps":       round(1000.0 / mean, 1) if mean > 0 else 0.0,
        }