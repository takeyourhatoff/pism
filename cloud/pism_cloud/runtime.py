"""Runtime cost helpers for PISM Batch jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional

from .aws import aws_region, on_demand_hourly_rate, spot_hourly_rate

@dataclass(frozen=True)
class CostSnapshot:
    hourly_rate_usd: float
    cost_to_date_usd: float


def hourly_rate(instance_type: str, use_spot: bool, region: Optional[str] = None) -> float:
    region = region or aws_region()
    if use_spot:
        return spot_hourly_rate(instance_type, region)
    return on_demand_hourly_rate(instance_type, region)


def _job_runtime_seconds(job: Dict[str, object], now_ms: int) -> float:
    started = int(job.get("startedAt") or 0)
    if started <= 0:
        return 0.0
    stopped = int(job.get("stoppedAt") or 0)
    end = stopped if stopped > 0 else now_ms
    if end < started:
        return 0.0
    return (end - started) / 1000.0


def summarize_costs(
    jobs: Iterable[Dict[str, object]],
    hourly_rate_usd: float,
    now: Optional[datetime] = None,
) -> CostSnapshot:
    now_ms = int((now or datetime.now(timezone.utc)).timestamp() * 1000)
    cost_to_date = 0.0
    for job in jobs:
        runtime_hours = _job_runtime_seconds(job, now_ms) / 3600.0
        if runtime_hours > 0:
            cost_to_date += runtime_hours * hourly_rate_usd
    return CostSnapshot(
        hourly_rate_usd=hourly_rate_usd,
        cost_to_date_usd=cost_to_date,
    )


def summarize_runtime_seconds(
    jobs: Iterable[Dict[str, object]],
    now: Optional[datetime] = None,
) -> float | None:
    now_ms = int((now or datetime.now(timezone.utc)).timestamp() * 1000)
    started_times = []
    end_times = []
    for job in jobs:
        started = int(job.get("startedAt") or 0)
        if started <= 0:
            continue
        stopped = int(job.get("stoppedAt") or 0)
        started_times.append(started)
        end_times.append(stopped if stopped > 0 else now_ms)
    if not started_times:
        return None
    return (max(end_times) - min(started_times)) / 1000.0
