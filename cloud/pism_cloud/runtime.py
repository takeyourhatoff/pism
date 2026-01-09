"""Runtime cost helpers for PISM Batch runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional

from .aws import aws_region, on_demand_hourly_rate, spot_hourly_rate

RUNNING_STATUSES = {"STARTING", "RUNNING"}


@dataclass(frozen=True)
class CostSnapshot:
    hourly_rate_usd: float
    running_rate_usd: float
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


def job_costs(
    jobs: Iterable[Dict[str, object]],
    hourly_rate_usd: float,
    now: Optional[datetime] = None,
) -> Dict[str, float]:
    now_ms = int((now or datetime.now(timezone.utc)).timestamp() * 1000)
    costs: Dict[str, float] = {}
    for job in jobs:
        job_id = str(job.get("jobId") or "")
        if not job_id:
            continue
        runtime_hours = _job_runtime_seconds(job, now_ms) / 3600.0
        costs[job_id] = runtime_hours * hourly_rate_usd if runtime_hours > 0 else 0.0
    return costs


def summarize_costs(
    jobs: Iterable[Dict[str, object]],
    hourly_rate_usd: float,
    now: Optional[datetime] = None,
) -> CostSnapshot:
    now_ms = int((now or datetime.now(timezone.utc)).timestamp() * 1000)
    running_jobs = 0
    cost_to_date = 0.0
    for job in jobs:
        runtime_hours = _job_runtime_seconds(job, now_ms) / 3600.0
        if runtime_hours > 0:
            cost_to_date += runtime_hours * hourly_rate_usd
        status = str(job.get("status") or "")
        if status in RUNNING_STATUSES:
            running_jobs += 1
    running_rate = running_jobs * hourly_rate_usd
    return CostSnapshot(
        hourly_rate_usd=hourly_rate_usd,
        running_rate_usd=running_rate,
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
