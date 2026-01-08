"""Cost estimation helpers for PISM Batch runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .aws import aws_region, on_demand_hourly_rate, spot_hourly_rate


@dataclass(frozen=True)
class CostEstimate:
    hourly_rate_usd: float
    total_usd: float
    use_spot: bool


def estimate_total_cost(
    instance_type: str,
    hours_per_member: float,
    ensemble_members: int,
    use_spot: bool,
    hourly_override: Optional[float] = None,
) -> CostEstimate:
    if hourly_override is not None:
        hourly = float(hourly_override)
    else:
        region = aws_region()
        if use_spot:
            hourly = spot_hourly_rate(instance_type, region)
        else:
            hourly = on_demand_hourly_rate(instance_type, region)

    total = hourly * hours_per_member * ensemble_members
    return CostEstimate(hourly_rate_usd=hourly, total_usd=total, use_spot=use_spot)
