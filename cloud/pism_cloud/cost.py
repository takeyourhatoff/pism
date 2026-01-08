"""Cost estimation helpers for PISM Batch runs."""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import INSTANCE_CATALOG

SPOT_DISCOUNT_FACTOR = 0.4
DEFAULT_SPOT_THRESHOLD_USD = 250.0


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
) -> CostEstimate:
    if instance_type not in INSTANCE_CATALOG:
        raise ValueError(f"Unknown instance type: {instance_type}")

    hourly = INSTANCE_CATALOG[instance_type]["on_demand_usd_per_hour"]
    if use_spot:
        hourly *= SPOT_DISCOUNT_FACTOR

    total = hourly * hours_per_member * ensemble_members
    return CostEstimate(hourly_rate_usd=hourly, total_usd=total, use_spot=use_spot)
