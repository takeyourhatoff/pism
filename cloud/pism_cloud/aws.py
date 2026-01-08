"""AWS helper utilities for instance discovery and pricing."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

import boto3


@dataclass(frozen=True)
class InstanceSpec:
    instance_type: str
    vcpus: int
    memory_mib: int
    gpus: int


def aws_region() -> str:
    session = boto3.session.Session()
    return session.region_name or os.environ.get("AWS_REGION", "us-east-1")


@lru_cache(maxsize=64)
def instance_spec(instance_type: str, region: Optional[str] = None) -> InstanceSpec:
    region = region or aws_region()
    ec2 = boto3.client("ec2", region_name=region)
    response = ec2.describe_instance_types(InstanceTypes=[instance_type])
    info = response["InstanceTypes"][0]

    gpus = 0
    if "GpuInfo" in info:
        gpus = sum(gpu.get("Count", 0) for gpu in info["GpuInfo"].get("Gpus", []))

    return InstanceSpec(
        instance_type=instance_type,
        vcpus=int(info["VCpuInfo"]["DefaultVCpus"]),
        memory_mib=int(info["MemoryInfo"]["SizeInMiB"]),
        gpus=int(gpus),
    )


@lru_cache(maxsize=128)
def on_demand_hourly_rate(instance_type: str, region: Optional[str] = None) -> float:
    region = region or aws_region()
    pricing = boto3.client("pricing", region_name="us-east-1")

    filters = [
        {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
        {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
        {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
        {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
        {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
        {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
        {"Type": "TERM_MATCH", "Field": "licenseModel", "Value": "No License required"},
    ]

    response = pricing.get_products(ServiceCode="AmazonEC2", Filters=filters)
    price_list = response.get("PriceList", [])
    if not price_list:
        raise ValueError(
            "Unable to fetch on-demand pricing from AWS. "
            "Set cost.estimated_usd_per_hour in the config to override."
        )

    data = json.loads(price_list[0])
    on_demand_terms = data.get("terms", {}).get("OnDemand", {})
    for term in on_demand_terms.values():
        price_dimensions = term.get("priceDimensions", {})
        for dimension in price_dimensions.values():
            price_per_unit = dimension.get("pricePerUnit", {})
            if "USD" in price_per_unit:
                return float(price_per_unit["USD"])

    raise ValueError(
        "Unable to parse on-demand pricing. "
        "Set cost.estimated_usd_per_hour in the config to override."
    )


def spot_hourly_rate(instance_type: str, region: Optional[str] = None) -> float:
    region = region or aws_region()
    ec2 = boto3.client("ec2", region_name=region)

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=1)

    response = ec2.describe_spot_price_history(
        InstanceTypes=[instance_type],
        ProductDescriptions=["Linux/UNIX"],
        StartTime=start,
        EndTime=end,
        MaxResults=100,
    )

    history = response.get("SpotPriceHistory", [])
    if not history:
        raise ValueError(
            "Unable to fetch spot pricing from AWS. "
            "Set cost.estimated_usd_per_hour in the config to override."
        )

    latest_by_az = {}
    for entry in history:
        az = entry["AvailabilityZone"]
        ts = entry["Timestamp"]
        price = float(entry["SpotPrice"])
        if az not in latest_by_az or ts > latest_by_az[az][0]:
            latest_by_az[az] = (ts, price)

    if not latest_by_az:
        raise ValueError(
            "Unable to compute spot pricing from AWS. "
            "Set cost.estimated_usd_per_hour in the config to override."
        )

    prices = [price for _, price in latest_by_az.values()]
    return sum(prices) / len(prices)
