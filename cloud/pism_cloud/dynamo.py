"""DynamoDB helpers for job metadata."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List

import boto3
from boto3.dynamodb.conditions import Attr

from .aws import aws_region
DEFAULT_TABLE_NAME = "pism-jobs"


def table_name() -> str:
    return os.environ.get("PISM_JOBS_TABLE", DEFAULT_TABLE_NAME)


def get_table():
    resource = boto3.resource("dynamodb", region_name=aws_region())
    return resource.Table(table_name())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_numbers(value: Any) -> Any:
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {key: _normalize_numbers(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_numbers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_numbers(item) for item in value)
    return value


def store_job(table, job: Dict[str, Any]) -> None:
    table.put_item(Item=_normalize_numbers(job))


def list_jobs(table) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    params = {"FilterExpression": Attr("sort_key").eq("JOB")}
    while True:
        response = table.scan(**params)
        items.extend(response.get("Items", []))
        if "LastEvaluatedKey" not in response:
            break
        params["ExclusiveStartKey"] = response["LastEvaluatedKey"]
    return items


def get_job(table, job_name: str) -> Dict[str, Any] | None:
    response = table.get_item(Key={"job_name": job_name, "sort_key": "JOB"})
    return response.get("Item")
