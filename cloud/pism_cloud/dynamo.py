"""DynamoDB helpers for run metadata."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import boto3
from boto3.dynamodb.conditions import Attr, Key

DEFAULT_TABLE_NAME = "pism-runs"


def table_name() -> str:
    return os.environ.get("PISM_RUNS_TABLE", DEFAULT_TABLE_NAME)


def get_table():
    resource = boto3.resource("dynamodb")
    return resource.Table(table_name())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def store_run(table, run: Dict[str, Any]) -> None:
    table.put_item(Item=run)


def store_job(table, job: Dict[str, Any]) -> None:
    table.put_item(Item=job)


def list_runs(table) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    params = {"FilterExpression": Attr("sort_key").eq("RUN")}
    while True:
        response = table.scan(**params)
        items.extend(response.get("Items", []))
        if "LastEvaluatedKey" not in response:
            break
        params["ExclusiveStartKey"] = response["LastEvaluatedKey"]
    return items


def list_jobs(table, run_id: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    params = {"KeyConditionExpression": Key("run_id").eq(run_id)}
    while True:
        response = table.query(**params)
        items.extend(response.get("Items", []))
        if "LastEvaluatedKey" not in response:
            break
        params["ExclusiveStartKey"] = response["LastEvaluatedKey"]
    return [item for item in items if item.get("sort_key") == "JOB"]


def get_run(table, run_id: str) -> Dict[str, Any] | None:
    response = table.get_item(Key={"run_id": run_id, "sort_key": "RUN"})
    return response.get("Item")
