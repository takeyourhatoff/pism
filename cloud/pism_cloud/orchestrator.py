"""Step Functions orchestration helpers."""

from __future__ import annotations

import json
from typing import Dict

import boto3

from .aws import aws_region


def start_execution(state_machine_arn: str, payload: Dict[str, object]) -> str:
    client = boto3.client("stepfunctions", region_name=aws_region())
    response = client.start_execution(
        stateMachineArn=state_machine_arn,
        input=json.dumps(payload),
    )
    return str(response.get("executionArn", ""))


def stop_execution(execution_arn: str, reason: str = "") -> None:
    client = boto3.client("stepfunctions", region_name=aws_region())
    payload = {"executionArn": execution_arn}
    if reason:
        payload["cause"] = reason
    client.stop_execution(**payload)
