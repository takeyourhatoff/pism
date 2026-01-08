"""CloudWatch Logs helpers for progress estimation."""

from __future__ import annotations

from typing import List, Tuple

import boto3

from .aws import aws_region


def fetch_log_events(
    log_group: str,
    log_stream: str,
    limit: int = 200,
) -> List[Tuple[int, str]]:
    client = boto3.client("logs", region_name=aws_region())
    response = client.get_log_events(
        logGroupName=log_group,
        logStreamName=log_stream,
        limit=limit,
        startFromHead=False,
    )
    events = response.get("events", [])
    return [(int(event.get("timestamp", 0)), str(event.get("message", ""))) for event in events]
