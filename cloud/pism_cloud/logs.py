"""CloudWatch Logs helpers for progress estimation."""

from __future__ import annotations

from typing import List, Tuple

import boto3

from .aws import aws_region


def fetch_log_events(
    log_group: str,
    log_stream: str,
    limit: int = 200,
    include_head: bool = False,
    head_limit: int = 200,
) -> List[Tuple[int, str]]:
    client = boto3.client("logs", region_name=aws_region())
    events: List[Tuple[int, str]] = []
    if include_head:
        response = client.get_log_events(
            logGroupName=log_group,
            logStreamName=log_stream,
            limit=head_limit,
            startFromHead=True,
        )
        head_events = response.get("events", [])
        events.extend(
            [(int(event.get("timestamp", 0)), str(event.get("message", ""))) for event in head_events]
        )
    response = client.get_log_events(
        logGroupName=log_group,
        logStreamName=log_stream,
        limit=limit,
        startFromHead=False,
    )
    tail_events = response.get("events", [])
    events.extend(
        [(int(event.get("timestamp", 0)), str(event.get("message", ""))) for event in tail_events]
    )
    return events
