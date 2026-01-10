"""CloudWatch Logs helpers for progress estimation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

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


def extract_log_streams(job: Dict[str, object]) -> List[str]:
    streams: List[str] = []
    container = job.get("container", {}) if isinstance(job.get("container"), dict) else {}
    current_stream = container.get("logStreamName")
    if current_stream:
        streams.append(str(current_stream))
    attempts = job.get("attempts", [])
    if isinstance(attempts, list):
        for attempt in attempts:
            container = attempt.get("container", {}) if isinstance(attempt, dict) else {}
            stream = container.get("logStreamName")
            if stream:
                streams.append(str(stream))
    seen: set[str] = set()
    unique: List[str] = []
    for stream in streams:
        if stream in seen:
            continue
        seen.add(stream)
        unique.append(stream)
    return unique


def extract_log_streams_for_jobs(jobs: Iterable[Dict[str, object]]) -> List[str]:
    streams: List[str] = []
    for job in jobs:
        streams.extend(extract_log_streams(job))
    seen: set[str] = set()
    unique: List[str] = []
    for stream in streams:
        if stream in seen:
            continue
        seen.add(stream)
        unique.append(stream)
    return unique


def fetch_log_events_for_streams(
    log_group: str,
    log_streams: Iterable[str],
    limit: int = 200,
    include_head: bool = False,
    head_limit: int = 200,
) -> List[Tuple[int, str]]:
    events: List[Tuple[int, str]] = []
    for stream in log_streams:
        events.extend(
            fetch_log_events(
                log_group,
                stream,
                limit=limit,
                include_head=include_head,
                head_limit=head_limit,
            )
        )
    return events
