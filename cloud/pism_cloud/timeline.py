"""Timeline helpers for AWS Batch job history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple

RESUME_PREFIX = "Resuming from "


@dataclass(frozen=True)
class TimelineEvent:
    timestamp_ms: int
    label: str
    detail: str

    def iso_timestamp(self) -> str:
        if self.timestamp_ms <= 0:
            return ""
        return datetime.fromtimestamp(self.timestamp_ms / 1000.0, tz=timezone.utc).isoformat()


def _classify_attempt_end(reason: str | None) -> tuple[str, str]:
    if not reason:
        return "Attempt ended", ""
    lower = reason.lower()
    if "host ec2" in lower or "instance terminated" in lower:
        return "Spot interruption", reason
    if "retry" in lower and "exceed" in lower:
        return "Retries exhausted", reason
    return "Attempt ended", reason


def extract_resume_events(events: Iterable[Tuple[int, str]]) -> List[TimelineEvent]:
    resume_events: List[TimelineEvent] = []
    for ts, message in events:
        text = message.strip()
        if text.startswith(RESUME_PREFIX):
            uri = text[len(RESUME_PREFIX) :].strip()
            resume_events.append(TimelineEvent(timestamp_ms=ts, label="Resumed", detail=uri))
    return resume_events


def _iso_to_ms(value: str | None) -> int:
    if not value:
        return 0
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return 0
    return int(parsed.timestamp() * 1000)


def build_timeline_for_attempts(
    job_item: Dict[str, object],
    jobs: List[Dict[str, object]],
    attempt_job_ids: List[str],
    log_events: Iterable[Tuple[int, str]],
) -> List[TimelineEvent]:
    events: List[TimelineEvent] = []
    created_at = _iso_to_ms(str(job_item.get("created_at") or ""))
    if created_at:
        events.append(TimelineEvent(timestamp_ms=created_at, label="Submitted", detail=""))

    job_map = {str(job.get("jobId") or ""): job for job in jobs}
    total_attempts = len(attempt_job_ids)
    for idx, job_id in enumerate(attempt_job_ids, start=1):
        job = job_map.get(job_id)
        if not job:
            continue
        started_at = int(job.get("startedAt") or 0)
        created_at_job = int(job.get("createdAt") or 0)
        if started_at:
            events.append(
                TimelineEvent(
                    timestamp_ms=started_at,
                    label=f"Attempt {idx} started",
                    detail="",
                )
            )
        elif created_at_job:
            events.append(
                TimelineEvent(
                    timestamp_ms=created_at_job,
                    label=f"Attempt {idx} submitted",
                    detail="",
                )
            )
        stopped_at = int(job.get("stoppedAt") or 0)
        status = str(job.get("status") or "")
        reason = str(job.get("statusReason") or "")
        if stopped_at and status == "FAILED":
            label, detail = _classify_attempt_end(reason)
            if label == "Attempt ended":
                label = f"Attempt {idx} failed"
            events.append(
                TimelineEvent(
                    timestamp_ms=stopped_at,
                    label=label,
                    detail=detail,
                )
            )
        elif stopped_at and status == "SUCCEEDED":
            if idx < total_attempts:
                events.append(
                    TimelineEvent(
                        timestamp_ms=stopped_at,
                        label=f"Attempt {idx} interrupted",
                        detail="resume requested",
                    )
                )
            else:
                events.append(
                    TimelineEvent(
                        timestamp_ms=stopped_at,
                        label="Job completed",
                        detail="",
                    )
                )

    events.extend(extract_resume_events(log_events))
    events.sort(key=lambda item: item.timestamp_ms)
    return events
