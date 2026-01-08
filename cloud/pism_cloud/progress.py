"""Progress estimation from PISM log output."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Tuple

RUN_TIME_RE = re.compile(
    r"\* Run time:\s*\[([+-]?\d+(?:\.\d+)?)\s+[^,]+,\s*([+-]?\d+(?:\.\d+)?)\s+[^\]]+\]"
)
S_LINE_RE = re.compile(r"^S\s+([^:\s]+):")


@dataclass(frozen=True)
class ProgressEstimate:
    start_time: float
    end_time: float
    current_time: float
    progress_fraction: float
    eta_seconds: float
    model_units_per_hour: float
    sample_count: int
    wall_seconds: float


def _parse_run_time(lines: Iterable[str]) -> tuple[float, float] | None:
    for line in lines:
        match = RUN_TIME_RE.search(line)
        if not match:
            continue
        try:
            start = float(match.group(1))
            end = float(match.group(2))
        except ValueError:
            continue
        if end == start:
            continue
        return start, end
    return None


def _parse_model_time(line: str) -> float | None:
    match = S_LINE_RE.match(line.strip())
    if not match:
        return None
    token = match.group(1)
    try:
        return float(token)
    except ValueError:
        return None


def estimate_progress(
    events: Iterable[Tuple[int, str]],
    sample_limit: int = 20,
) -> ProgressEstimate | None:
    ordered = sorted(events, key=lambda item: item[0])
    lines = [message for _, message in ordered]
    run_time = _parse_run_time(lines)
    if run_time is None:
        return None
    start_time, end_time = run_time

    samples = []
    for ts, message in ordered:
        model_time = _parse_model_time(message)
        if model_time is None:
            continue
        samples.append((ts, model_time))

    if len(samples) < 2:
        return None

    samples = samples[-sample_limit:]
    t0, m0 = samples[0]
    t1, m1 = samples[-1]
    if t1 <= t0 or m1 <= m0:
        return None

    wall_seconds = (t1 - t0) / 1000.0
    rate = (m1 - m0) / wall_seconds
    if rate <= 0:
        return None

    remaining = max(0.0, end_time - m1)
    eta_seconds = remaining / rate
    progress = (m1 - start_time) / (end_time - start_time)
    progress = max(0.0, min(1.0, progress))

    return ProgressEstimate(
        start_time=start_time,
        end_time=end_time,
        current_time=m1,
        progress_fraction=progress,
        eta_seconds=eta_seconds,
        model_units_per_hour=rate * 3600.0,
        sample_count=len(samples),
        wall_seconds=wall_seconds,
    )
