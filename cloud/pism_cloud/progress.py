"""Progress estimation from PISM log output."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Tuple

RUN_TIME_RE = re.compile(
    r"\* Run time:\s*\[([+-]?\d+(?:\.\d+)?)\s+[^,]+,\s*([+-]?\d+(?:\.\d+)?)\s+[^\]]+\]"
)
RUN_TIME_DATE_RE = re.compile(
    r"\* Run time:\s*\[([+-]?\d+-\d{2}-\d{2}\s+\d+(?:\.\d+)?h)\s*,\s*"
    r"([+-]?\d+-\d{2}-\d{2}\s+\d+(?:\.\d+)?h)\s*\]"
)
S_LINE_RE = re.compile(r"^S\s+([^:\s]+):")
S_DATE_RE = re.compile(r"^S\s+([+-]?\d+)-(\d{2})-(\d{2})\s+(\d+(?:\.\d+)?)h:")
DATE_TIME_TOKEN_RE = re.compile(r"([+-]?\d+)-(\d{2})-(\d{2})\s+(\d+(?:\.\d+)?)h")
MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


@dataclass(frozen=True)
class ProgressEstimate:
    end_time: float
    current_time: float
    progress_fraction: float
    eta_seconds: float
    sample_count: int
    wall_seconds: float


def _date_parts_to_years(
    year_str: str,
    month_str: str,
    day_str: str,
    hour_str: str,
) -> float | None:
    try:
        year = int(year_str)
        month = int(month_str)
        day = int(day_str)
        hour = float(hour_str)
    except ValueError:
        return None
    if month < 1 or month > 12:
        return None
    month_len = MONTH_LENGTHS[month - 1]
    if day < 1 or day > month_len:
        return None
    if hour < 0 or hour >= 24:
        return None
    day_of_year = sum(MONTH_LENGTHS[: month - 1]) + day
    return year + ((day_of_year - 1) + hour / 24.0) / 365.0


def _parse_date_time_token(token: str) -> float | None:
    match = DATE_TIME_TOKEN_RE.search(token.strip())
    if not match:
        return None
    return _date_parts_to_years(*match.groups())


def _parse_run_time(lines: Iterable[str]) -> tuple[float, float] | None:
    for line in lines:
        match = RUN_TIME_RE.search(line)
        if match:
            try:
                start = float(match.group(1))
                end = float(match.group(2))
            except ValueError:
                start = None
                end = None
            if start is not None and end is not None and end != start:
                return start, end
        match = RUN_TIME_DATE_RE.search(line)
        if not match:
            continue
        start = _parse_date_time_token(match.group(1))
        end = _parse_date_time_token(match.group(2))
        if start is None or end is None or end == start:
            continue
        return start, end
    return None


def _parse_model_time(line: str) -> float | None:
    match = S_LINE_RE.match(line.strip())
    if not match:
        date_match = S_DATE_RE.match(line.strip())
        if not date_match:
            return None
        return _date_parts_to_years(*date_match.groups())
    token = match.group(1)
    try:
        return float(token)
    except ValueError:
        date_match = S_DATE_RE.match(line.strip())
        if not date_match:
            return None
        return _date_parts_to_years(*date_match.groups())


def estimate_progress(
    events: Iterable[Tuple[int, str]],
    sample_limit: int = 20,
    monotonic: bool = False,
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

    if monotonic:
        monotonic_samples: list[tuple[int, float]] = []
        max_time: float | None = None
        for ts, model_time in samples:
            if max_time is None or model_time > max_time:
                monotonic_samples.append((ts, model_time))
                max_time = model_time
        if len(monotonic_samples) < 2:
            return None
        monotonic_samples = monotonic_samples[-sample_limit:]
        t0, m0 = monotonic_samples[0]
        t1, m1 = monotonic_samples[-1]
    else:
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
        end_time=end_time,
        current_time=m1,
        progress_fraction=progress,
        eta_seconds=eta_seconds,
        sample_count=len(samples),
        wall_seconds=wall_seconds,
    )
