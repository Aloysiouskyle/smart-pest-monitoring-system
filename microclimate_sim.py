"""Generate synthetic hourly maize-field telemetry for the two WP 2.1 scenarios."""

import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypedDict


class TelemetryRecord(TypedDict):
    """One hourly observation with an ISO 8601 timestamp in East Africa Time."""

    timestamp: str
    temperature_c: float
    relative_humidity_pct: float
    rainfall_mm_per_hour: float
    soil_moisture_pct: float


@dataclass(frozen=True)
class _Profile:
    temperature_midpoint: float
    humidity_midpoint: float
    rain_probability: float
    max_rainfall: float
    soil_moisture_midpoint: float


# Illustrative inputs chosen to exercise two scenarios. See the bounds in README.
_PROFILES = {
    "normal": _Profile(24, 70, 0.12, 4, 36),
    "higher_risk": _Profile(27, 80, 0.20, 6, 44),
}
_PROJECT_ROOT = Path(__file__).resolve().parent
_EAT = timezone(timedelta(hours=3))
_DEFAULT_START = datetime(2026, 11, 18, tzinfo=_EAT)


def generate_telemetry(
    scenario: str, *, seed: int = 42, start: datetime = _DEFAULT_START
) -> list[TelemetryRecord]:
    """Return 72 hourly records starting at an aware datetime on an EAT hour.

    The local random generator and fixed default start make repeated calls with
    the same inputs reproducible. No files are read or written.
    """
    if not isinstance(scenario, str) or scenario not in _PROFILES:
        raise ValueError("scenario must be 'normal' or 'higher_risk'")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(start, datetime) or start.utcoffset() is None:
        raise ValueError("start must be a timezone-aware datetime")
    start = start.astimezone(_EAT)
    if start.minute or start.second or start.microsecond:
        raise ValueError("start must fall on a whole hour in East Africa Time")

    profile = _PROFILES[scenario]
    rng = random.Random(seed)
    records: list[TelemetryRecord] = []
    for offset in range(72):
        timestamp = start + timedelta(hours=offset)
        # Peak warmth at 15:00 EAT, with humidity rising again overnight.
        warmth = math.cos(math.tau * (timestamp.hour - 15) / 24)
        rain = 0.0
        if rng.random() < profile.rain_probability:
            rain = round(rng.uniform(0.1, profile.max_rainfall), 2)
        temperature = profile.temperature_midpoint + 4 * warmth + rng.uniform(-1, 1)
        humidity = profile.humidity_midpoint - 12 * warmth + rng.uniform(-3, 3)
        moisture = profile.soil_moisture_midpoint + rain + rng.uniform(-2, 2)
        records.append(
            TelemetryRecord(
                timestamp=timestamp.isoformat(),
                temperature_c=round(temperature, 2),
                relative_humidity_pct=round(humidity, 2),
                rainfall_mm_per_hour=rain,
                soil_moisture_pct=round(moisture, 2),
            )
        )
    return records


def export_scenario(
    scenario: str, file_format: str, *, seed: int = 42, start: datetime = _DEFAULT_START
) -> Path:
    """Generate one scenario and replace data/generated/<scenario>.<format>."""
    if file_format not in ("csv", "json"):
        raise ValueError("file_format must be 'csv' or 'json'")
    records = generate_telemetry(scenario, seed=seed, start=start)
    directory = _PROJECT_ROOT / "data" / "generated"
    if directory.resolve() != directory:
        raise ValueError("Export directory must resolve to the project's data/generated")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{scenario}.{file_format}"
    # Replace a complete file so readers cannot see a partial export.
    output = NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=directory, delete=False
    )
    temporary = Path(output.name)
    try:
        with output:
            if file_format == "csv":
                writer = csv.DictWriter(output, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            else:
                json.dump(records, output, indent=2)
                output.write("\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
