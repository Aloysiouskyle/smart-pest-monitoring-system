"""Assess Fall Armyworm evidence in one synthetic WP 2.1 observation window."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from microclimate_sim import TelemetryRecord


# The six detector classes in data.yaml. Only the four FAW classes contribute.
FAW_LABELS = frozenset({
    "fall-armyworm-egg", "fall-armyworm-frass",
    "fall-armyworm-larva", "fall-armyworm-larval-damage",
})
MODEL_LABELS = FAW_LABELS | {"healthy-maize", "maize-streak-disease"}
LIMITATION = (
    "This is a simulated rule-based assessment, not a scientifically validated "
    "48–72-hour forecast. YOLO confidence describes a detection in the submitted image."
)


def _number(value: object, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number from {low} to {high}")
    return float(value)


@dataclass(frozen=True)
class RiskRules:
    """Inclusive demonstration cutoffs within the WP 2.1 synthetic ranges."""

    min_temperature_c: float = 25
    min_humidity_pct: float = 72
    min_matching_hours: int = 18
    min_detection_confidence: float = 0.50

    def __post_init__(self) -> None:
        _number(self.min_temperature_c, "min_temperature_c", 19, 32)
        _number(self.min_humidity_pct, "min_humidity_pct", 55, 95)
        _number(self.min_detection_confidence, "min_detection_confidence", 0, 1)
        if type(self.min_matching_hours) is not int or not 1 <= self.min_matching_hours <= 72:
            raise ValueError("min_matching_hours must be an integer from 1 to 72")


@dataclass(frozen=True)
class RiskAssessment:
    risk_level: Literal["Low", "Moderate", "High"]
    contributing_factors: tuple[str, str]
    telemetry_hours: int
    matching_hours: int
    image_status: Literal["no_image", "zero_detections", "detections_present"]
    strongest_faw_confidence: float | None
    limitation: str = LIMITATION
    visual_rule_met: bool | None = None  # Older assessments may omit the rule outcome.


def _matching_hours(records: Sequence[TelemetryRecord], rules: RiskRules) -> int:
    if not isinstance(records, Sequence) or len(records) != 72:
        raise ValueError("A complete sequence of 72 hourly observations is required")
    previous = None
    matches = 0
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise ValueError(f"Record {index} must be a telemetry mapping")
        stamp = row.get("timestamp")
        if not isinstance(stamp, str):
            raise ValueError(f"Record {index} requires an ISO timestamp")
        instant = datetime.fromisoformat(stamp)
        if instant.utcoffset() is None:
            raise ValueError(f"Record {index} requires a timezone offset")
        instant = instant.astimezone(timezone.utc)
        if instant.minute or instant.second or instant.microsecond:
            raise ValueError(f"Record {index} must fall on a whole hour in East Africa Time")
        if previous is not None and instant - previous != timedelta(hours=1):
            raise ValueError(f"Record {index} must follow the previous record by one hour")
        previous = instant
        temperature = _number(row.get("temperature_c"), "temperature_c", 19, 32)
        humidity = _number(row.get("relative_humidity_pct"), "relative_humidity_pct", 55, 95)
        _number(row.get("rainfall_mm_per_hour"), "rainfall_mm_per_hour", 0, 6)
        _number(row.get("soil_moisture_pct"), "soil_moisture_pct", 34, 52)
        if temperature >= rules.min_temperature_c and humidity >= rules.min_humidity_pct:
            matches += 1
    return matches


def assess_risk(
    records: Sequence[TelemetryRecord],
    detections: Mapping[str, list[float]] | None = None,
    *,
    rules: RiskRules = RiskRules(),
) -> RiskAssessment:
    """Combine one complete telemetry window with optional grouped YOLO scores.

    Pass None for no submitted image and an empty mapping for a processed image
    with zero boxes. Malformed or incomplete evidence raises ValueError.
    """
    if not isinstance(rules, RiskRules):
        raise ValueError("rules must be a RiskRules instance")
    matches = _matching_hours(records, rules)
    climate_factor = (
        f"{matches}/72 hours met both cutoffs ({rules.min_temperature_c:g} °C and "
        f"{rules.min_humidity_pct:g}% RH). The climate rule requires {rules.min_matching_hours} hours."
    )
    best = None
    best_label = None
    image_status = "no_image"
    visual_factor = "No image submitted. Visual evidence is unavailable."
    if detections is not None:
        if not isinstance(detections, Mapping):
            raise ValueError("detections must map class labels to lists of confidence scores")
        image_status = "zero_detections"
        visual_factor = "Image processed with zero detections."
        for label, scores in detections.items():
            if label not in MODEL_LABELS:
                raise ValueError(f"Unknown detection class {label!r}")
            if not isinstance(scores, list) or not scores:
                raise ValueError(f"{label} requires a nonempty list of confidence scores")
            image_status = "detections_present"
            for score in scores:
                confidence = _number(score, f"{label} confidence", 0, 1)
                if label in FAW_LABELS and (best is None or confidence > best):
                    best_label, best = label, confidence
        if detections and best is None:
            visual_factor = "Image contained detections only from other maize classes."
    visual_match = best is not None and best >= rules.min_detection_confidence
    if best is not None:
        outcome = "met" if visual_match else "fell below"
        visual_factor = (
            f"Strongest Fall Armyworm box ({best_label}, score {best}) "
            f"{outcome} the visual cutoff of {rules.min_detection_confidence:g}."
        )
    climate_match = matches >= rules.min_matching_hours
    levels = ("Low", "Moderate", "High")
    return RiskAssessment(
        risk_level=levels[int(climate_match) + int(visual_match)],
        contributing_factors=(climate_factor, visual_factor),
        telemetry_hours=len(records),
        matching_hours=matches,
        image_status=image_status,
        strongest_faw_confidence=best,
        visual_rule_met=visual_match,
    )
