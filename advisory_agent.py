"""Bilingual SMS previews for simulated Fall Armyworm assessments."""

from dataclasses import dataclass
from string import ascii_letters, digits

from forecasting_agent import RiskAssessment


# A deliberately small subset of the default GSM-7 alphabet. Each costs one septet.
_SMS_ALPHABET = frozenset(ascii_letters + digits + " .,:'-!?")

# Each entry contains the simulated risk heading and a field action.
_ACTIONS = {
    "en": {
        "Low": ("Demo: low fall armyworm risk.", "Keep checking under maize leaves for eggs."),
        "Moderate": ("Demo: moderate fall armyworm risk.", "Check young maize leaves for caterpillars today."),
        "High": ("Demo: high fall armyworm risk.", "Check maize today and ask your farm adviser for help."),
    },
    # Luganda drafts require human review before use with farmers.
    "lg": {
        "Low": ("Okugezesa: obulabe butono.", "Keberanga wansi w'amakoola ga kasooli oba waliwo amagi."),
        "Moderate": ("Okugezesa: obulabe bwa wakati.", "Leero noonya ebiwuka ku makoola ga kasooli omuto."),
        "High": ("Okugezesa: obulabe bungi.", "Kebera kasooli leero. Weebuuze ku mukugu mu byobulimi."),
    },
}

_EVIDENCE = {
    "en": {
        "no_image": "Weather only. No photo submitted.",
        "unprocessed_image": "Weather only. Photo check unavailable.",
        "zero_detections": "Weather only. Photo check flagged nothing.",
        "other_classes": "Weather only. Photo flagged other maize features.",
        "weak_sign": "Photo hint too weak to count. Risk uses weather only.",
        "possible_sign": "Photo suggests possible armyworm signs.",
    },
    "lg": {
        "no_image": "Embeera y'obudde yokka. Tewali kifaananyi.",
        "unprocessed_image": "Embeera y'obudde yokka. Ekifaananyi tekikebereddwa.",
        "zero_detections": "Embeera y'obudde yokka. Tewali kyazuuliddwa mu kifaananyi.",
        "other_classes": "Embeera y'obudde yokka. Ekifaananyi kiraze ebirala ku kasooli.",
        "weak_sign": "Akabonero ku kifaananyi tekategeerekeka bulungi. Embeera y'obudde yokka.",
        "possible_sign": "Ekifaananyi kiyinza okulaga obubonero bw'ebiwuka.",
    },
}


@dataclass(frozen=True)
class FarmerAdvisory:
    """A forecast and its two SMS previews, ready for display."""

    assessment: RiskAssessment
    english: str
    luganda: str
    luganda_requires_review: bool = True


def _evidence_key(assessment: RiskAssessment) -> str:
    status = assessment.image_status
    score = assessment.strongest_faw_confidence
    met = assessment.visual_rule_met
    if status not in ("no_image", "unprocessed_image", "zero_detections", "detections_present"):
        raise ValueError("Unknown assessment image status")
    if score is not None:
        if type(score) not in (float, int) or not 0 <= score <= 1:
            raise ValueError("Fall Armyworm confidence must be a finite number from 0 to 1")
        if status != "detections_present":
            raise ValueError("A detection score requires a processed image with detections")
    if met is not None and type(met) is not bool:
        raise ValueError("visual_rule_met must be a boolean or None")
    if met and (score is None or assessment.risk_level == "Low"):
        raise ValueError("A visual rule match requires a score and at least Moderate risk")
    if assessment.risk_level == "High" and (score is None or met is False):
        raise ValueError("High risk requires a qualifying Fall Armyworm detection")
    if status != "detections_present":
        return status
    if score is None:
        return "other_classes"
    if met is False:
        return "weak_sign"
    # With older results the rule outcome may be unknown. Describe only the photo.
    return "possible_sign"


def _message(language: str, level: str, evidence: str) -> str:
    heading, action = _ACTIONS[language][level]
    text = " ".join((heading, _EVIDENCE[language][evidence], action))
    if len(text) > 160:
        raise ValueError(f"{language} SMS template exceeds 160 characters")
    if not _SMS_ALPHABET.issuperset(text):
        raise ValueError(f"{language} SMS contains characters outside the one-septet template alphabet")
    return text


def build_advisory(assessment: RiskAssessment) -> FarmerAdvisory:
    """Render both draft messages using the evidence from one WP 2.2 assessment."""
    if not isinstance(assessment, RiskAssessment):
        raise TypeError("assessment must be a forecasting_agent.RiskAssessment")
    if assessment.risk_level not in ("Low", "Moderate", "High"):
        raise ValueError("Unknown risk level")
    if type(assessment.telemetry_hours) is not int or assessment.telemetry_hours != 72:
        raise ValueError("A complete 72-hour assessment is required")
    if type(assessment.matching_hours) is not int or not 0 <= assessment.matching_hours <= 72:
        raise ValueError("matching_hours must be an integer from 0 to 72")
    evidence = _evidence_key(assessment)
    return FarmerAdvisory(
        assessment,
        _message("en", assessment.risk_level, evidence),
        _message("lg", assessment.risk_level, evidence),
    )
