"""Shared multi-person framing choices and prompt-designer guidance."""

from __future__ import annotations

from dataclasses import dataclass


MULTI_PERSON_FRAMING_CHOICES = (
    "auto",
    "both_faces",
    "upper_body",
    "cowboy",
    "full_body",
    "partial_partner_ok",
)
MULTI_PERSON_FRAMING_LABELS = {
    "auto": "Auto",
    "both_faces": "Both faces",
    "upper_body": "Upper body",
    "cowboy": "Cowboy shot",
    "full_body": "Full body",
    "partial_partner_ok": "Partial partner OK",
}

# These are existing verified Animagine vocabulary entries.  Semantic modes
# deliberately have no invented Positive tag.
MULTI_PERSON_FRAMING_TAGS = {
    "upper_body": "upper body",
    "cowboy": "cowboy shot",
    "full_body": "full body",
}


@dataclass(frozen=True)
class MultiPersonFramingDecision:
    requested: str = "auto"
    effective: str = "auto"
    source: str = "system_default"
    explicit_tags: tuple[str, ...] = ()
    preferred_tag: str = ""

    @property
    def is_semantic(self) -> bool:
        return self.effective in {"both_faces", "partial_partner_ok"}


def validate_multi_person_framing(value: object) -> str:
    key = str(value or "auto").strip().casefold()
    if key not in MULTI_PERSON_FRAMING_CHOICES:
        raise ValueError("multi_person_framingはauto、both_faces、upper_body、cowboy、full_body、partial_partner_okのいずれかを指定してください。")
    return key


def build_multi_person_framing_instruction(mode: object) -> str:
    """Return a bounded Prompt Designer instruction for the selected mode."""

    selected = validate_multi_person_framing(mode)
    if selected == "auto":
        return ""
    details = {
        "both_faces": (
            "When the request requires multiple people, keep both people's faces visible when reasonably possible. "
            "This is a semantic constraint, not a request to invent a new Positive tag."
        ),
        "upper_body": "When the request requires multiple people, prefer an upper-body framing for the people.",
        "cowboy": "When the request requires multiple people, prefer a cowboy-shot framing for the people.",
        "full_body": "When the request requires multiple people, show the people full body when reasonably possible.",
        "partial_partner_ok": (
            "When the request requires multiple people, the partner may be partially cropped if needed. "
            "Do not turn this into a solo composition or force both faces to be visible."
        ),
    }
    return (
        "Multi-person framing selector: apply this only when the request semantically requires multiple people. "
        "An explicit framing instruction in the user's request has priority over this selector. "
        + details[selected]
    )


__all__ = [
    "MULTI_PERSON_FRAMING_CHOICES",
    "MULTI_PERSON_FRAMING_LABELS",
    "MULTI_PERSON_FRAMING_TAGS",
    "MultiPersonFramingDecision",
    "build_multi_person_framing_instruction",
    "validate_multi_person_framing",
]
