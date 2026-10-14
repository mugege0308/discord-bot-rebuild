"""Production interaction-profile ledger and Phase 1 release gate.

The ledger is owned by the Backend.  This module intentionally exposes only
the small prompt-only surface released in Phase 1; profile details for future
IPAdapter or framing modes remain unavailable to the production request path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


LEDGER_VERSION = 1
_DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "interaction_profiles"
LEDGER_PATH = _DATA_ROOT / "interaction_profiles.json"
RELEASE_STATUS_PATH = _DATA_ROOT / "release_status.json"

PHASE1_RELEASED_IDS = (
    "holding_hands",
    "dancing_together",
    "hand_on_shoulder",
    "arm_around_shoulder",
    "leaning_on_partner",
)


class InteractionProfileError(ValueError):
    """Base error with a stable client-facing error code."""

    code = "interaction_profile_schema_invalid"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code


class InteractionProfileNotFound(InteractionProfileError):
    code = "interaction_not_found"


class InteractionProfileNotReleased(InteractionProfileError):
    code = "interaction_not_released"


class InteractionProfileNotReady(InteractionProfileError):
    code = "interaction_not_ready"


class InteractionProfileModeUnsupported(InteractionProfileError):
    code = "interaction_mode_unsupported"


@dataclass(frozen=True)
class InteractionProfile:
    id: str
    display_name: str
    status: str
    mode: str
    classification: str
    ready: bool
    ledger_version: int
    prompt_additions: tuple[str, ...]
    negative_additions: tuple[str, ...]

    def public_dict(self) -> dict[str, object]:
        """Return only values safe for the Gallery profile-listing API."""

        return {
            "id": self.id,
            "display_name": self.display_name,
            "mode": self.mode,
            "classification": self.classification,
        }

    def metadata(self) -> dict[str, object]:
        return {
            "requested_id": self.id,
            "profile_id": self.id,
            "classification": self.classification,
            "mode": self.mode,
            "ledger_version": self.ledger_version,
        }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise InteractionProfileError(
            f"interaction profile ledgerを読み込めません: {path.name}"
        ) from exc
    if not isinstance(value, dict):
        raise InteractionProfileError("interaction profile ledgerの形式が不正です。")
    return value


def _dedupe_tags(values: object) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise InteractionProfileError("interaction profileのtag形式が不正です。")
        tag = value.strip()
        key = " ".join(tag.casefold().replace("_", " ").split())
        if tag and key and key not in seen:
            result.append(tag)
            seen.add(key)
    return tuple(result)


def _raw_profile(profile_id: str) -> tuple[dict[str, Any], dict[str, Any], int]:
    ledger = _load_json(LEDGER_PATH)
    version = ledger.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise InteractionProfileError("interaction profile ledger versionが不正です。")
    profiles = ledger.get("profiles")
    if not isinstance(profiles, dict):
        raise InteractionProfileError("interaction profile ledger profilesが不正です。")
    raw = profiles.get(profile_id)
    if not isinstance(raw, dict):
        raise InteractionProfileNotFound(f"interaction profileが見つかりません: {profile_id}")
    if str(raw.get("id") or "") != profile_id:
        raise InteractionProfileError("interaction profile idがledger keyと一致しません。")

    release_manifest = _load_json(RELEASE_STATUS_PATH)
    release_version = release_manifest.get("version")
    if release_version != version:
        raise InteractionProfileError("interaction profile ledger/release versionが一致しません。")
    release_profiles = release_manifest.get("profiles")
    if not isinstance(release_profiles, dict):
        raise InteractionProfileError("interaction profile release manifestが不正です。")
    release = release_profiles.get(profile_id)
    if not isinstance(release, dict):
        raise InteractionProfileError("interaction profile release情報が見つかりません。")
    return raw, release, version


def resolve_interaction_profile(interaction_id: str) -> InteractionProfile:
    """Resolve an exact profile id without aliases or fallback."""

    if not isinstance(interaction_id, str) or not interaction_id.strip():
        raise InteractionProfileNotFound("interaction_idを指定してください。")
    profile_id = interaction_id.strip()
    raw, release, version = _raw_profile(profile_id)
    required = ("display_name", "status", "mode")
    if any(not isinstance(raw.get(key), str) or not str(raw.get(key)).strip() for key in required):
        raise InteractionProfileError("interaction profile required fieldが不正です。")
    classification = release.get("classification")
    ready = release.get("ready")
    if not isinstance(classification, str) or not isinstance(ready, bool):
        raise InteractionProfileError("interaction profile release fieldが不正です。")
    return InteractionProfile(
        id=profile_id,
        display_name=str(raw["display_name"]).strip(),
        status=str(raw["status"]).strip(),
        mode=str(raw["mode"]).strip(),
        classification=classification.strip(),
        ready=ready,
        ledger_version=version,
        prompt_additions=_dedupe_tags(raw.get("prompt_additions", ())),
        negative_additions=_dedupe_tags(raw.get("negative_additions", ())),
    )


def resolve_phase1_interaction(interaction_id: str) -> InteractionProfile:
    """Resolve and validate a production-released Phase 1 profile."""

    profile = resolve_interaction_profile(interaction_id)
    if not profile.ready:
        raise InteractionProfileNotReady(f"interaction profileは未準備です: {profile.id}")
    if profile.classification != "production_candidate" or profile.id not in PHASE1_RELEASED_IDS:
        raise InteractionProfileNotReleased(f"interaction profileはProduction未公開です: {profile.id}")
    if profile.mode != "prompt_only":
        raise InteractionProfileModeUnsupported(f"interaction modeはPhase 1未対応です: {profile.mode}")
    if profile.status != "validated":
        raise InteractionProfileNotReleased(f"interaction profileは未検証です: {profile.id}")
    return profile


def list_phase1_profiles(*, released: str = "production_candidate") -> list[dict[str, object]]:
    """Return the safe, selectable Phase 1 profile catalog."""

    if released != "production_candidate":
        return []
    result: list[dict[str, object]] = []
    for profile_id in PHASE1_RELEASED_IDS:
        result.append(resolve_phase1_interaction(profile_id).public_dict())
    return result


def apply_prompt_only_profile(
    profile: InteractionProfile,
    positive: str,
    negative: str,
) -> tuple[str, str]:
    """Append profile additions without replacing existing prompt layers."""

    def merge(base: object, additions: tuple[str, ...]) -> str:
        tokens = [part.strip() for part in str(base or "").split(",") if part.strip()]
        seen = {" ".join(token.casefold().replace("_", " ").split()) for token in tokens}
        for addition in additions:
            key = " ".join(addition.casefold().replace("_", " ").split())
            if addition and key not in seen:
                tokens.append(addition)
                seen.add(key)
        return ", ".join(tokens)

    return merge(positive, profile.prompt_additions), merge(negative, profile.negative_additions)


def interaction_metadata(interaction_id: str | None) -> dict[str, object] | None:
    if interaction_id is None:
        return None
    return resolve_phase1_interaction(interaction_id).metadata()


__all__ = [
    "InteractionProfile",
    "InteractionProfileError",
    "InteractionProfileModeUnsupported",
    "InteractionProfileNotFound",
    "InteractionProfileNotReady",
    "InteractionProfileNotReleased",
    "LEDGER_PATH",
    "LEDGER_VERSION",
    "PHASE1_RELEASED_IDS",
    "RELEASE_STATUS_PATH",
    "apply_prompt_only_profile",
    "interaction_metadata",
    "list_phase1_profiles",
    "resolve_interaction_profile",
    "resolve_phase1_interaction",
]
