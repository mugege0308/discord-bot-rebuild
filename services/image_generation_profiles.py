from dataclasses import dataclass
from pathlib import Path

from services.sd_control import COMFY_CHECKPOINTS


DEFAULT_IMAGE_GENERATION_MODEL_KEY = "animagine"


@dataclass(frozen=True)
class ImageGenerationProfile:
    model_key: str
    display_name: str
    checkpoint: str
    prompt_designer_instruction: Path
    filename_prefix: str
    positive_prefix: str | None = None
    positive_suffix: str | None = None
    negative_default: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg: float | None = None
    sampler: str | None = None
    scheduler: str | None = None
    denoise: float | None = None


IMAGE_GENERATION_PROFILES: dict[str, ImageGenerationProfile] = {
    "animagine": ImageGenerationProfile(
        model_key="animagine",
        display_name="Animagine XL 4.0 Clear",
        checkpoint=COMFY_CHECKPOINTS["animagine"],
        prompt_designer_instruction=Path("shogou_chan_prompt_designer_instruction.md"),
        filename_prefix="Shogou_AnimagineXL",
    ),
    "janku": ImageGenerationProfile(
        model_key="janku",
        display_name="JANKU v7.77",
        checkpoint=COMFY_CHECKPOINTS["janku"],
        prompt_designer_instruction=Path("shogou_chan_prompt_designer_instruction_janku.md"),
        filename_prefix="Shogou_JANKU",
    ),
}


def get_image_generation_profile(model_key: str | None = None) -> ImageGenerationProfile:
    resolved_key = model_key or DEFAULT_IMAGE_GENERATION_MODEL_KEY
    try:
        return IMAGE_GENERATION_PROFILES[resolved_key]
    except KeyError as exc:
        raise ValueError("指定された画像生成モデルには対応していません。") from exc


def get_model_key_for_checkpoint(checkpoint: str | None) -> str | None:
    if not checkpoint:
        return None
    matches = [
        profile.model_key
        for profile in IMAGE_GENERATION_PROFILES.values()
        if profile.checkpoint == checkpoint
    ]
    return matches[0] if len(matches) == 1 else None
