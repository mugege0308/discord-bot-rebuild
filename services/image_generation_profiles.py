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
    is_adult: bool = False
    prompt_designer_addendum: Path | None = None
    positive_prefix: str | None = None
    positive_suffix: str | None = None
    negative_default: str | None = None
    system_positive_tags: tuple[str, ...] = ()
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg: float | None = None
    sampler: str | None = None
    scheduler: str | None = None
    denoise: float | None = None
    model_family: str = "sdxl"


IMAGE_GENERATION_PROFILES: dict[str, ImageGenerationProfile] = {
    "animagine": ImageGenerationProfile(
        model_key="animagine",
        display_name="Animagine XL 4.0 Clear",
        checkpoint=COMFY_CHECKPOINTS["animagine"],
        prompt_designer_instruction=Path("shogou_chan_prompt_designer_instruction.md"),
        filename_prefix="Shogou_AnimagineXL",
        system_positive_tags=("masterpiece", "high score", "great score", "absurdres"),
    ),
    "animagine_adult": ImageGenerationProfile(
        model_key="animagine_adult",
        display_name="Animagine XL 4.0 Clear Adult",
        checkpoint=COMFY_CHECKPOINTS["animagine_adult"],
        prompt_designer_instruction=Path("shogou_chan_prompt_designer_instruction.md"),
        filename_prefix="Shogou_AnimagineXL_Adult",
        is_adult=True,
        prompt_designer_addendum=Path("shogou_chan_prompt_designer_instruction_animagine_adult.md"),
        system_positive_tags=("masterpiece", "high score", "great score", "absurdres"),
    ),
    "janku": ImageGenerationProfile(
        model_key="janku",
        display_name="JANKU v7.77",
        checkpoint=COMFY_CHECKPOINTS["janku"],
        prompt_designer_instruction=Path("shogou_chan_prompt_designer_instruction_janku.md"),
        filename_prefix="Shogou_JANKU",
    ),
    "janku_adult": ImageGenerationProfile(
        model_key="janku_adult",
        display_name="JANKU v7.77 Adult",
        checkpoint=COMFY_CHECKPOINTS["janku_adult"],
        prompt_designer_instruction=Path("shogou_chan_prompt_designer_instruction_janku_adult.md"),
        filename_prefix="Shogou_JANKU_Adult",
        is_adult=True,
        prompt_designer_addendum=None,
    ),
}

# A checkpoint cannot distinguish the regular and Adult prompt profiles.
# This mapping is used only for legacy rows whose model_key is NULL; newly
# saved rows retain their explicit profile key and are resolved before this.
LEGACY_CHECKPOINT_MODEL_KEYS = {
    COMFY_CHECKPOINTS["animagine"]: "animagine",
    COMFY_CHECKPOINTS["janku"]: "janku",
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
    legacy_key = LEGACY_CHECKPOINT_MODEL_KEYS.get(checkpoint)
    if legacy_key is None:
        return None
    profile = IMAGE_GENERATION_PROFILES.get(legacy_key)
    if profile is None or profile.checkpoint != checkpoint:
        return None
    return legacy_key
