import os
from dataclasses import dataclass
from pathlib import Path

from services.sd_control import COMFY_CHECKPOINTS


DEFAULT_IMAGE_GENERATION_MODEL_KEY = "animagine"
QWEN_IMAGE_MODEL_KEY = "qwen_image_2_1"
QWEN_IMAGE_ENGINE = "qwen_image"
QWEN_IMAGE_FEATURE_FLAG = "ENABLE_QWEN_IMAGE_2_1"


@dataclass(frozen=True)
class ImageGenerationProfile:
    model_key: str
    display_name: str
    checkpoint: str | None
    prompt_designer_instruction: Path | None
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
    # The engine/capability fields are additive.  Existing callers continue
    # to use the SDXL defaults, while non-SDXL engines can select their own
    # prompt/workflow contract without changing the public model_key shape.
    engine: str = "sdxl_tag"
    supports_lora: bool = True
    supports_reference: bool = True
    supports_pose: bool = True
    supports_bulk: bool = True
    supports_overnight: bool = True
    supported_sizes: tuple[str, ...] = ("square", "portrait", "landscape")
    default_size: str = "square"
    default_steps: int | None = 20
    default_cfg: float | None = 7.0
    default_sampler: str | None = "Euler a"
    default_scheduler: str | None = "normal"
    workflow_key: str | None = None
    model_components: tuple[tuple[str, str], ...] = ()


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
    QWEN_IMAGE_MODEL_KEY: ImageGenerationProfile(
        model_key=QWEN_IMAGE_MODEL_KEY,
        display_name="Qwen-Image 2.1 (W4A8)",
        checkpoint=None,
        prompt_designer_instruction=None,
        filename_prefix="Qwen21",
        model_family="qwen_image",
        engine=QWEN_IMAGE_ENGINE,
        supports_lora=False,
        supports_reference=False,
        supports_pose=False,
        supports_bulk=False,
        supports_overnight=False,
        supported_sizes=("square",),
        default_size="square",
        width=512,
        height=512,
        steps=8,
        cfg=1.0,
        sampler="euler",
        scheduler="simple",
        default_steps=8,
        default_cfg=1.0,
        default_sampler="euler",
        default_scheduler="simple",
        workflow_key="qwen_image_2_1_w4a8",
        model_components=(
            ("diffusion_model", "qwen_image_2.1_int8_convrot.safetensors"),
            ("text_encoder", "qwen3vl_8b_w4a8.safetensors"),
            ("vae", "qwen_image_2.1_vae_bf16.safetensors"),
        ),
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


def is_image_generation_profile_enabled(profile: ImageGenerationProfile) -> bool:
    """Return whether a profile may be exposed by the standalone API.

    Qwen is deliberately opt-in so a production backend still behaves exactly
    as before when it points at the older ComfyUI instance.
    """

    if profile.engine != QWEN_IMAGE_ENGINE:
        return True
    value = os.getenv(QWEN_IMAGE_FEATURE_FLAG, "0").strip().casefold()
    return value in {"1", "true", "yes", "on"}


def enabled_image_generation_profiles() -> tuple[ImageGenerationProfile, ...]:
    return tuple(
        profile
        for profile in IMAGE_GENERATION_PROFILES.values()
        if is_image_generation_profile_enabled(profile)
    )


def profile_model_components(profile: ImageGenerationProfile) -> dict[str, str]:
    return {key: value for key, value in profile.model_components}


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
