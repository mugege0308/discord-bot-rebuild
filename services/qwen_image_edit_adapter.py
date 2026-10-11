"""Qwen-Image 2.1 single-reference edit workflow adapter.

This adapter is intentionally separate from the existing text-to-image
workflow.  The 2.1 encoder exposes an image slot (``image_1``) and returns
positive, negative, and reference-sized latent outputs, so it can edit one
reference without using the 2509/2511 edit models.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from services.qwen_workflow_adapter import QWEN_MODEL_COMPONENTS, QwenWorkflowError
from services.runtime_paths import project_path, resolve_project_path


QWEN_EDIT_WORKFLOW_FILE = project_path("workflows", "qwen_image_2_1_edit_w4a8.json")
QWEN_EDIT_ENCODER_NODE = "TextEncodeQwenImage21"
QWEN_EDIT_REQUIRED_NODE_TYPES = frozenset(
    {
        "UNETLoader",
        "QwenImage21Cache",
        "CLIPLoader",
        "VAELoader",
        "LoadImage",
        QWEN_EDIT_ENCODER_NODE,
        "KSampler",
        "VAEDecode",
        "SaveImage",
    }
)
# These are the image-conditioning nodes shipped by ComfyUI-Next.  The
# native 2.1 encoder is preferred for the existing 2.1 W4A8 checkpoint; the
# edit-specific nodes are accepted as a runtime capability for future
# workflow templates without exposing them to the Production UI yet.
QWEN_EDIT_RUNTIME_NODE_ALTERNATIVES = frozenset(
    {QWEN_EDIT_ENCODER_NODE, "TextEncodeQwenImageEdit", "TextEncodeQwenImageEditPlus"}
)


def _nodes_by_class(workflow: Mapping[str, Any], class_type: str) -> list[str]:
    return [
        str(node_id)
        for node_id, node in workflow.items()
        if isinstance(node, Mapping) and str(node.get("class_type", "")) == class_type
    ]


def _single_node(workflow: Mapping[str, Any], class_type: str) -> str:
    ids = _nodes_by_class(workflow, class_type)
    if len(ids) != 1:
        raise QwenWorkflowError(
            f"Qwen edit workflowには{class_type}が1個必要です（検出数: {len(ids)}）。"
        )
    return ids[0]


def _inputs(workflow: Mapping[str, Any], node_id: str) -> dict[str, Any]:
    node = workflow.get(str(node_id))
    if not isinstance(node, Mapping) or not isinstance(node.get("inputs"), dict):
        raise QwenWorkflowError(f"Qwen edit workflowのnode {node_id}にinputsがありません。")
    return node["inputs"]


def validate_qwen_edit_workflow(workflow: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(workflow, Mapping) or not workflow:
        raise QwenWorkflowError("Qwen edit workflowが空です。")
    for node_id, node in workflow.items():
        if not isinstance(node, Mapping) or not isinstance(node.get("class_type"), str):
            raise QwenWorkflowError(f"Qwen edit workflowのnode {node_id}が不正です。")
    missing = sorted(
        class_type
        for class_type in QWEN_EDIT_REQUIRED_NODE_TYPES
        if not _nodes_by_class(workflow, class_type)
    )
    if missing:
        raise QwenWorkflowError("Qwen edit workflowに必要なnodeがありません: " + ", ".join(missing))
    return {class_type: _single_node(workflow, class_type) for class_type in QWEN_EDIT_REQUIRED_NODE_TYPES}


def load_qwen_edit_workflow(path: Path | None = None) -> tuple[dict[str, Any], Path]:
    workflow_file = resolve_project_path(path or QWEN_EDIT_WORKFLOW_FILE, QWEN_EDIT_WORKFLOW_FILE)
    if not workflow_file.is_file():
        raise QwenWorkflowError(f"Qwen edit workflow fileが見つかりません: {workflow_file}")
    try:
        workflow = json.loads(workflow_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QwenWorkflowError("Qwen edit workflow JSONを読み込めませんでした。") from exc
    validate_qwen_edit_workflow(workflow)
    return workflow, workflow_file


def validate_qwen_edit_runtime_nodes(object_info: Mapping[str, Any]) -> None:
    if not isinstance(object_info, Mapping):
        raise QwenWorkflowError("ComfyUI /object_info の応答が不正です。")
    required = {"QwenImage21Cache", QWEN_EDIT_ENCODER_NODE}
    missing = sorted(node for node in required if node not in object_info)
    if missing:
        alternatives = sorted(node for node in QWEN_EDIT_RUNTIME_NODE_ALTERNATIVES if node in object_info)
        if alternatives:
            raise QwenWorkflowError(
                "Qwen 2.1 edit用のnative nodeがありません: "
                + ", ".join(missing)
                + "（利用可能な候補: "
                + ", ".join(alternatives)
                + "）"
            )
        raise QwenWorkflowError("ComfyUI-NextにQwen edit用nodeがありません: " + ", ".join(missing))


def patch_qwen_edit_workflow(
    workflow: Mapping[str, Any],
    *,
    reference_image_name: str,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int = 8,
    cfg: float = 1.0,
    sampler: str = "euler",
    scheduler: str = "simple",
    output_prefix: str = "Qwen21Edit",
    components: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Patch a copy of the checked-in single-reference workflow."""

    patched = copy.deepcopy(dict(workflow))
    node_ids = validate_qwen_edit_workflow(patched)
    model_components = dict(QWEN_MODEL_COMPONENTS)
    model_components.update(dict(components or {}))

    _inputs(patched, node_ids["UNETLoader"])["unet_name"] = model_components["diffusion_model"]
    clip_inputs = _inputs(patched, node_ids["CLIPLoader"])
    clip_inputs["clip_name"] = model_components["text_encoder"]
    clip_inputs["type"] = "qwen_image"
    _inputs(patched, node_ids["VAELoader"])["vae_name"] = model_components["vae"]
    _inputs(patched, node_ids["LoadImage"])["image"] = str(reference_image_name)

    encoder = _inputs(patched, node_ids[QWEN_EDIT_ENCODER_NODE])
    encoder["prompt"] = str(prompt)
    encoder["negative_prompt"] = ""
    encoder["resolution"] = int(max(width, height))
    # ComfyUI 0.39's Autogrow input is normalized as one ``images`` mapping
    # before calling TextEncodeQwenImage21.execute(images=...).  Sending a
    # flat ``image_1`` key reaches the Python method as an unexpected kwarg.
    encoder["images"] = {"image_1": [node_ids["LoadImage"], 0]}

    sampler_inputs = _inputs(patched, node_ids["KSampler"])
    sampler_inputs["seed"] = int(seed)
    sampler_inputs["steps"] = int(steps)
    sampler_inputs["cfg"] = float(cfg)
    sampler_inputs["sampler_name"] = str(sampler)
    sampler_inputs["scheduler"] = str(scheduler)
    sampler_inputs["positive"] = [node_ids[QWEN_EDIT_ENCODER_NODE], 0]
    sampler_inputs["negative"] = [node_ids[QWEN_EDIT_ENCODER_NODE], 1]
    sampler_inputs["latent_image"] = [node_ids[QWEN_EDIT_ENCODER_NODE], 2]
    _inputs(patched, node_ids["SaveImage"])["filename_prefix"] = str(output_prefix)
    return patched
