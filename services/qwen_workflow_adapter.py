"""Qwen-Image 2.1 W4A8 ComfyUI API workflow adapter."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from services.runtime_paths import project_path, resolve_project_path


QWEN_WORKFLOW_FILE = project_path("workflows", "qwen_image_2_1_w4a8.json")
QWEN_REQUIRED_NODE_TYPES = frozenset(
    {
        "UNETLoader",
        "QwenImage21Cache",
        "CLIPLoader",
        "VAELoader",
        "TextEncodeQwenImage21",
        "EmptyLatentImage",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    }
)
QWEN_REQUIRED_RUNTIME_NODES = frozenset({"TextEncodeQwenImage21", "QwenImage21Cache"})
QWEN_MODEL_COMPONENTS = {
    "diffusion_model": "qwen_image_2.1_int8_convrot.safetensors",
    "text_encoder": "qwen3vl_8b_w4a8.safetensors",
    "vae": "qwen_image_2.1_vae_bf16.safetensors",
}


class QwenWorkflowError(RuntimeError):
    """Raised when a Qwen workflow or ComfyUI capability is unsafe to use."""


def _nodes_by_class(workflow: Mapping[str, Any], class_type: str) -> list[str]:
    return [
        str(node_id)
        for node_id, node in workflow.items()
        if isinstance(node, Mapping) and str(node.get("class_type", "")) == class_type
    ]


def _single_node(workflow: Mapping[str, Any], class_type: str) -> str:
    node_ids = _nodes_by_class(workflow, class_type)
    if len(node_ids) != 1:
        raise QwenWorkflowError(
            f"Qwen workflowには{class_type}が1個必要です（検出数: {len(node_ids)}）。"
        )
    return node_ids[0]


def _inputs(workflow: Mapping[str, Any], node_id: str) -> dict[str, Any]:
    node = workflow.get(str(node_id))
    if not isinstance(node, Mapping) or not isinstance(node.get("inputs"), dict):
        raise QwenWorkflowError(f"Qwen workflowのnode {node_id}にinputsがありません。")
    return node["inputs"]


def validate_qwen_workflow(workflow: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(workflow, Mapping) or not workflow:
        raise QwenWorkflowError("Qwen workflowが空です。")
    for node_id, node in workflow.items():
        if not isinstance(node, Mapping) or not isinstance(node.get("class_type"), str):
            raise QwenWorkflowError(f"Qwen workflowのnode {node_id}が不正です。")
    missing = sorted(
        class_type
        for class_type in QWEN_REQUIRED_NODE_TYPES
        if not _nodes_by_class(workflow, class_type)
    )
    if missing:
        raise QwenWorkflowError(f"Qwen workflowに必要なnodeがありません: {', '.join(missing)}")
    return {class_type: _single_node(workflow, class_type) for class_type in QWEN_REQUIRED_NODE_TYPES}


def load_qwen_workflow(path: Path | None = None) -> tuple[dict[str, Any], Path]:
    workflow_file = resolve_project_path(path or QWEN_WORKFLOW_FILE, QWEN_WORKFLOW_FILE)
    if not workflow_file.is_file():
        raise QwenWorkflowError(f"Qwen workflow fileが見つかりません: {workflow_file}")
    try:
        workflow = json.loads(workflow_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QwenWorkflowError("Qwen workflow JSONを読み込めませんでした。") from exc
    validate_qwen_workflow(workflow)
    return workflow, workflow_file


def validate_qwen_runtime_nodes(object_info: Mapping[str, Any]) -> None:
    if not isinstance(object_info, Mapping):
        raise QwenWorkflowError("ComfyUI /object_info の応答が不正です。")
    missing = sorted(node for node in QWEN_REQUIRED_RUNTIME_NODES if node not in object_info)
    if missing:
        raise QwenWorkflowError(
            "ComfyUI-NextにQwen用nodeがありません: " + ", ".join(missing)
        )


def patch_qwen_workflow(
    workflow: Mapping[str, Any],
    *,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int = 8,
    cfg: float = 1.0,
    sampler: str = "euler",
    scheduler: str = "simple",
    output_prefix: str = "Qwen21",
    components: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a patched copy, leaving the checked-in template untouched."""

    patched = copy.deepcopy(dict(workflow))
    node_ids = validate_qwen_workflow(patched)
    model_components = dict(QWEN_MODEL_COMPONENTS)
    model_components.update(dict(components or {}))

    _inputs(patched, node_ids["UNETLoader"])["unet_name"] = model_components["diffusion_model"]
    _inputs(patched, node_ids["CLIPLoader"])["clip_name"] = model_components["text_encoder"]
    _inputs(patched, node_ids["CLIPLoader"])["type"] = "qwen_image"
    _inputs(patched, node_ids["VAELoader"])["vae_name"] = model_components["vae"]

    encoder = _inputs(patched, node_ids["TextEncodeQwenImage21"])
    encoder["prompt"] = str(prompt)
    encoder["negative_prompt"] = ""
    encoder["resolution"] = int(max(width, height))

    latent = _inputs(patched, node_ids["EmptyLatentImage"])
    latent["width"] = int(width)
    latent["height"] = int(height)
    latent["batch_size"] = 1

    sampler_inputs = _inputs(patched, node_ids["KSampler"])
    sampler_inputs["seed"] = int(seed)
    sampler_inputs["steps"] = int(steps)
    sampler_inputs["cfg"] = float(cfg)
    sampler_inputs["sampler_name"] = str(sampler)
    sampler_inputs["scheduler"] = str(scheduler)

    _inputs(patched, node_ids["SaveImage"])["filename_prefix"] = str(output_prefix)
    return patched
