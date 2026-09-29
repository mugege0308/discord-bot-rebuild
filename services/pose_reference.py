"""Quality-gated pose-reference helpers for the ComfyUI backend.

The bot deliberately does not import cv2 here.  Pose estimation and drawing
remain ComfyUI custom-node work; this module only builds the small diagnostic
graphs and validates the keypoint evidence returned by ComfyUI history.
"""

from __future__ import annotations

import json
import math
from typing import Any


REFERENCE_MODE_POSE = "pose"
DEFAULT_POSE_STRENGTH = 0.7
DEFAULT_POSE_START_PERCENT = 0.0
DEFAULT_POSE_END_PERCENT = 0.8
DEFAULT_POSE_DETECTION_RESOLUTION = 1024
DEFAULT_POSE_FIT_MODE = "contain"
POSE_REFERENCE_PROFILE_FULL_BODY = "full_body_reference"
POSE_REFERENCE_PROFILE_UPPER_BODY = "upper_body_reference"
POSE_REFERENCE_PROFILE_UNKNOWN = "unknown"
POSE_FIT_MODES = {"contain", "cover", "stretch"}
POSE_BBOX_DETECTOR_NAME = "yolox_l.onnx"
POSE_CONTROLNET_NAME = "controlnet-openpose-sdxl-1.0.safetensors"
POSE_PREPROCESSOR_NAME = "dw-ll_ucoco_384.onnx"
POSE_SDPOSE_CHECKPOINT_NAME = "sdpose_wholebody_fp16.safetensors"
POSE_SDPOSE_DRAW_THRESHOLD = 0.2
POSE_SDPOSE_MIN_DRAWN_BODY_KEYPOINTS = 10
POSE_SDPOSE_MIN_COMPLETED_CHAINS = 4
POSE_ANIMEPOSE_CONFIDENCE = 0.15
POSE_ANIMEPOSE_MASK_GATE = True
POSE_ANIMEPOSE_MASK_TOLERANCE = 0.04
POSE_ANIMEPOSE_BORDER_MARGIN = 0.0
POSE_ANIMEPOSE_BBOX_TRIM = 0.05
POSE_ANIMEPOSE_MIN_VALID_BODY_KEYPOINTS = 15
POSE_ANIMEPOSE_MIN_COMPLETED_CHAINS = 4
POSE_UPPER_BODY_MIN_HIP_NORMALIZED_Y = 0.85
POSE_UPPER_BODY_MAX_DETECTED_LOWER_LIMB_POINTS = 1
POSE_UPPER_BODY_REQUIRED_ARM_CHAINS = ("left_arm", "right_arm")

SDPOSE_BODY_KEYPOINT_NAMES = (
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip",
    "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
)
ANIMEPOSE_OPENPOSE18_NAMES = SDPOSE_BODY_KEYPOINT_NAMES
POSE_QUALITY_LOG_JOINTS = (
    "neck", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
POSE_CHAIN_DEFINITIONS = {
    "left_arm": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_arm": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_leg": ("left_hip", "left_knee"),
    "right_leg": ("right_hip", "right_knee"),
    "torso": ("neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip"),
}


def pose_fit_plan(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    fit_mode: str = DEFAULT_POSE_FIT_MODE,
) -> dict[str, int | str]:
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("pose referenceと生成canvasの画像サイズが不正です。")
    if fit_mode not in POSE_FIT_MODES:
        raise ValueError(f"未対応のpose fit modeです: {fit_mode}")
    if fit_mode == "contain":
        scale = min(target_width / source_width, target_height / source_height)
        resize_width = max(1, min(target_width, round(source_width * scale)))
        resize_height = max(1, min(target_height, round(source_height * scale)))
        left = (target_width - resize_width) // 2
        top = (target_height - resize_height) // 2
        return {
            "fit_mode": fit_mode, "resize_width": resize_width, "resize_height": resize_height,
            "left": left, "top": top,
            "right": target_width - resize_width - left,
            "bottom": target_height - resize_height - top,
            "crop_left": 0, "crop_top": 0, "crop_right": 0, "crop_bottom": 0,
        }
    if fit_mode == "cover":
        scale = max(target_width / source_width, target_height / source_height)
        resize_width = max(target_width, round(source_width * scale))
        resize_height = max(target_height, round(source_height * scale))
        crop_left = max(0, (resize_width - target_width) // 2)
        crop_top = max(0, (resize_height - target_height) // 2)
        return {
            "fit_mode": fit_mode, "resize_width": resize_width, "resize_height": resize_height,
            "left": 0, "top": 0, "right": 0, "bottom": 0,
            "crop_left": crop_left, "crop_top": crop_top,
            "crop_right": max(0, resize_width - target_width - crop_left),
            "crop_bottom": max(0, resize_height - target_height - crop_top),
        }
    return {
        "fit_mode": fit_mode, "resize_width": target_width, "resize_height": target_height,
        "left": 0, "top": 0, "right": 0, "bottom": 0,
        "crop_left": 0, "crop_top": 0, "crop_right": 0, "crop_bottom": 0,
    }


def _point_present(point: dict[str, Any] | None, minimum_confidence: float) -> bool:
    if not isinstance(point, dict):
        return False
    try:
        x = float(point.get("x"))
        y = float(point.get("y"))
        confidence = float(point.get("confidence"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and math.isfinite(y) and math.isfinite(confidence) and confidence >= minimum_confidence


def pose_chain_quality(landmarks: dict[str, dict[str, Any]], *, minimum_confidence: float) -> dict[str, Any]:
    completed = {
        name: all(_point_present(landmarks.get(joint), minimum_confidence) for joint in joints)
        for name, joints in POSE_CHAIN_DEFINITIONS.items()
    }
    valid_count = sum(_point_present(point, minimum_confidence) for point in landmarks.values())
    confidences = [
        float(point.get("confidence"))
        for point in landmarks.values()
        if isinstance(point, dict)
        and isinstance(point.get("confidence"), (int, float))
        and math.isfinite(float(point["confidence"]))
    ]
    return {
        "completed_chains": [name for name, value in completed.items() if value],
        "completed_chain_count": sum(completed.values()),
        "chain_status": completed,
        "torso_complete": completed["torso"],
        "valid_body_keypoints": valid_count,
        "critical_joint_confidences": {name: landmarks.get(name, {}).get("confidence") for name in POSE_QUALITY_LOG_JOINTS},
        "quality_score": round(min(100.0, (sum(confidences) / len(confidences) * 50 if confidences else 0) + sum(completed.values()) / 5 * 35 + min(valid_count / 18, 1) * 15), 3),
    }


def _first_frame(raw_json: Any) -> dict[str, Any]:
    value = raw_json
    if isinstance(value, list):
        value = value[0] if value else {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _flatten_keypoints(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    if value and isinstance(value[0], (list, tuple)):
        result: list[Any] = []
        for row in value:
            if isinstance(row, (list, tuple)):
                result.extend(list(row)[:3])
        return result
    return value


def _landmarks_from_flat(body: list[Any], names: tuple[str, ...], threshold: float) -> dict[str, dict[str, Any]]:
    landmarks: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        offset = index * 3
        try:
            x = float(body[offset])
            y = float(body[offset + 1])
            confidence = float(body[offset + 2])
        except (IndexError, TypeError, ValueError):
            x = y = confidence = None
        landmarks[name] = {
            "x": x, "y": y, "confidence": confidence,
            "drawn": bool(confidence is not None and confidence >= threshold),
        }
    return landmarks


def sdpose_keypoint_summary(raw_json: Any, score_threshold: float = POSE_SDPOSE_DRAW_THRESHOLD) -> dict[str, Any]:
    frame = _first_frame(raw_json)
    people = frame.get("people", [])
    if not isinstance(people, list):
        people = []
    candidates: list[tuple[int, int, list[Any]]] = []
    for person in people:
        if not isinstance(person, dict):
            continue
        body = _flatten_keypoints(person.get("pose_keypoints_2d"))
        scores = [
            float(body[i]) for i in range(2, len(body), 3)
            if isinstance(body[i], (int, float))
        ]
        candidates.append((sum(score >= score_threshold for score in scores), sum(score > 0 for score in scores), body))
    _, valid_count, body = max(candidates, default=(0, 0, []), key=lambda item: (item[0], item[1]))
    scores = [float(body[i]) for i in range(2, len(body), 3) if isinstance(body[i], (int, float))]
    landmarks = _landmarks_from_flat(body, SDPOSE_BODY_KEYPOINT_NAMES, score_threshold)
    quality = pose_chain_quality(landmarks, minimum_confidence=score_threshold)
    return {
        "people": len(people), "body_keypoints_total": len(body) // 3,
        "body_keypoints_valid_gt0": valid_count,
        "body_keypoints_drawn": sum(score >= score_threshold for score in scores),
        "body_landmarks": landmarks,
        "canvas_width": frame.get("canvas_width"), "canvas_height": frame.get("canvas_height"),
        **quality,
    }


def animepose_keypoint_summary(keypoints18: Any, *, image_size: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = list(keypoints18) if isinstance(keypoints18, (list, tuple)) else []
    landmarks: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(ANIMEPOSE_OPENPOSE18_NAMES):
        row = rows[index] if index < len(rows) else ()
        try:
            x, y, confidence = float(row[0]), float(row[1]), float(row[2])
        except (IndexError, TypeError, ValueError):
            x = y = confidence = None
        landmarks[name] = {"x": x, "y": y, "confidence": confidence, "drawn": bool(confidence and confidence > 0)}
    quality = pose_chain_quality(landmarks, minimum_confidence=1e-6)
    return {
        "people": 1 if quality["valid_body_keypoints"] > 0 else 0,
        "body_keypoints_total": 18,
        "body_keypoints_valid": quality["valid_body_keypoints"],
        "image_width": image_size.get("width") if isinstance(image_size, dict) else None,
        "image_height": image_size.get("height") if isinstance(image_size, dict) else None,
        "body_landmarks": landmarks,
        **quality,
    }


def _upper_body_evaluation(summary: dict[str, Any], *, skeleton_stats: dict[str, Any] | None, minimum_confidence: float, full_body_pass: bool, allow_upper_body: bool) -> dict[str, Any]:
    landmarks = summary.get("body_landmarks", {}) if isinstance(summary, dict) else {}
    detected = {name: _point_present(landmarks.get(name), minimum_confidence) for name in POSE_QUALITY_LOG_JOINTS}
    height = summary.get("canvas_height") or summary.get("image_height")
    hip_y = {
        name: (float(landmarks[name]["y"]) / float(height) if detected[name] and height else None)
        for name in ("left_hip", "right_hip")
    }
    reasons: list[str] = []
    if not allow_upper_body:
        reasons.append("backend_upper_profile_disabled")
    if not skeleton_stats or skeleton_stats.get("complete_black", True):
        reasons.append("skeleton_complete_black")
    if summary.get("people", 0) < 1:
        reasons.append("people_lt_1")
    if not detected["neck"]:
        reasons.append("neck_missing")
    if not detected["left_shoulder"] or not detected["right_shoulder"]:
        reasons.append("both_shoulders_required")
    torso_possible = bool(
        detected["neck"]
        and detected["left_shoulder"]
        and detected["right_shoulder"]
        and (detected["left_hip"] or detected["right_hip"])
    )
    if not torso_possible:
        reasons.append("hip_missing")
    lower_limb_count = sum(detected[name] for name in ("left_knee", "right_knee", "left_ankle", "right_ankle"))
    if lower_limb_count > POSE_UPPER_BODY_MAX_DETECTED_LOWER_LIMB_POINTS:
        reasons.append(f"too_many_lower_limb_points={lower_limb_count}")
    usable_hips = [value for value in hip_y.values() if value is not None]
    if not usable_hips:
        reasons.append("hip_y_unavailable")
    elif min(usable_hips) < POSE_UPPER_BODY_MIN_HIP_NORMALIZED_Y:
        reasons.append("hip_not_in_lower_image=" + ",".join(f"{value:.3f}" for value in usable_hips))
    chain_status = summary.get("chain_status", {})
    arm_chains = {name: bool(chain_status.get(name, False)) for name in POSE_UPPER_BODY_REQUIRED_ARM_CHAINS}
    upper_pass = not reasons and all(arm_chains.values())
    if not upper_pass and not reasons:
        reasons.append("required_arm_chain_missing")
    return {
        "pose_quality_profile": "full_body" if full_body_pass else ("upper_body" if upper_pass else "full_body"),
        "upper_body_candidate": not reasons,
        "upper_body_candidate_reason": "pass" if not reasons else ",".join(reasons),
        "detected": detected,
        "hip_normalized_y": hip_y,
        "upper_body_arm_chains": arm_chains,
        "torso_possible": torso_possible,
        "upper_body_quality_pass": upper_pass,
        "upper_body_quality_reason": "pass" if upper_pass else (";".join(reasons) or "required_arm_chain_missing"),
        "detected_lower_limb_points": lower_limb_count,
        "full_body_pass": bool(full_body_pass),
        "full_body_leg_chain_evidence": bool(chain_status.get("left_leg") or chain_status.get("right_leg")),
        "pose_quality_pass": bool(full_body_pass or upper_pass),
    }


def evaluate_pose_quality(summary: dict[str, Any] | None, skeleton_stats: dict[str, Any] | None, *, backend: str) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {"success": False, "failure_reason": "summary_missing"}
    nonblack = bool(skeleton_stats and not skeleton_stats.get("complete_black", True))
    people = int(summary.get("people", 0) or 0)
    chains = int(summary.get("completed_chain_count", 0) or 0)
    torso = bool(summary.get("torso_complete"))
    if backend == "animepose":
        full_body = people >= 1 and nonblack and int(summary.get("valid_body_keypoints", 0) or 0) >= POSE_ANIMEPOSE_MIN_VALID_BODY_KEYPOINTS and chains >= POSE_ANIMEPOSE_MIN_COMPLETED_CHAINS and torso and bool(summary.get("chain_status", {}).get("left_leg") or summary.get("chain_status", {}).get("right_leg"))
        minimum = 1e-6
    elif backend == "sdpose":
        full_body = people >= 1 and nonblack and int(summary.get("body_keypoints_drawn", 0) or 0) >= POSE_SDPOSE_MIN_DRAWN_BODY_KEYPOINTS and chains >= POSE_SDPOSE_MIN_COMPLETED_CHAINS and torso and bool(summary.get("chain_status", {}).get("left_leg") or summary.get("chain_status", {}).get("right_leg"))
        minimum = POSE_SDPOSE_DRAW_THRESHOLD
    else:
        full_body = people >= 1 and nonblack and int(summary.get("valid_body_keypoints", summary.get("body_keypoints_valid_gt0", 0)) or 0) > 0 and chains >= 4 and torso and bool(summary.get("chain_status", {}).get("left_leg") or summary.get("chain_status", {}).get("right_leg"))
        minimum = POSE_SDPOSE_DRAW_THRESHOLD
    evaluation = _upper_body_evaluation(summary, skeleton_stats=skeleton_stats, minimum_confidence=minimum, full_body_pass=full_body, allow_upper_body=backend in {"sdpose", "animepose"})
    return {**summary, **evaluation, "backend": backend, "success": bool(evaluation["pose_quality_pass"]), "complete_black": not nonblack, "failure_reason": None if evaluation["pose_quality_pass"] else evaluation["upper_body_quality_reason"] if backend != "dwpose" else "dwpose_full_body_quality_failed"}


def classify_pose_reference_profile(result: dict[str, Any] | None, *, backend: str) -> tuple[str, str]:
    if not isinstance(result, dict):
        return POSE_REFERENCE_PROFILE_UNKNOWN, "pose_preflight_result_missing"
    if backend == "dwpose" and result.get("lower_limb_evidence_available") is not True:
        return POSE_REFERENCE_PROFILE_UNKNOWN, "dwpose_lower_limb_evidence_unavailable"
    if result.get("people", 0) >= 1 and result.get("complete_black") is False and result.get("full_body_pass") is True and result.get("full_body_leg_chain_evidence") is True:
        return POSE_REFERENCE_PROFILE_FULL_BODY, "full_body_pass_with_leg_chain"
    if result.get("full_body_pass") is False and result.get("upper_body_quality_pass") is True:
        return POSE_REFERENCE_PROFILE_UPPER_BODY, "upper_body_quality_pass_without_full_body"
    return POSE_REFERENCE_PROFILE_UNKNOWN, "full_body_and_upper_body_conditions_not_met"


def build_dwpose_preflight_workflow(image_name: str, *, detect_hand: bool = True, detect_face: bool = True) -> tuple[dict[str, Any], str, str, str]:
    workflow = {
        "1": {"inputs": {"image": image_name}, "class_type": "LoadImage", "_meta": {"title": "Pose preflight reference image"}},
        "2": {"inputs": {"image": ["1", 0], "detect_hand": "enable" if detect_hand else "disable", "detect_body": "enable", "detect_face": "enable" if detect_face else "disable", "resolution": DEFAULT_POSE_DETECTION_RESOLUTION, "bbox_detector": POSE_BBOX_DETECTOR_NAME, "pose_estimator": POSE_PREPROCESSOR_NAME, "scale_stick_for_xinsr_cn": "disable"}, "class_type": "DWPreprocessor", "_meta": {"title": "DWPose preflight"}},
        "3": {"inputs": {"images": ["2", 0]}, "class_type": "PreviewImage", "_meta": {"title": "DWPose preflight skeleton preview"}},
        "4": {"inputs": {"source": ["2", 1]}, "class_type": "PreviewAny", "_meta": {"title": "DWPose preflight raw keypoint metadata"}},
    }
    return workflow, "2", "3", "4"


def build_sdpose_preflight_workflow(image_name: str, filename_prefix: str) -> tuple[dict[str, Any], str, str, str]:
    workflow = {
        "1": {"inputs": {"image": image_name}, "class_type": "LoadImage", "_meta": {"title": "SDPose fallback reference image"}},
        "2": {"inputs": {"resize_type": "scale longer dimension", "resize_type.longer_size": DEFAULT_POSE_DETECTION_RESOLUTION, "scale_method": "area", "input": ["1", 0]}, "class_type": "ResizeImageMaskNode", "_meta": {"title": "SDPose fallback resize"}},
        "3": {"inputs": {"model": ["4", 0], "vae": ["4", 2], "image": ["2", 0], "bboxes": None, "batch_size": 16}, "class_type": "SDPoseKeypointExtractor", "_meta": {"title": "SDPose OOD fallback keypoint extractor"}},
        "4": {"inputs": {"ckpt_name": POSE_SDPOSE_CHECKPOINT_NAME}, "class_type": "CheckpointLoaderSimple", "_meta": {"title": "SDPose wholebody fallback checkpoint"}},
        "5": {"inputs": {"keypoints": ["3", 0], "draw_body": True, "draw_hands": True, "draw_face": True, "draw_feet": True, "draw_head": True, "stick_width": 4, "face_point_size": 2, "score_threshold": POSE_SDPOSE_DRAW_THRESHOLD}, "class_type": "SDPoseDrawKeypoints", "_meta": {"title": "SDPose OOD fallback skeleton"}},
        "6": {"inputs": {"images": ["5", 0]}, "class_type": "PreviewImage", "_meta": {"title": "SDPose fallback skeleton preview"}},
        "7": {"inputs": {"pose_kps": ["3", 0], "filename_prefix": filename_prefix}, "class_type": "SavePoseKpsAsJsonFile", "_meta": {"title": "SDPose fallback keypoint JSON"}},
    }
    return workflow, "3", "6", "7"


def build_animepose_preflight_workflow(image_name: str) -> tuple[dict[str, Any], str, str, str]:
    workflow = {
        "1": {"inputs": {"image": image_name}, "class_type": "LoadImage", "_meta": {"title": "AnimePose preflight reference image"}},
        "2": {"inputs": {"image": ["1", 0], "confidence": POSE_ANIMEPOSE_CONFIDENCE, "mask_gate": POSE_ANIMEPOSE_MASK_GATE, "mask_tolerance": POSE_ANIMEPOSE_MASK_TOLERANCE, "border_margin": POSE_ANIMEPOSE_BORDER_MARGIN, "bbox_trim": POSE_ANIMEPOSE_BBOX_TRIM}, "class_type": "AnimePoseEstimatorWithMetadata", "_meta": {"title": "AnimePose preflight with metadata"}},
        "3": {"inputs": {"images": ["2", 0]}, "class_type": "PreviewImage", "_meta": {"title": "AnimePose preflight skeleton preview"}},
        "4": {"inputs": {"source": ["2", 1]}, "class_type": "PreviewAny", "_meta": {"title": "AnimePose preflight metadata"}},
    }
    return workflow, "2", "3", "4"


def extract_json_candidates(value: Any) -> list[Any]:
    candidates: list[Any] = []
    values = value if isinstance(value, list) else [value]
    for item in values:
        if isinstance(item, (dict, list)):
            candidates.append(item)
        elif isinstance(item, str):
            try:
                candidates.append(json.loads(item))
            except json.JSONDecodeError:
                continue
    return candidates
