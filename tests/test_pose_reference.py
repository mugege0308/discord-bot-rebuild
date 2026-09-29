import asyncio
import unittest
from unittest import mock

from discord import app_commands

import commands.prompt as prompt_commands
from services import pose_reference
from services.sd_control import (
    SdGenerationOptions,
    _apply_comfy_options,
    _pose_history_execution_error,
    _run_pose_pipeline,
    validate_options,
)


def _raw_sdpose(*, upper_body: bool = False, broken_left_arm: bool = False) -> dict:
    points = {
        "nose": (50, 10, 0.9),
        "neck": (50, 25, 0.9),
        "right_shoulder": (35, 30, 0.9),
        "right_elbow": (25, 45, 0.9),
        "right_wrist": (15, 60, 0.9),
        "left_shoulder": (65, 30, 0.9),
        "left_elbow": (75, 45, 0.0 if broken_left_arm else 0.9),
        "left_wrist": (85, 60, 0.9),
        "right_hip": (40, 90, 0.9),
        "right_knee": (38, 98, 0.0 if upper_body else 0.9),
        "right_ankle": (36, 99, 0.0),
        "left_hip": (60, 90, 0.9),
        "left_knee": (62, 98, 0.0 if upper_body else 0.9),
        "left_ankle": (64, 99, 0.0),
        "right_eye": (45, 15, 0.9),
        "left_eye": (55, 15, 0.9),
        "right_ear": (40, 18, 0.9),
        "left_ear": (60, 18, 0.9),
    }
    flat = []
    for name in pose_reference.SDPOSE_BODY_KEYPOINT_NAMES:
        flat.extend(points[name])
    return {"people": [{"pose_keypoints_2d": flat}], "canvas_width": 100, "canvas_height": 100}


def _workflow() -> dict:
    return {
        "positive": {"inputs": {"text": "old positive"}, "class_type": "CLIPTextEncode"},
        "negative": {"inputs": {"text": "old negative"}, "class_type": "CLIPTextEncode"},
        "latent": {
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
            "class_type": "EmptyLatentImage",
        },
        "sampler": {
            "inputs": {
                "positive": ["positive", 0],
                "negative": ["negative", 0],
                "latent_image": ["latent", 0],
                "seed": 1,
            },
            "class_type": "KSampler",
        },
        "save": {
            "inputs": {"images": ["decode", 0], "filename_prefix": "test"},
            "class_type": "SaveImage",
        },
    }


class PoseQualityTests(unittest.TestCase):
    def test_full_body_requires_nonblack_and_leg_evidence(self):
        summary = pose_reference.sdpose_keypoint_summary(_raw_sdpose())
        passed = pose_reference.evaluate_pose_quality(
            summary, {"complete_black": False}, backend="sdpose"
        )
        self.assertTrue(passed["full_body_pass"])
        self.assertTrue(passed["pose_quality_pass"])
        self.assertTrue(passed["chain_status"]["left_leg"])
        self.assertTrue(passed["chain_status"]["right_leg"])

        black = pose_reference.evaluate_pose_quality(
            summary, {"complete_black": True}, backend="sdpose"
        )
        self.assertFalse(black["pose_quality_pass"])

    def test_upper_body_requires_arms_and_lower_hip_position(self):
        summary = pose_reference.sdpose_keypoint_summary(_raw_sdpose(upper_body=True))
        result = pose_reference.evaluate_pose_quality(
            summary, {"complete_black": False}, backend="sdpose"
        )
        self.assertFalse(result["full_body_pass"])
        self.assertTrue(result["upper_body_quality_pass"])
        self.assertTrue(result["upper_body_arm_chains"]["left_arm"])
        self.assertTrue(result["upper_body_arm_chains"]["right_arm"])
        profile, reason = pose_reference.classify_pose_reference_profile(result, backend="sdpose")
        self.assertEqual(profile, pose_reference.POSE_REFERENCE_PROFILE_UPPER_BODY)
        self.assertEqual(reason, "upper_body_quality_pass_without_full_body")

    def test_broken_arm_people_zero_and_missing_dwpose_evidence_fail(self):
        broken = pose_reference.evaluate_pose_quality(
            pose_reference.sdpose_keypoint_summary(_raw_sdpose(upper_body=True, broken_left_arm=True)),
            {"complete_black": False},
            backend="sdpose",
        )
        self.assertFalse(broken["pose_quality_pass"])

        no_people = pose_reference.evaluate_pose_quality(
            pose_reference.sdpose_keypoint_summary({"people": []}),
            {"complete_black": False},
            backend="sdpose",
        )
        self.assertFalse(no_people["pose_quality_pass"])

        dwpose_unknown, reason = pose_reference.classify_pose_reference_profile(
            {"people": 1, "complete_black": False, "full_body_pass": True,
             "full_body_leg_chain_evidence": True, "lower_limb_evidence_available": False},
            backend="dwpose",
        )
        self.assertEqual(dwpose_unknown, pose_reference.POSE_REFERENCE_PROFILE_UNKNOWN)
        self.assertEqual(reason, "dwpose_lower_limb_evidence_unavailable")

    def test_preflight_workflow_order_and_controlnet_constants(self):
        dwpose, _, _, _ = pose_reference.build_dwpose_preflight_workflow("reference.png")
        sdpose, _, _, _ = pose_reference.build_sdpose_preflight_workflow("reference.png", "prefix")
        animepose, _, _, _ = pose_reference.build_animepose_preflight_workflow("reference.png")
        self.assertEqual(dwpose["2"]["class_type"], "DWPreprocessor")
        self.assertEqual(sdpose["3"]["class_type"], "SDPoseKeypointExtractor")
        self.assertEqual(animepose["2"]["class_type"], "AnimePoseEstimatorWithMetadata")


class PosePipelineTests(unittest.TestCase):
    def test_comfy_execution_error_keeps_node_details(self):
        message = _pose_history_execution_error(
            {
                "status": {
                    "status_str": "error",
                    "messages": [
                        [
                            "execution_error",
                            {
                                "node_id": "20",
                                "node_type": "SDPoseKeypointExtractor",
                                "exception_message": "model failed",
                                "traceback": "trace",
                            },
                        ]
                    ],
                }
            }
        )
        self.assertIn("node_id=20", message)
        self.assertIn("SDPoseKeypointExtractor", message)
        self.assertIn("model failed", message)

    def test_pipeline_uses_preprocessed_dwpose_then_sdpose(self):
        failed = {"backend": "dwpose", "success": False, "failure_reason": "quality"}
        sdpose = {
            "backend": "sdpose", "success": True,
            "profile": "full_body_reference", "skeleton_path": mock.Mock(is_file=mock.Mock(return_value=True)),
        }
        with (
            mock.patch("services.sd_control._run_pose_backend_preflight", side_effect=[failed, sdpose]) as run,
            mock.patch("services.sd_control._prepare_pose_detection_image", return_value=b"prepared"),
            mock.patch("services.sd_control._upload_comfy_reference_image", return_value="prepared.png"),
        ):
            result = _run_pose_pipeline(
                "reference.png", b"reference",
                options=SdGenerationOptions(prompt="positive", pose_reference_bytes=b"reference", reference_mode="pose"),
                client_id="client",
            )
        self.assertIs(result, sdpose)
        self.assertEqual([call.kwargs["backend"] for call in run.call_args_list], ["dwpose", "dwpose"])

    def test_pipeline_reports_all_quality_failures_in_order(self):
        failed = {"backend": "failed", "success": False, "failure_reason": "quality"}
        with (
            mock.patch("services.sd_control._run_pose_backend_preflight", side_effect=[
                {**failed, "backend": "dwpose"},
                {**failed, "backend": "dwpose"},
                {**failed, "backend": "sdpose"},
                {**failed, "backend": "animepose"},
            ]) as run,
            mock.patch("services.sd_control._prepare_pose_detection_image", return_value=b"prepared"),
            mock.patch("services.sd_control._upload_comfy_reference_image", return_value="prepared.png"),
        ):
            with self.assertRaisesRegex(RuntimeError, "未検証の骨格は使用しません"):
                _run_pose_pipeline(
                    "reference.png", b"reference",
                    options=SdGenerationOptions(prompt="positive", pose_reference_bytes=b"reference", reference_mode="pose"),
                    client_id="client",
                )
        self.assertEqual([call.kwargs["backend"] for call in run.call_args_list], ["dwpose", "dwpose", "sdpose", "animepose"])


class PoseGenerationWiringTests(unittest.TestCase):
    def test_pose_requires_explicit_mode(self):
        with self.assertRaisesRegex(ValueError, "pose reference画像はposeモード"):
            validate_options(
                SdGenerationOptions(
                    prompt="positive",
                    pose_reference_bytes=b"pose",
                    pose_reference_content_type="image/png",
                    reference_mode="none",
                )
            )

    def test_none_does_not_add_controlnet_nodes(self):
        workflow = _workflow()
        result, _, _ = _apply_comfy_options(
            workflow,
            SdGenerationOptions(prompt="positive", negative_prompt="negative", reference_mode="none"),
        )
        self.assertNotIn("ControlNetLoader", {node["class_type"] for node in result.values()})

    def test_pose_adds_fit_and_openpose_controlnet(self):
        workflow = _workflow()
        options = SdGenerationOptions(
            prompt="positive",
            negative_prompt="negative",
            reference_mode="pose",
            pose_reference_bytes=b"not-a-real-png",
            pose_reference_content_type="image/png",
            pose_reference_width=600,
            pose_reference_height=800,
            pose_strength=0.7,
            pose_start_percent=0.0,
            pose_end_percent=0.8,
        )
        result, _, _ = _apply_comfy_options(
            workflow,
            options,
            pose_reference_name="reference.png",
            pose_control_image_name="skeleton.png",
            pose_backend="sdpose",
        )
        classes = [node["class_type"] for node in result.values()]
        self.assertIn("ImageScale", classes)
        self.assertIn("ImageCompositeMasked", classes)
        self.assertIn("ControlNetLoader", classes)
        apply_nodes = [node for node in result.values() if node["class_type"] == "ControlNetApplyAdvanced"]
        self.assertEqual(len(apply_nodes), 1)
        self.assertEqual(apply_nodes[0]["inputs"]["strength"], 0.7)
        self.assertEqual(apply_nodes[0]["inputs"]["end_percent"], 0.8)
        self.assertEqual(result["sampler"]["inputs"]["positive"], [str(next(key for key, node in result.items() if node is apply_nodes[0])), 0])


class PosePanelTests(unittest.TestCase):
    def test_panel_pose_mode_forwards_reuse_and_strength(self):
        captured = []

        async def callback(interaction, request, **kwargs):
            captured.append((interaction, request, kwargs))

        previous = prompt_commands._generate_ai_panel_callback
        prompt_commands._generate_ai_panel_callback = callback
        try:
            asyncio.run(
                prompt_commands.run_generate_ai_from_panel(
                    "interaction", request="pose", model_key="animagine",
                    input_mode="ja_ai", rating="auto", reference_mode="pose",
                    use_last_reference=True, reference_strength=0.7,
                    selection_ai=True, candidate_count=1, size_key="square",
                    composition_preset="none",
                )
            )
        finally:
            prompt_commands._generate_ai_panel_callback = previous
        payload = captured[0][2]
        self.assertEqual(payload["reference_mode"].value, "pose")
        self.assertTrue(payload["use_last_reference"])
        self.assertEqual(payload["reference_strength"], 0.7)

    def test_panel_pose_without_reuse_is_cleared(self):
        captured = []

        async def callback(interaction, request, **kwargs):
            captured.append(kwargs)

        previous = prompt_commands._generate_ai_panel_callback
        prompt_commands._generate_ai_panel_callback = callback
        try:
            asyncio.run(
                prompt_commands.run_generate_ai_from_panel(
                    "interaction", request="ordinary", model_key="animagine",
                    input_mode="ja_ai", rating="auto", reference_mode="pose",
                    use_last_reference=False, reference_strength=0.7,
                    selection_ai=True, candidate_count=1, size_key="square",
                    composition_preset="none",
                )
            )
        finally:
            prompt_commands._generate_ai_panel_callback = previous
        self.assertEqual(captured[0]["reference_mode"].value, "none")
        self.assertFalse(captured[0]["use_last_reference"])
        self.assertIsNone(captured[0]["reference_strength"])


if __name__ == "__main__":
    unittest.main()
