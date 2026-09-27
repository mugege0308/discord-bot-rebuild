import json
import unittest
from pathlib import Path
from unittest import mock

from services import shogou_prompt_designer
from services.prompt_tag_dictionary import load_prompt_tag_dictionary
from services.prompt_tag_resolver import (
    apply_prompt_tag_resolution,
    resolve_prompt_tags,
)
from services.prompt_tag_validator import validate_prompt_tag_dictionary_file


ROOT = Path(__file__).resolve().parent.parent
DICTIONARY_PATH = ROOT / "data" / "prompt_tags" / "animagine_xl_4.json"


class PromptTagDictionaryTestCase(unittest.TestCase):
    def test_dictionary_is_valid_small_and_versioned(self):
        report = validate_prompt_tag_dictionary_file(DICTIONARY_PATH)
        self.assertTrue(report.is_usable, report.errors)
        dictionary = load_prompt_tag_dictionary()
        self.assertEqual(dictionary.version, "1.0.0")
        self.assertGreaterEqual(len(dictionary.entries), 30)
        self.assertLessEqual(len(dictionary.entries), 50)
        self.assertEqual(
            {entry.category for entry in dictionary.entries},
            {
                "framing",
                "camera_elevation",
                "camera_horizontal",
                "gaze_direction",
                "basic_pose",
                "expression",
                "dynamic_composition",
            },
        )

    def test_upper_body_from_front_resolves_framing(self):
        result = resolve_prompt_tags("上半身を正面から")
        self.assertIn("upper body", result.resolved_prompt_tags)
        self.assertIn("straight-on", result.resolved_prompt_tags)

    def test_lower_angle_and_looking_back_resolve_together(self):
        result = resolve_prompt_tags("少し下から振り返る")
        self.assertEqual(result.resolved_prompt_tags, ("from below", "looking back"))

    def test_above_and_sitting_resolve_without_inventing_furniture(self):
        result = resolve_prompt_tags("上から見下ろして座っている")
        self.assertEqual(result.resolved_prompt_tags, ("from above", "sitting"))
        positive = apply_prompt_tag_resolution("1girl, safe", result)
        self.assertIn("from above", positive)
        self.assertIn("sitting", positive)
        self.assertNotIn("chair", positive)
        self.assertNotIn("floor", positive)
        self.assertNotIn("crossed legs", positive)

    def test_dynamic_composition_does_not_choose_running_or_jumping(self):
        result = resolve_prompt_tags("動きのある構図")
        self.assertEqual(result.resolved_prompt_tags, ("dynamic pose",))
        positive = apply_prompt_tag_resolution(
            "1girl, safe, running, jumping, from above",
            result,
        )
        self.assertIn("dynamic pose", positive)
        self.assertNotIn("running", positive)
        self.assertNotIn("jumping", positive)

    def test_normal_request_without_composition_tags_keeps_existing_positive(self):
        result = resolve_prompt_tags("赤い髪の女の子が公園にいる")
        original = "1girl, solo, safe, red hair, park"
        self.assertEqual(apply_prompt_tag_resolution(original, result), original)
        self.assertEqual(result.resolved_prompt_tags, ())

    def test_conflicting_elevation_uses_first_explicit_cue(self):
        result = resolve_prompt_tags("上から、下から")
        self.assertEqual(result.resolved_prompt_tags, ("from above",))
        self.assertTrue(result.conflicts)

    def test_panel_preset_is_used_only_when_request_has_no_framing(self):
        result = resolve_prompt_tags("座っている", composition_preset="upper_body")
        self.assertIn("upper body", result.resolved_prompt_tags)
        explicit = resolve_prompt_tags("全身で座っている", composition_preset="upper_body")
        self.assertIn("full body", explicit.resolved_prompt_tags)
        self.assertNotIn("upper body", explicit.resolved_prompt_tags)

    def test_animagine_shogou_output_contains_canonical_resolved_tags(self):
        payload = json.dumps(
            {
                "positive": "1girl, solo, safe, from above, looking back, smile",
                "negative": "lowres, bad anatomy",
                "summary": "構図テスト",
                "assumptions": [],
            }
        )
        with mock.patch.object(shogou_prompt_designer, "_post_chat_completion", return_value=payload):
            result = shogou_prompt_designer.generate_shogou_prompt_result(
                "少し下から振り返って笑っている",
                model_key="animagine",
                requested_rating="safe",
            )

        self.assertEqual(
            result.resolved_prompt_tags,
            ("from below", "looking back", "smile"),
        )
        self.assertIn("from below", result.positive)
        self.assertIn("looking back", result.positive)
        self.assertIn("smile", result.positive)
        self.assertNotIn("from above", result.positive)


if __name__ == "__main__":
    unittest.main()
