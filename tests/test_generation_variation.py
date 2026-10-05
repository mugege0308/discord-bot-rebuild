from __future__ import annotations

import unittest

from services.generation_variation import (
    VARIATION_AUTO,
    VARIATION_OFF,
    merge_variation_tags,
    plan_variation_slots,
)


class GenerationVariationTests(unittest.TestCase):
    def test_off_keeps_same_empty_slots(self):
        slots = plan_variation_slots(4, VARIATION_OFF)
        self.assertEqual([1, 2, 3, 4], [slot.index for slot in slots])
        self.assertTrue(all(slot.tags == () for slot in slots))

    def test_off_supports_eight_sequential_slots_without_variation_tags(self):
        slots = plan_variation_slots(8, VARIATION_OFF)
        self.assertEqual(list(range(1, 9)), [slot.index for slot in slots])
        self.assertTrue(all(slot.tags == () for slot in slots))

    def test_auto_uses_verified_core_tags_and_distributes_slots(self):
        slots = plan_variation_slots(4, VARIATION_AUTO)
        self.assertEqual(4, len(slots))
        self.assertEqual(4, len({slot.tags[0] for slot in slots}))
        self.assertTrue(all(slot.tags for slot in slots))
        self.assertNotIn("classroom", {tag for slot in slots for tag in slot.tags})

    def test_auto_eight_slots_do_not_repeat_the_first_four_recipes(self):
        slots = plan_variation_slots(8, VARIATION_AUTO)
        self.assertEqual(list(range(1, 9)), [slot.index for slot in slots])
        recipes = [slot.tags for slot in slots]
        self.assertEqual(8, len(set(recipes)))
        self.assertNotEqual(recipes[:4], recipes[4:])

    def test_start_index_continues_auto_variations_at_slots_five_to_eight(self):
        first = plan_variation_slots(4, VARIATION_AUTO)
        continuation = plan_variation_slots(4, VARIATION_AUTO, start_index=4)
        self.assertEqual([5, 6, 7, 8], [slot.index for slot in continuation])
        self.assertTrue(set(slot.tags for slot in first).isdisjoint({slot.tags for slot in continuation}))

    def test_start_index_is_bounded_by_eight_slot_window(self):
        with self.assertRaises(ValueError):
            plan_variation_slots(5, VARIATION_AUTO, start_index=4)

    def test_explicit_categories_are_locked(self):
        slots = plan_variation_slots(
            4,
            VARIATION_AUTO,
            explicit_tags=("from side", "upper body", "closed mouth"),
        )
        all_tags = {tag for slot in slots for tag in slot.tags}
        self.assertFalse({"from side", "from behind", "three-quarter view", "straight-on"} & all_tags)
        self.assertFalse({"portrait", "upper body", "cowboy shot", "full body"} & all_tags)
        self.assertFalse({"smile", "serious", "surprised", "blush"} & all_tags)

    def test_request_text_locks_expression_pose_framing_and_camera(self):
        slots = plan_variation_slots(
            4,
            VARIATION_AUTO,
            request_text="smile, running, full body, from side",
        )
        all_tags = {tag for slot in slots for tag in slot.tags}
        self.assertFalse({"smile", "serious", "surprised", "blush"} & all_tags)
        self.assertFalse({"standing", "sitting", "walking", "kneeling"} & all_tags)
        self.assertFalse({"portrait", "upper body", "cowboy shot", "full body"} & all_tags)
        self.assertFalse({"straight-on", "three-quarter view", "from side", "from behind"} & all_tags)

    def test_pose_reference_locks_pose_and_framing(self):
        slots = plan_variation_slots(4, VARIATION_AUTO, reference_mode="pose")
        all_tags = {tag for slot in slots for tag in slot.tags}
        self.assertFalse({"standing", "sitting", "walking", "kneeling"} & all_tags)
        self.assertFalse({"portrait", "upper body", "cowboy shot", "full body"} & all_tags)

    def test_composition_reference_locks_camera_dimensions(self):
        slots = plan_variation_slots(4, VARIATION_AUTO, reference_mode="composition")
        all_tags = {tag for slot in slots for tag in slot.tags}
        self.assertFalse({"portrait", "upper body", "cowboy shot", "full body"} & all_tags)
        self.assertFalse({"straight-on", "three-quarter view", "from side", "from behind"} & all_tags)

    def test_merge_replaces_only_core_exclusive_group(self):
        merged = merge_variation_tags(
            "1girl, portrait, from side, safe, masterpiece",
            ("full body", "three-quarter view", "serious"),
        )
        self.assertIn("1girl", merged)
        self.assertIn("safe", merged)
        self.assertIn("masterpiece", merged)
        self.assertIn("full body", merged)
        self.assertIn("three-quarter view", merged)
        self.assertIn("serious", merged)
        self.assertNotIn("portrait", merged)
        self.assertNotIn("from side", merged)


if __name__ == "__main__":
    unittest.main()
