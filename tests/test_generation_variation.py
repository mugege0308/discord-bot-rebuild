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

    def test_auto_uses_verified_core_tags_and_distributes_slots(self):
        slots = plan_variation_slots(4, VARIATION_AUTO)
        self.assertEqual(4, len(slots))
        self.assertEqual(4, len({slot.tags[0] for slot in slots}))
        self.assertTrue(all(slot.tags for slot in slots))
        self.assertNotIn("classroom", {tag for slot in slots for tag in slot.tags})

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
